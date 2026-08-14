# Code block 0
import numpy as np

# ==============================================================================
# cube_lifting fix code - robust single-attempt grasp with retry
# ==============================================================================

def make_topdown_quat():
    """Create a top-down grasp quaternion (gripper pointing straight down)."""
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    return rotation_matrix_to_quaternion(R)

def localize_cube(rgb, depth, K, cam_to_world):
    """Localize the red cube using SAM3 with fallback to Molmo."""
    # Try SAM3 text prompts
    for prompt in ["red cube", "red block", "cube"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if masks:
            # Filter by pixel count to exclude robot arm (>12K) and noise (<50)
            valid = [m for m in masks if 50 < m["mask"].astype(np.int32).sum() < 12000]
            if valid:
                best = max(valid, key=lambda m: m.get("score", 0.0))
                return best["mask"].astype(np.uint8)

    # Fallback: Molmo point prompt -> SAM3 point prompt
    pts = point_prompt_molmo(rgb, "the red cube")
    if isinstance(pts, dict):
        for v in pts.values():
            if v[0] is not None and v[1] is not None:
                masks = segment_sam3_point_prompt(rgb, point_coords=(float(v[0]), float(v[1])))
                if masks:
                    valid = [m for m in masks if 50 < m["mask"].astype(np.int32).sum() < 12000]
                    if valid:
                        best = max(valid, key=lambda m: m.get("score", 0.0))
                        return best["mask"].astype(np.uint8)

    return None

def get_cube_pose(mask, depth, K, cam_to_world):
    """Get cube center position from mask."""
    world_points = mask_to_world_points(mask, depth, K, cam_to_world)
    if world_points.shape[0] == 0:
        return None, None, None

    # Use median for robustness against outliers
    center = np.median(world_points, axis=0)
    top_z = np.percentile(world_points[:, 2], 95)  # 95th percentile for robustness
    bot_z = np.percentile(world_points[:, 2], 5)

    return center, top_z, bot_z

def attempt_grasp(rgb, depth, K, cam_to_world, attempt_num=0):
    """Single grasp attempt. Returns True if gripper closed on object."""
    print(f"\n--- Grasp attempt {attempt_num + 1} ---", flush=True)

    # Localize cube
    mask = localize_cube(rgb, depth, K, cam_to_world)
    if mask is None:
        print("Failed to localize cube!", flush=True)
        return False

    print(f"Mask pixels: {mask.sum()}", flush=True)

    # Get cube 3D position
    center, top_z, bot_z = get_cube_pose(mask, depth, K, cam_to_world)
    if center is None:
        print("Failed to get cube 3D position!", flush=True)
        return False

    print(f"Cube center: {center}, top_z: {top_z:.4f}, bot_z: {bot_z:.4f}", flush=True)

    # Plan grasp using GraspNet
    grasps_cam, scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask.astype(np.int32))

    grasp_pos = None
    grasp_quat = None

    if grasps_cam is not None and len(grasps_cam) > 0:
        best_world_T, best_score = select_top_down_grasp(grasps_cam, scores, cam_to_world, vertical_threshold=0.5)
        if best_world_T is not None:
            gpos, gquat = decompose_transform(best_world_T)
            print(f"GraspNet top-down: pos={gpos}, score={best_score:.4f}", flush=True)
            # Use GraspNet XY but override Z to cube center height
            grasp_pos = gpos.copy()
            grasp_quat = gquat.copy()
        else:
            # Use highest scoring grasp
            idx = int(np.argmax(scores))
            best_world_T = cam_to_world @ grasps_cam[idx]
            gpos, gquat = decompose_transform(best_world_T)
            print(f"GraspNet best-score: pos={gpos}, score={scores[idx]:.4f}", flush=True)
            grasp_pos = gpos.copy()
            grasp_quat = gquat.copy()

    if grasp_pos is None:
        # Fallback: use centroid with top-down orientation
        print("Using centroid fallback for grasp", flush=True)
        grasp_pos = center.copy()
        grasp_quat = make_topdown_quat()

    # Override grasp Z to cube center height for reliable grip
    cube_center_z = (top_z + bot_z) / 2.0
    grasp_pos[2] = cube_center_z
    print(f"Final grasp pos (z overridden to center): {grasp_pos}", flush=True)

    # Pre-grasp: 12cm above grasp position
    pregrasp_pos = grasp_pos.copy()
    pregrasp_pos[2] += 0.12

    # Execute grasp sequence
    open_gripper()

    # Move to pre-grasp
    joints = solve_ik(pregrasp_pos, grasp_quat)
    move_to_joints(joints)
    print("At pre-grasp", flush=True)

    # Descend to grasp with fine steps
    for p in interpolate_segment(pregrasp_pos, grasp_pos, step=0.02):
        j = solve_ik(np.array(p, dtype=np.float64), grasp_quat)
        move_to_joints(j)
    print("At grasp position", flush=True)

    # Close gripper
    close_gripper()

    # Check if we grasped by reading gripper state
    obs_after = get_observation()
    gripper_qpos = obs_after.get("robot0_gripper_qpos", None)
    if gripper_qpos is not None:
        gw = gripper_qpos[0]
        print(f"Gripper qpos after close: {gw:.4f}", flush=True)
        if gw < 0.003:
            print("Air grasp detected (gripper fully closed)", flush=True)
            return False

    return True

def lift_cube(grasp_pos, grasp_quat, lift_height=0.25):
    """Lift the cube to specified height above grasp position."""
    lift_pos = grasp_pos.copy()
    lift_pos[2] += lift_height

    for p in interpolate_segment(grasp_pos, lift_pos, step=0.02):
        j = solve_ik(np.array(p, dtype=np.float64), grasp_quat)
        move_to_joints(j)

    print(f"Lifted to z={lift_pos[2]:.4f}", flush=True)
    return lift_pos

# ==============================================================================
# Main execution
# ==============================================================================

# Step 1: Move arm to safe position for clear camera view
home_joints = np.array([0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78], dtype=np.float64)
open_gripper()
move_to_joints(home_joints)

# Step 2: Get observation
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_to_world = cam["pose_mat"]

# Step 3: Attempt grasp (with up to 2 retries)
MAX_ATTEMPTS = 3
success = False
last_grasp_pos = None
last_grasp_quat = None

for attempt in range(MAX_ATTEMPTS):
    # Localize cube
    mask = localize_cube(rgb, depth, K, cam_to_world)
    if mask is None:
        print(f"Attempt {attempt+1}: Failed to localize cube", flush=True)
        continue

    center, top_z, bot_z = get_cube_pose(mask, depth, K, cam_to_world)
    if center is None:
        print(f"Attempt {attempt+1}: Failed to get cube position", flush=True)
        continue

    print(f"Attempt {attempt+1}: Cube center={center}, top_z={top_z:.4f}, bot_z={bot_z:.4f}", flush=True)

    # Plan grasp
    grasps_cam, scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask.astype(np.int32))

    grasp_pos = None
    grasp_quat = None

    if grasps_cam is not None and len(grasps_cam) > 0:
        best_world_T, best_score = select_top_down_grasp(grasps_cam, scores, cam_to_world, vertical_threshold=0.5)
        if best_world_T is not None:
            grasp_pos, grasp_quat = decompose_transform(best_world_T)
            print(f"Top-down grasp found, score={best_score:.4f}", flush=True)
        else:
            idx = int(np.argmax(scores))
            best_world_T = cam_to_world @ grasps_cam[idx]
            grasp_pos, grasp_quat = decompose_transform(best_world_T)
            print(f"Using best-score grasp, score={scores[idx]:.4f}", flush=True)

    if grasp_pos is None:
        grasp_pos = center.copy()
        grasp_quat = make_topdown_quat()
        print("Using centroid + top-down fallback", flush=True)

    # Override Z to cube center height
    cube_center_z = (top_z + bot_z) / 2.0
    grasp_pos[2] = cube_center_z
    last_grasp_pos = grasp_pos.copy()
    last_grasp_quat = grasp_quat.copy()

    print(f"Grasp pos: {grasp_pos}", flush=True)

    # Pre-grasp position
    pregrasp_pos = grasp_pos.copy()
    pregrasp_pos[2] += 0.12

    # Execute
    open_gripper()

    j = solve_ik(pregrasp_pos, grasp_quat)
    move_to_joints(j)

    for p in interpolate_segment(pregrasp_pos, grasp_pos, step=0.02):
        j = solve_ik(np.array(p, dtype=np.float64), grasp_quat)
        move_to_joints(j)

    close_gripper()

    # Check grasp quality
    obs_check = get_observation()
    gripper_qpos = obs_check.get("robot0_gripper_qpos", None)
    if gripper_qpos is not None:
        gw = gripper_qpos[0]
        print(f"Gripper qpos: {gw:.4f}", flush=True)
        if gw < 0.003:
            print("Air grasp - retrying", flush=True)
            open_gripper()
            # Move back up
            j = solve_ik(pregrasp_pos, grasp_quat)
            move_to_joints(j)
            # Re-observe
            move_to_joints(home_joints)
            obs = get_observation()
            cam = obs["robot0_robotview"]
            rgb = cam["images"]["rgb"]
            depth = cam["images"]["depth"]
            K = cam["intrinsics"]
            cam_to_world = cam["pose_mat"]
            continue

    # Grasp seems OK - lift
    lift_pos = grasp_pos.copy()
    lift_pos[2] += 0.25

    for p in interpolate_segment(grasp_pos, lift_pos, step=0.02):
        j = solve_ik(np.array(p, dtype=np.float64), grasp_quat)
        move_to_joints(j)

    print(f"Lifted to z={lift_pos[2]:.4f}", flush=True)
    success = True
    break

if not success and last_grasp_pos is not None:
    # Last resort: just try to lift from wherever we are
    print("All attempts may have failed, doing a final lift", flush=True)
    close_gripper()
    lift_pos = last_grasp_pos.copy()
    lift_pos[2] += 0.25
    for p in interpolate_segment(last_grasp_pos, lift_pos, step=0.02):
        j = solve_ik(np.array(p, dtype=np.float64), last_grasp_quat)
        move_to_joints(j)

print("cube_lifting complete", flush=True)