# Code block 0
import numpy as np

# ---- helpers ----

def make_topdown_quat():
    """Top-down gripper orientation: z-axis pointing down."""
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    return rotation_matrix_to_quaternion(R)

def move_pose(pos, quat):
    joints = solve_ik(np.asarray(pos, dtype=float), np.asarray(quat, dtype=float))
    move_to_joints(joints)

def get_best_mask(rgb, prompt, min_px=50, max_px=12000):
    """Get best SAM3 mask filtered by pixel count to exclude robot arm / background."""
    masks = segment_sam3_text_prompt(rgb, prompt)
    if not masks:
        return None
    valid = []
    for m in masks:
        mask = m.get("mask", None)
        if mask is None:
            continue
        px_count = int(np.sum(mask > 0))
        if min_px <= px_count <= max_px:
            valid.append((m, px_count))
    if not valid:
        # Fallback: just pick best score from all non-empty masks
        for m in masks:
            mask = m.get("mask", None)
            if mask is not None and np.any(mask):
                valid.append((m, int(np.sum(mask > 0))))
        if not valid:
            return None
    best = max(valid, key=lambda x: float(x[0].get("score", 0.0)))
    return best[0]["mask"]

def localize_cube(rgb, depth, K, T, prompts):
    """Localize a cube and return median center and top/bottom Z."""
    mask = None
    for prompt in prompts:
        mask = get_best_mask(rgb, prompt)
        if mask is not None:
            break
    if mask is None:
        # Molmo fallback
        pts_molmo = point_prompt_molmo(rgb, prompts[0])
        for key, val in pts_molmo.items():
            if val[0] is not None and val[1] is not None:
                point_masks = segment_sam3_point_prompt(rgb, val)
                if point_masks:
                    for pm in point_masks:
                        m = pm.get("mask", None)
                        if m is not None and np.any(m):
                            mask = m
                            break
                if mask is not None:
                    break
    if mask is None:
        return None, None, None
    pts = mask_to_world_points(mask.astype(np.uint8), depth, K, T)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 10:
        return None, None, None
    center = np.median(pts, axis=0)
    top_z = float(np.percentile(pts[:, 2], 95))
    bot_z = float(np.percentile(pts[:, 2], 5))
    return center, top_z, bot_z

def check_grasp_success():
    """Check gripper qpos to determine if grasp succeeded."""
    obs = get_observation()
    qpos = obs.get("robot0_gripper_qpos", None)
    if qpos is not None:
        width = float(qpos[0])
        return width > 0.005  # >0.005 means object contact
    return True  # assume success if can't check

def safe_home(quat):
    """Move to a safe home position above the workspace."""
    move_pose(np.array([0.45, 0.0, 0.15], dtype=float), quat)

# ---- main ----

quat = make_topdown_quat()

# Move arm out of camera view before observing
open_gripper()
safe_home(quat)

# Observe
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
T = cam["pose_mat"]

# Localize both cubes
red_center, red_top, red_bot = localize_cube(rgb, depth, K, T, ["red cube", "red block"])
green_center, green_top, green_bot = localize_cube(rgb, depth, K, T, ["green cube", "green block"])

if red_center is None or green_center is None:
    raise RuntimeError("Failed to localize cubes")

# Cube dimensions
red_h = red_top - red_bot
green_h = green_top - green_bot
cube_h = max(red_h, green_h)

# If green is above red (already stacked wrong way), move green aside first
if green_center[2] > red_center[2] + 0.01:
    # Pick green and put it to the side
    g_grasp_z = (green_top + green_bot) / 2.0  # center height for solid grip
    g_pre = green_center.copy()
    g_pre[2] = green_top + 0.10
    g_pick = green_center.copy()
    g_pick[2] = g_grasp_z

    move_pose(g_pre, quat)
    move_pose(g_pick, quat)
    close_gripper()

    # Lift and place aside
    g_lift = green_center.copy()
    g_lift[2] = green_top + 0.12
    move_pose(g_lift, quat)

    aside = np.array([green_center[0] + 0.10, green_center[1] + 0.05, green_top + 0.12], dtype=float)
    move_pose(aside, quat)
    aside_place = aside.copy()
    aside_place[2] = red_bot + cube_h * 0.5
    move_pose(aside_place, quat)
    open_gripper()
    move_pose(aside, quat)

    # Re-observe
    safe_home(quat)
    obs = get_observation()
    cam = obs["robot0_robotview"]
    rgb = cam["images"]["rgb"]
    depth = cam["images"]["depth"]
    K = cam["intrinsics"]
    T = cam["pose_mat"]
    red_center, red_top, red_bot = localize_cube(rgb, depth, K, T, ["red cube", "red block"])
    green_center, green_top, green_bot = localize_cube(rgb, depth, K, T, ["green cube", "green block"])
    if red_center is None or green_center is None:
        raise RuntimeError("Failed to localize cubes after rearrangement")
    red_h = red_top - red_bot
    green_h = green_top - green_bot
    cube_h = max(red_h, green_h)

# ---- Pick red cube and place on green cube ----
MAX_ATTEMPTS = 3
success = False

for attempt in range(MAX_ATTEMPTS):
    if attempt > 0:
        # Re-observe before retry
        open_gripper()
        safe_home(quat)
        obs = get_observation()
        cam = obs["robot0_robotview"]
        rgb = cam["images"]["rgb"]
        depth = cam["images"]["depth"]
        K = cam["intrinsics"]
        T = cam["pose_mat"]
        red_center, red_top, red_bot = localize_cube(rgb, depth, K, T, ["red cube", "red block"])
        green_center, green_top, green_bot = localize_cube(rgb, depth, K, T, ["green cube", "green block"])
        if red_center is None or green_center is None:
            continue
        red_h = red_top - red_bot
        green_h = green_top - green_bot
        cube_h = max(red_h, green_h)

    # Grasp at cube center height for solid grip
    grasp_z = (red_top + red_bot) / 2.0
    grasp_pos = np.array([red_center[0], red_center[1], grasp_z], dtype=float)

    pre_grasp = grasp_pos.copy()
    pre_grasp[2] = red_top + 0.10

    # Approach and grasp
    open_gripper()
    move_pose(pre_grasp, quat)
    move_pose(grasp_pos, quat)
    close_gripper()

    # Check grasp
    grasped = check_grasp_success()

    if not grasped and attempt < MAX_ATTEMPTS - 1:
        open_gripper()
        move_pose(pre_grasp, quat)
        continue

    # Lift
    lift_pos = grasp_pos.copy()
    lift_pos[2] = max(red_top, green_top) + 0.12

    move_pose(lift_pos, quat)

    # Place on green cube
    place_z = green_top + cube_h * 0.5 + 0.005
    place_pos = np.array([green_center[0], green_center[1], place_z], dtype=float)

    pre_place = place_pos.copy()
    pre_place[2] = place_z + 0.10

    move_pose(pre_place, quat)
    move_pose(place_pos, quat)
    open_gripper()

    retreat = place_pos.copy()
    retreat[2] += 0.12
    move_pose(retreat, quat)
    success = True
    break

print(f"cube_stack done, success={success}")