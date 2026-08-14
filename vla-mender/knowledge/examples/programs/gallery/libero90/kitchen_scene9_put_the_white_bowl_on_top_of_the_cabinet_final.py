import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


def localize_object(rgb, depth_img, K, E, prompts):
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        best = max(masks, key=lambda d: d["score"])
        mask = best["mask"].astype(np.uint8)
        pts = mask_to_world_points(mask, depth_img, K, E)
        if pts is None or len(pts) < 10:
            continue
        center = get_oriented_bounding_box_from_3d_points(pts)["center"]
        return center, pts, mask
    return None, None, None


def select_white_bowl_mask(rgb, depth_img, K, E):
    """Find the white bowl on the table.
    Filters: on-table (center.x in [0.3,1.0], z<0.10), bowl-sized (ext_xy 0.04-0.20)."""
    candidates = []
    for prompt in ("white bowl", "white ceramic bowl", "ceramic bowl", "bowl"):
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:6]:
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 50:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            center, extent = obb["center"], obb["extent"]
            if center[0] < 0.3 or center[0] > 1.0:
                continue
            if center[2] > 0.15:
                continue
            if max(extent[0], extent[1]) > 0.20:
                continue
            if max(extent[0], extent[1]) < 0.04:
                continue
            candidates.append((m["score"], center, pts, mask))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, center, pts, mask = candidates[0]
    return center, pts, mask


def select_cabinet_top_mask(rgb, depth_img, K, E):
    """Find the on-table cabinet's top horizontal surface.
    Returns (target_xy[3], top_pts, surface_z)."""
    candidates = []
    for prompt in ("cabinet top", "top of cabinet", "wooden cabinet top"):
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:8]:
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 100:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            center, extent = obb["center"], obb["extent"]
            # Reject background wall cabinets (x<0)
            if center[0] < 0.3 or center[0] > 1.0:
                continue
            # Cabinet top is elevated
            if center[2] < 0.10:
                continue
            # Reject huge masks
            if extent[0] > 0.6 or extent[1] > 0.6:
                continue
            # Allow front-face leak; we'll z-filter inside
            if extent[2] > 0.25:
                continue
            candidates.append((m["score"], center, pts, mask))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, center, pts, mask = candidates[0]
    # Extract top-only points (top 30% by z)
    z_sorted = np.sort(pts[:, 2])
    z_thresh = z_sorted[int(len(z_sorted) * 0.70)]
    top_pts = pts[pts[:, 2] >= z_thresh]
    if len(top_pts) < 50:
        top_pts = pts
    surface_z = float(top_pts[:, 2].max())
    target_xy = np.array([
        (np.percentile(top_pts[:, 0], 5) + np.percentile(top_pts[:, 0], 95)) / 2.0,
        (np.percentile(top_pts[:, 1], 5) + np.percentile(top_pts[:, 1], 95)) / 2.0,
        surface_z,
    ])
    return target_xy, top_pts, surface_z


def step_to(target_pos, quat, n_steps=4):
    """Move to target_pos in n_steps interpolated IK waypoints."""
    obs = get_observation()
    current = np.array(obs['robot_cartesian_pos'][:3])
    for k in range(1, n_steps + 1):
        wp = current + (target_pos - current) * (k / n_steps)
        j = solve_ik(wp.tolist(), quat.tolist())
        if j is not None:
            move_to_joints(j)


# ---------------- Physics settle ----------------
goto_home_joint_position()
for _ in range(3):
    open_gripper()
    close_gripper()
open_gripper()

# ---------------- Observation ----------------
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]

bowl_center, bowl_pts, bowl_mask = select_white_bowl_mask(rgb, depth_img, K, E)
if bowl_center is None:
    bowl_center, bowl_pts, bowl_mask = localize_object(
        rgb, depth_img, K, E, ["white bowl", "ceramic bowl", "bowl"])
if bowl_center is None:
    raise RuntimeError("White bowl not found")

bowl_obb = get_oriented_bounding_box_from_3d_points(bowl_pts)
bowl_top_z = float(bowl_pts[:, 2].max())
bowl_height = float(bowl_obb["extent"][2])

target_xy, tgt_pts, surface_z = select_cabinet_top_mask(rgb, depth_img, K, E)
if target_xy is None:
    raise RuntimeError("Cabinet top not found")

print(f"[task] bowl_center={bowl_center} bowl_top_z={bowl_top_z:.3f} "
      f"bowl_h={bowl_height:.3f} target_xy={target_xy} surface_z={surface_z:.3f}",
      flush=True)

# ---------------- Grasp the bowl ----------------
grasp_poses, grasp_scores = plan_grasp(depth, K, bowl_mask)
best_grasp, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
if best_grasp is None:
    best_grasp = E @ grasp_poses[grasp_scores.argmax()]
grasp_pos, grasp_quat = decompose_transform(best_grasp)
grasp_pos = np.asarray(grasp_pos, dtype=float)
grasp_quat = np.asarray(grasp_quat, dtype=float)

# Always snap XY to OBB center for a small circular bowl.
bowl_xy = np.array([bowl_center[0], bowl_center[1]])
grasp_pos[0] = bowl_xy[0]
grasp_pos[1] = bowl_xy[1]

# Just below the bowl rim — fingers grip the rim from outside.
target_grasp_z = bowl_top_z - 0.005
grasp_pos[2] = target_grasp_z

# Top-down quat at yaw=0 (matches KITCHEN_SCENE1).
grasp_quat = make_topdown_quat(0)

print(f"[task] grasp_pos={grasp_pos.round(3)}", flush=True)

open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.15)
goto_pose(grasp_pos, grasp_quat)
close_gripper()
obs2 = get_observation()
grip_w = float(obs2["robot_cartesian_pos"][7])
print(f"[task] grip_width: {grip_w:.3f}", flush=True)

# ---------------- Lift HIGH (clear pan), step_to lateral, descend in steps ----------------
# Lift z = max(grasp+0.30, surface+0.25, 0.50) — high lift to reduce bowl swing.
lift_z = max(grasp_pos[2] + 0.30, surface_z + 0.25, 0.50)

lift_pos = np.array([grasp_pos[0], grasp_pos[1], lift_z])
joints = solve_ik(lift_pos.tolist(), grasp_quat.tolist())
if joints is not None:
    move_to_joints(joints)

# Slow lateral via step_to (interpolated for stability).
above = np.array([target_xy[0], target_xy[1], lift_z])
step_to(above, grasp_quat, n_steps=4)

# Lower to release height in 3 steps for smooth descent.
release_z = surface_z + bowl_height + 0.005
release_pos = np.array([target_xy[0], target_xy[1], release_z])
step_to(release_pos, grasp_quat, n_steps=3)

open_gripper()

# Retreat upward to clear placed bowl.
retreat_pos = np.array([target_xy[0], target_xy[1], release_z + 0.20])
joints = solve_ik(retreat_pos.tolist(), grasp_quat.tolist())
if joints is not None:
    move_to_joints(joints)

for _ in range(5):
    get_observation()
