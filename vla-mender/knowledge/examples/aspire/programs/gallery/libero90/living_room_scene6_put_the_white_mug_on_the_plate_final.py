import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def localize_object(rgb, depth, K, E, prompts):
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
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


def average_topdown_grasp_xy(grasp_poses, grasp_scores, E, z_thresh=-0.95, top_k=10):
    """Average xy of top-K scoring top-down grasps for stability."""
    candidates = []
    for i, g in enumerate(grasp_poses):
        Tw = E @ g
        z_axis = Tw[:3, 2]
        if z_axis[2] < z_thresh:
            candidates.append((float(grasp_scores[i]), Tw[0, 3], Tw[1, 3]))
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])
    sel = candidates[:top_k]
    xs = [c[1] for c in sel]
    ys = [c[2] for c in sel]
    return np.array([float(np.mean(xs)), float(np.mean(ys))])


obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]

# Pick: white mug specifically
obj_center, obj_pts, obj_mask = localize_object(
    rgb, depth, K, E,
    ["white mug", "white coffee mug", "white ceramic mug", "plain white mug"]
)
if obj_center is None:
    raise RuntimeError("White mug not found")

# Target: plate
tgt_center, tgt_pts, _ = localize_object(
    rgb, depth, K, E,
    ["plate", "white plate", "dinner plate", "round plate"]
)
if tgt_center is None:
    raise RuntimeError("Plate not found")

mug_top_z = float(obj_pts[:, 2].max())
mug_bottom_z = float(obj_pts[:, 2].min())
surface_z = float(tgt_pts[:, 2].max())

# Compute candidate grasp xys:
# 1) average of top-down grasps from plan_grasp
# 2) body-mean (median + 4cm radius)
# 3) OBB center
# Average these for robustness.
grasp_poses, grasp_scores = plan_grasp(depth, K, obj_mask)
gnet_xy = average_topdown_grasp_xy(grasp_poses, grasp_scores, E, z_thresh=-0.95, top_k=10)
if gnet_xy is None:
    gnet_xy = average_topdown_grasp_xy(grasp_poses, grasp_scores, E, z_thresh=-0.85, top_k=10)

mx, my = float(np.median(obj_pts[:, 0])), float(np.median(obj_pts[:, 1]))
dist = np.sqrt((obj_pts[:, 0] - mx)**2 + (obj_pts[:, 1] - my)**2)
body_pts = obj_pts[dist < 0.04]
body_xy = np.array([float(body_pts[:, 0].mean()), float(body_pts[:, 1].mean())]) if len(body_pts) >= 10 else obj_center[:2]

print(f"OBB center xy: ({obj_center[0]:.3f}, {obj_center[1]:.3f})", flush=True)
print(f"body_xy: {body_xy}", flush=True)
print(f"gnet_xy: {gnet_xy}", flush=True)

# Use body_xy (mean of points within 4cm of XY-median) as grasp xy.
# This filters out handle outliers and gives a stable cylinder-body center.
grasp_xy = body_xy

print(f"Plate center: {tgt_center}", flush=True)
print(f"Mug top_z={mug_top_z:.3f}, bottom_z={mug_bottom_z:.3f}, plate_z={surface_z:.3f}", flush=True)

# Top-down grasp
quat = make_topdown_quat(yaw_deg=0)
grasp_z = mug_top_z - 0.025  # 2.5cm below rim
grasp_pos = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
print(f"Grasp pos: {grasp_pos}, quat: {quat}", flush=True)

# Pre-grasp
open_gripper()
goto_pose(grasp_pos, quat, z_approach=0.18)
goto_pose(grasp_pos, quat)
close_gripper()

# Settle
for _ in range(2):
    get_observation()

# Lift
lift_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.20])
joints = solve_ik(lift_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# Move above plate
above = np.array([tgt_center[0], tgt_center[1], lift_pos[2]])
joints = solve_ik(above.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# Lower so mug bottom is just above plate surface
mug_height = mug_top_z - mug_bottom_z
release_z = surface_z + mug_height + 0.005
release_pos = np.array([tgt_center[0], tgt_center[1], release_z])
joints = solve_ik(release_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

open_gripper()

# Allow physics to settle
for _ in range(3):
    get_observation()

# Retreat
retreat = np.array([tgt_center[0], tgt_center[1], release_z + 0.18])
joints = solve_ik(retreat.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

for _ in range(5):
    get_observation()
