import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


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


def average_topdown_grasp_xy(grasp_poses, grasp_scores, E, z_thresh=-0.95, top_k=10):
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


# ---- Perception ----
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
E = cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

# White mug — use specific prompts; "mug" alone may match the other (red/decorated) mugs
obj_center, obj_pts, obj_mask = localize_object(
    rgb, depth_img, K, E,
    ["white ceramic mug", "white coffee mug", "white mug", "plain white mug"],
)
if obj_center is None:
    raise RuntimeError("White mug not found")

# Left plate — direct spatial prompt validated to work on this scene
tgt_center, tgt_pts, _ = localize_object(
    rgb, depth_img, K, E,
    ["left plate"],
)
# Fallback: if "left plate" fails, take all plate masks and pick the one with smaller world-y
if tgt_center is None:
    plate_masks = segment_sam3_text_prompt(rgb, "plate")
    cands = []
    for m in plate_masks[:10]:
        pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
        if pts is None or len(pts) < 10:
            continue
        c = get_oriented_bounding_box_from_3d_points(pts)["center"]
        if c[2] > 0.10:  # filter non-table-level results
            continue
        # Dedupe by 3D distance
        if any(np.linalg.norm(c[:2] - prev[0][:2]) < 0.06 for prev in cands):
            continue
        cands.append((c, pts))
    if not cands:
        raise RuntimeError("No plates found")
    # Left plate = smaller (more negative) world-y
    cands.sort(key=lambda x: x[0][1])
    tgt_center, tgt_pts = cands[0]

print(f"White mug center: {obj_center}", flush=True)
print(f"Left plate center: {tgt_center}", flush=True)

mug_top_z = float(obj_pts[:, 2].max())
mug_bottom_z = float(obj_pts[:, 2].min())
surface_z = float(tgt_pts[:, 2].max())

# Body-XY centroid (iterative): mug handle biases the OBB; iteratively shrink to body center.
def body_xy_centroid(pts, radius=0.04, iters=3):
    xy = pts[:, :2]
    cx, cy = float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))
    for _ in range(iters):
        dist = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
        inliers = pts[dist < radius]
        if len(inliers) < 5:
            break
        cx, cy = float(inliers[:, 0].mean()), float(inliers[:, 1].mean())
    return cx, cy

bx, by = body_xy_centroid(obj_pts, radius=0.04, iters=3)
body_xy = np.array([bx, by])

grasp_poses, grasp_scores = plan_grasp(depth, K, obj_mask)
gnet_xy = average_topdown_grasp_xy(grasp_poses, grasp_scores, E, z_thresh=-0.95, top_k=10)
if gnet_xy is None:
    gnet_xy = average_topdown_grasp_xy(grasp_poses, grasp_scores, E, z_thresh=-0.85, top_k=10)

print(f"OBB center xy: ({obj_center[0]:.3f}, {obj_center[1]:.3f})", flush=True)
print(f"body_xy: {body_xy}", flush=True)
print(f"gnet_xy: {gnet_xy}", flush=True)

# Use body_xy (handle-filtered iterative centroid) as the grasp XY.
grasp_xy = body_xy

# Top-down grasp at top-z minus 2.5cm (validated for white mug)
quat = make_topdown_quat(yaw_deg=0)
grasp_z = mug_top_z - 0.025
grasp_pos = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
print(f"Mug top_z={mug_top_z:.3f}, bottom_z={mug_bottom_z:.3f}, plate_z={surface_z:.3f}", flush=True)
print(f"Grasp pos: {grasp_pos}", flush=True)

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

# Lower so mug bottom lands just above plate surface (slight drop for clean release)
mug_height = mug_top_z - mug_bottom_z
release_z = surface_z + mug_height + 0.005
release_pos = np.array([tgt_center[0], tgt_center[1], release_z])
joints = solve_ik(release_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

open_gripper()

for _ in range(3):
    get_observation()

# Retreat
retreat = np.array([tgt_center[0], tgt_center[1], release_z + 0.18])
joints = solve_ik(retreat.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

for _ in range(5):
    get_observation()
