"""Pick the MIDDLE black bowl (sorted by world-x), place on cabinet top."""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def collect_bowl_candidates(rgb, depth_img, K, E):
    """Collect deduplicated black-bowl candidates on the table."""
    candidates = []
    for prompt in ("black bowl", "bowl", "dark bowl"):
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:12]:
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 200:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c, e = obb["center"], obb["extent"]
            # Bowl-on-table filters
            if c[2] > 0.10 or c[0] < 0.30 or c[0] > 1.0:
                continue
            if max(e[0], e[1]) > 0.18 or max(e[0], e[1]) < 0.06:
                continue
            if e[2] > 0.12 or e[2] < 0.02:
                continue
            candidates.append((m.get("score", 0.0), c, pts, mask))
    # Dedupe by 3D center
    deduped = []
    for s, c, pts, mask in sorted(candidates, key=lambda x: -x[0]):
        is_dup = False
        for _, c2, _, _ in deduped:
            if np.linalg.norm(c[:2] - c2[:2]) < 0.06:
                is_dup = True
                break
        if not is_dup:
            deduped.append((s, c, pts, mask))
    return deduped


def select_cabinet_top(rgb, depth_img, K, E):
    candidates = []
    for prompt in ("cabinet top", "top of cabinet", "wooden cabinet top"):
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:8]:
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 100:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c, e = obb["center"], obb["extent"]
            if c[0] < 0.3 or c[0] > 1.0:
                continue
            if c[2] < 0.10:
                continue
            if e[0] > 0.5 or e[1] > 0.5 or e[2] > 0.10:
                continue
            candidates.append((m.get("score", 0.0), c, pts, mask))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda x: -x[0])
    _, c, pts, mask = candidates[0]
    return c, pts, mask


# ---------------- Main ----------------
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]

bowls = collect_bowl_candidates(rgb, depth_img, K, E)
print(f"[task] found {len(bowls)} bowl candidates", flush=True)
for s, c, pts, _ in bowls:
    print(f"  bowl: score={s:.3f} center=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})", flush=True)

if len(bowls) < 1:
    raise RuntimeError("No bowls found")

# Pick the MIDDLE bowl by sorting all candidates by world-x and selecting the median.
bowls_by_x = sorted(bowls, key=lambda b: b[1][0])
mid_idx = len(bowls_by_x) // 2
if len(bowls_by_x) >= 3:
    mid_idx = 1  # middle of 3
elif len(bowls_by_x) == 2:
    mid_idx = 0
_, bowl_center, bowl_pts, bowl_mask = bowls_by_x[mid_idx]
bowl_obb = get_oriented_bounding_box_from_3d_points(bowl_pts)
bowl_top_z = float(bowl_pts[:, 2].max())
bowl_height = float(bowl_obb["extent"][2])
print(f"[task] picked MIDDLE bowl: center=({bowl_center[0]:.3f},{bowl_center[1]:.3f},{bowl_center[2]:.3f})", flush=True)

tgt_center, tgt_pts, _ = select_cabinet_top(rgb, depth_img, K, E)
if tgt_center is None:
    raise RuntimeError("Cabinet top not found")
surface_z = float(tgt_pts[:, 2].max())
print(f"[task] cabinet_top: center=({tgt_center[0]:.3f},{tgt_center[1]:.3f},{tgt_center[2]:.3f}) surface_z={surface_z:.3f}", flush=True)

# ---------------- Grasp the bowl ----------------
grasp_poses, grasp_scores = plan_grasp(depth, K, bowl_mask)
best_grasp, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
if best_grasp is None:
    best_grasp = E @ grasp_poses[grasp_scores.argmax()]
grasp_pos, grasp_quat = decompose_transform(best_grasp)
grasp_pos = np.asarray(grasp_pos, dtype=float)
grasp_quat = np.asarray(grasp_quat, dtype=float)

# Snap XY to bowl center if planner is far off (rim grasp)
bowl_xy = np.array([bowl_center[0], bowl_center[1]])
dist_xy = np.linalg.norm(grasp_pos[:2] - bowl_xy)
bowl_radius = max(float(bowl_obb["extent"][0]), float(bowl_obb["extent"][1])) / 2.0
if dist_xy > 0.5 * bowl_radius:
    print(f"[task] snapping grasp xy {grasp_pos[:2]} -> {bowl_xy} (off by {dist_xy:.3f})", flush=True)
    grasp_pos[0] = bowl_xy[0]
    grasp_pos[1] = bowl_xy[1]

# Clamp grasp z just under bowl rim
target_grasp_z = bowl_top_z - 0.005
if grasp_pos[2] > bowl_top_z + 0.02 or grasp_pos[2] < bowl_top_z - bowl_height:
    grasp_pos[2] = target_grasp_z

# Top-down quat fallback
R_default = Rotation.from_quat([grasp_quat[1], grasp_quat[2], grasp_quat[3], grasp_quat[0]]).as_matrix()
gripper_z_world = R_default @ np.array([0, 0, 1])
if gripper_z_world[2] > -0.7:
    print(f"[task] forcing top-down quat (z_world_z={gripper_z_world[2]:.2f})", flush=True)
    grasp_quat = make_topdown_quat(0)

open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.15)
goto_pose(grasp_pos, grasp_quat)
close_gripper()

# Lift well above the cabinet top
lift_z = max(grasp_pos[2] + 0.20, surface_z + 0.18)
lift_pos = np.array([grasp_pos[0], grasp_pos[1], lift_z])
joints = solve_ik(lift_pos.tolist(), grasp_quat.tolist())
if joints is not None:
    move_to_joints(joints)

# Move above target
above = np.array([tgt_center[0], tgt_center[1], lift_z])
joints = solve_ik(above.tolist(), grasp_quat.tolist())
if joints is not None:
    move_to_joints(joints)

# Lower to release height
release_pos = np.array([tgt_center[0], tgt_center[1], surface_z + 0.05])
joints = solve_ik(release_pos.tolist(), grasp_quat.tolist())
if joints is not None:
    move_to_joints(joints)

open_gripper()

# Retreat upward
retreat_pos = np.array([tgt_center[0], tgt_center[1], surface_z + 0.20])
joints = solve_ik(retreat_pos.tolist(), grasp_quat.tolist())
if joints is not None:
    move_to_joints(joints)

for _ in range(3):
    get_observation()
