"""
LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray
Task type: pick-and-place

Pick: ketchup bottle (tall, orange/yellow with white cap, "Tomato Ketchup" label).
       In the initial agentview frame, the bottle is OCCLUDED by the robot arm
       in HOME position. Must move arm aside before localization.
       Best SAM3 prompts: "ketchup bottle" (0.88), "orange bottle" (0.94), "tall bottle" (0.93).
Place: wooden tray on the right side of the table.
       Best SAM3 prompt: "tray" (score 0.77).

Scene caveats:
- 4 visually similar objects in the foreground (2 cans, butter, cream cheese)
  can confuse generic prompts; bottle is at world-y < -0.1 (back-left).
- Once the arm moves to a side position, the bottle becomes visible.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


def localize_object(rgb, depth, K, E, prompts, min_score=0.0):
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        best = max(masks, key=lambda d: d["score"])
        if best["score"] < min_score:
            continue
        mask = best["mask"].astype(np.uint8)
        pts = mask_to_world_points(mask, depth_img, K, E)
        if pts is None or len(pts) < 10:
            continue
        center = get_oriented_bounding_box_from_3d_points(pts)["center"]
        return center, pts, mask, best["score"]
    return None, None, None, 0.0


# === Step 1: Move arm aside to deocclude the scene ===
quat_topdown = make_topdown_quat(0)
side_pos = np.array([0.4, -0.4, 0.3])
joints = solve_ik(side_pos.tolist(), quat_topdown.tolist())
if joints is not None:
    move_to_joints(joints)
else:
    # Fallback: try a different side position
    for fallback_pos in [[0.3, -0.4, 0.35], [0.3, 0.4, 0.35], [0.5, -0.3, 0.3]]:
        joints = solve_ik(fallback_pos, quat_topdown.tolist())
        if joints is not None:
            move_to_joints(joints)
            break

# Settle
for _ in range(3):
    obs = get_observation()

# === Step 2: Localize the ketchup bottle ===
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]

# These prompts strongly identify the upright orange/yellow ketchup bottle.
ketchup_prompts = ["ketchup bottle", "orange bottle", "tall bottle", "ketchup", "tomato ketchup"]
obj_center, obj_pts, obj_mask, obj_score = localize_object(rgb, depth, K, E, ketchup_prompts, min_score=0.4)
if obj_center is None:
    raise RuntimeError("Ketchup not found")
print(f"[INFO] ketchup score={obj_score:.3f} center={obj_center.tolist()}")

# Localize tray
tray_prompts = ["tray", "wooden tray", "serving tray"]
tgt_center, tgt_pts, _, tgt_score = localize_object(rgb, depth, K, E, tray_prompts, min_score=0.3)
if tgt_center is None:
    raise RuntimeError("Tray not found")
tray_rim_z = tgt_pts[:, 2].max()
print(f"[INFO] tray score={tgt_score:.3f} center={tgt_center.tolist()} rim_z={tray_rim_z:.3f}")

# === Step 3: Decide grasp ===
# Robust grasp position for a tall narrow bottle:
# 1. The SAM3 mask can include stray pixels (shadows, table, etc.). Filter
#    outliers by distance from the median XY.
# 2. Use the inlier centroid for grasp XY.
# 3. Use plan_grasp for the z (which sees the actual visible surface).

# Filter outliers: iterate to converge on the dominant cluster.
# Start with median XY; keep points within 4cm; recompute median; repeat.
inliers = obj_pts.copy()
for _it in range(3):
    med = np.median(inliers[:, :2], axis=0)
    xy_dist = np.linalg.norm(inliers[:, :2] - med, axis=1)
    new_in = inliers[xy_dist < 0.04]
    if len(new_in) < 10:
        break
    inliers = new_in
print(f"[INFO] inliers={len(inliers)}/{len(obj_pts)} from mask")

inlier_x_med = np.median(inliers[:, 0])
inlier_y_med = np.median(inliers[:, 1])
inlier_z_max = inliers[:, 2].max()
inlier_z_min = inliers[:, 2].min()
print(f"[INFO] inlier XY median=({inlier_x_med:.3f}, {inlier_y_med:.3f})  z_range=[{inlier_z_min:.3f}, {inlier_z_max:.3f}]")

# For a tall narrow bottle, use a fixed top-down quat. Yaw=0 worked best
# across seeds (4/5 on 51-55).
quat = make_topdown_quat(0)

# Grasp z: target 4cm below the bottle top. This is the upper body
# (just below the cap) — empirically the most stable grasp z.
target_grasp_z = inlier_z_max - 0.04
target_grasp_z = max(target_grasp_z, inlier_z_min + 0.04)
print(f"[INFO] target_grasp_z={target_grasp_z:.3f}")
grasp_pos = np.array([inlier_x_med, inlier_y_med, target_grasp_z])
print(f"[INFO] grasp_pos={grasp_pos.tolist()}")

# === Step 4: Execute pick ===
open_gripper()
goto_pose(grasp_pos, quat, z_approach=0.15)
goto_pose(grasp_pos, quat)
close_gripper()

# Lift
lift_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.20])
joints = solve_ik(lift_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 5: Move above tray ===
above_tgt = np.array([tgt_center[0], tgt_center[1], lift_pos[2]])
joints = solve_ik(above_tgt.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 6: Lower to release ===
release_pos = np.array([tgt_center[0], tgt_center[1], tray_rim_z + 0.08])
joints = solve_ik(release_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# Release
open_gripper()

# Settle
for _ in range(5):
    get_observation()
