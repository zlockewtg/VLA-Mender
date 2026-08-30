import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
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


# ---- Perception ----
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
E = cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

# Red mug
obj_center, obj_pts, obj_mask = localize_object(
    rgb, depth, K, E,
    ["red mug", "red coffee mug", "red ceramic mug", "red cup"],
)
if obj_center is None:
    raise RuntimeError("Red mug not found")

# Plate (single plate in SCENE6)
tgt_center, tgt_pts, _ = localize_object(
    rgb, depth, K, E,
    ["dinner plate", "plate", "white plate", "round plate"],
)
if tgt_center is None:
    raise RuntimeError("Plate not found")

# ---- Grasp planning ----
mug_top_z = float(obj_pts[:, 2].max())
top_slice = obj_pts[obj_pts[:, 2] > mug_top_z - 0.02]
if len(top_slice) >= 10:
    rim_x, rim_y = float(top_slice[:, 0].mean()), float(top_slice[:, 1].mean())
else:
    rim_x, rim_y = body_xy_centroid(obj_pts, radius=0.04, iters=3)

grasp_pos = np.array([rim_x, rim_y, mug_top_z - 0.06])

print(f"red mug center={obj_center.round(3)} top_z={mug_top_z:.3f}", flush=True)
print(f"rim_xy=({rim_x:.3f},{rim_y:.3f})", flush=True)
print(f"plate center={tgt_center.round(3)} surface_z={float(tgt_pts[:, 2].max()):.3f}", flush=True)


def attempt_grasp(pos, quat):
    open_gripper()
    goto_pose(pos, quat, z_approach=0.15)
    goto_pose(pos, quat)
    close_gripper()
    obs_check = get_observation()
    return float(obs_check["robot_cartesian_pos"][7])


# Try multiple yaws, pick the one giving the WIDEST grip (= solidly grabbing body
# diameter, not handle). Stop early if we find a body-grip (>0.30) — that's stable.
yaw_candidates = [90, 0, 45, 135, 30, 60]
best_yaw = 90
best_grip = -1.0
for yaw_try in yaw_candidates:
    q_try = make_topdown_quat(yaw_try)
    grip_w = attempt_grasp(grasp_pos, q_try)
    print(f"  yaw={yaw_try}: grip_w={grip_w:.3f}", flush=True)
    if grip_w > best_grip:
        best_grip = grip_w
        best_yaw = yaw_try
    if grip_w > 0.30:
        # Solid body grip: keep this and proceed.
        break
    # Reset: open and lift
    open_gripper()
    lift_check = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.15])
    j = solve_ik(lift_check.tolist(), q_try.tolist())
    if j is not None:
        move_to_joints(j)

# Final grasp with best yaw if not currently held
quat = make_topdown_quat(best_yaw)
print(f"chosen yaw={best_yaw} grip_w={best_grip:.3f}", flush=True)

obs_check = get_observation()
if float(obs_check["robot_cartesian_pos"][7]) < 0.05:
    open_gripper()
    goto_pose(grasp_pos, quat, z_approach=0.15)
    goto_pose(grasp_pos, quat)
    close_gripper()
    for _ in range(2):
        get_observation()

# ---- Lift ----
lift_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.18])
joints = solve_ik(lift_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# ---- Transit above target ----
above_tgt = np.array([tgt_center[0], tgt_center[1], lift_pos[2]])
joints = solve_ik(above_tgt.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# ---- Descend to release (two-stage to stabilize top-heavy flared mug) ----
surface_z = float(tgt_pts[:, 2].max())
mug_bot_z = float(obj_pts[:, 2].min())

# Stage 1: gentle hover above plate
release_z_high = grasp_pos[2] + (surface_z - mug_bot_z) + 0.020
release_pos_high = np.array([tgt_center[0], tgt_center[1], release_z_high])
joints = solve_ik(release_pos_high.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# Stage 2: lower so mug bottom rests on plate
release_z_low = grasp_pos[2] + (surface_z - mug_bot_z) - 0.005
release_pos_low = np.array([tgt_center[0], tgt_center[1], release_z_low])
joints = solve_ik(release_pos_low.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# Settle so mug stabilizes on plate
for _ in range(2):
    get_observation()

open_gripper()
for _ in range(5):
    get_observation()

# ---- Retreat (fully clear of plate) ----
retreat = np.array([release_pos_low[0], release_pos_low[1], release_pos_low[2] + 0.20])
joints = solve_ik(retreat.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)
# Extra settling time after retreat — let physics engine register contact.
for _ in range(8):
    get_observation()
