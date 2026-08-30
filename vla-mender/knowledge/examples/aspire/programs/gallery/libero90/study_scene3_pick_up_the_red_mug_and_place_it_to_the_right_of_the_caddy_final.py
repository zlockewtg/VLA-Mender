import numpy as np
from scipy.spatial.transform import Rotation

def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

# === Settle physics ===
for _ in range(3):
    open_gripper(); close_gripper()
open_gripper()

obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
E = cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

# === Localize Red Mug ===
masks_m = segment_sam3_text_prompt(rgb, "red mug")
m = max(masks_m, key=lambda d: d["score"])
mug_mask = m["mask"].astype(np.uint8)
mug_pts = mask_to_world_points(mug_mask, depth_img, K, E)

mid_pts = mug_pts[(mug_pts[:, 2] > 0.04) & (mug_pts[:, 2] < 0.08)]
if len(mid_pts) < 30:
    mid_pts = mug_pts[(mug_pts[:, 2] > mug_pts[:,2].min()+0.04) & (mug_pts[:, 2] < mug_pts[:,2].min()+0.10)]
body_x = float(np.median(mid_pts[:, 0]))
body_y = float(np.median(mid_pts[:, 1]))
mug_top_z = float(mug_pts[:, 2].max())
print(f"[mug] body=({body_x:.3f},{body_y:.3f}), top_z={mug_top_z:.3f}")

# === Caddy ===
masks_c = segment_sam3_text_prompt(rgb, "desk organizer")
m_c = max(masks_c, key=lambda d: d["score"])
caddy_mask = m_c["mask"].astype(np.uint8)
caddy_pts = mask_to_world_points(caddy_mask, depth_img, K, E)
caddy_obb = get_oriented_bounding_box_from_3d_points(caddy_pts)
caddy_center = caddy_obb["center"]
caddy_y_max = float(np.percentile(caddy_pts[:, 1], 95))

target_x = float(caddy_center[0])
target_y = caddy_y_max + 0.10

# === GRASP STRATEGY: choose ONCE based on body position ===
quat = make_topdown_quat(yaw_deg=90)
grasp_z = mug_top_z - 0.025

if body_x <= 0.745:
    # Easy case: body grip yaw=90 at body center
    tx = body_x
    ty = body_y
    print(f"[grasp body_grip] tgt=({tx:.3f},{ty:.3f}), yaw=90")
else:
    # Hard case: front-rim grip yaw=0
    quat = make_topdown_quat(yaw_deg=0)
    tx = min(body_x - 0.040, 0.730)
    ty = body_y - 0.025
    print(f"[grasp rim_grip] tgt=({tx:.3f},{ty:.3f}), yaw=0")

# Pre-grasp
j = solve_ik([tx, ty, 0.30], quat.tolist())
if j is not None: move_to_joints(j)

# Descend (multi-pass)
for _ in range(6):
    j = solve_ik([tx, ty, grasp_z], quat.tolist())
    if j is not None: move_to_joints(j)

close_gripper()
cp = get_observation()["robot_cartesian_pos"]
gw = float(cp[7])
hand = cp[:3].copy()
print(f"[grasp] hand=({hand[0]:.3f},{hand[1]:.3f},{hand[2]:.3f}) gw={gw:.3f}")

if gw < 0.05:
    print("[FAIL] no grasp")
else:
    # Lift
    j = solve_ik([hand[0], hand[1], 0.45], quat.tolist())
    if j is not None: move_to_joints(j)
    for _ in range(2):
        j = solve_ik([hand[0], hand[1], 0.45], quat.tolist())
        if j is not None: move_to_joints(j)
    cp = get_observation()["robot_cartesian_pos"]
    print(f"[lift] hand=({cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f}) gw={cp[7]:.3f}")

    if cp[7] < 0.03:
        print("[FAIL] dropped")
    else:
        # Transit
        quat_top = make_topdown_quat(0)
        j = solve_ik([cp[0], cp[1], 0.45], quat_top.tolist())
        if j is not None: move_to_joints(j)
        j = solve_ik([0.55, 0.10, 0.45], quat_top.tolist())
        if j is not None: move_to_joints(j)
        j = solve_ik([target_x, target_y, 0.45], quat_top.tolist())
        if j is not None: move_to_joints(j)
        cp = get_observation()["robot_cartesian_pos"]
        print(f"[transit] hand=({cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f}) gw={cp[7]:.3f}")

        for inter_z in [0.30, 0.20, 0.12]:
            j = solve_ik([target_x, target_y, inter_z], quat_top.tolist())
            if j is not None: move_to_joints(j)

        cp = get_observation()["robot_cartesian_pos"]
        print(f"[before release] gw={cp[7]:.3f}")

        open_gripper()
        for _ in range(3): get_observation()

        j = solve_ik([target_x, target_y, 0.40], quat_top.tolist())
        if j is not None: move_to_joints(j)
        for _ in range(5): get_observation()
print("[done]")
