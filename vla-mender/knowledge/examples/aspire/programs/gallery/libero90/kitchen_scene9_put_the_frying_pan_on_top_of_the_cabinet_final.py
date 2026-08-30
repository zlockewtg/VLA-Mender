"""
KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet — pick-and-place via HANDLE grasp.

Scene:
- Frying pan body at world XY ≈ (0.70, 0.01), 20cm diameter, ~4cm thick, body Z: -0.011 to 0.028.
- Pan handle extends in +Y, ~14.5cm long, ~4.5cm wide, ~3.7cm thick. Handle center ≈ (0.692, 0.186, 0.024). Handle Z top ~0.043.
- Dark wooden cabinet on table. SAM3 prompt: "tabletop dark cabinet" (foreground filter X∈[0.4,1.0], Y<0, Z>0.05).
  Cabinet top ≈ z=0.218, X∈[0.49, 0.84], Y∈[-0.37, -0.18]. Center XY ≈ (0.66, -0.27).

Strategy:
1. Localize cabinet (foreground), compute target placement at top center.
2. Localize pan HANDLE via SAM3 ("pan handle"/"frying pan handle"): score ~0.94, very reliable.
3. Grasp handle top-down with yaw=90° so gripper fingers close along world X (perpendicular to handle Y axis).
   Handle is 4.5cm wide (X extent) → fits within 8cm gripper.
   Grip near body (Y ≈ handle_center_y - 0.02, e.g. Y=0.16) for better stability/balance.
4. Lift HIGH (z=0.50+), interpolate transit to cabinet center, descend, release.

CRITICAL filters:
- Cabinet candidates: X > 0.4, Y < 0, z_center > 0.05 → reject background (X≈-0.75) and table junk.
- Handle SAM3 returns one mask in foreground reliably; reject any score<0.5 candidate or wrong-place candidate.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ np.array(
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    )
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz for API


# yaw=90° so gripper closes along world X (handle is oriented along world Y, so X is the cross-axis).
HANDLE_QUAT = make_topdown_quat(yaw_deg=90)
TOP_DOWN_QUAT = make_topdown_quat(yaw_deg=0)


def localize_cabinet(rgb, depth_img, K, E):
    """Find foreground tabletop cabinet by filtering candidates by world coords."""
    for prompt in ["tabletop dark cabinet", "dark drawer unit", "dark wooden block", "dark block", "dark wooden cabinet"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:3]:
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 200:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c = np.array(obb["center"])
            if 0.40 < c[0] < 1.0 and -0.50 < c[1] < 0.0 and c[2] > 0.05:
                return pts, c, prompt
    return None, None, None


def localize_handle(rgb, depth_img, K, E):
    """Find pan handle. Returns (pts, center) or (None, None)."""
    for prompt in ["frying pan handle", "pan handle", "black handle", "handle"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:5]:
            if m["score"] < 0.5:
                continue
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 50:
                continue
            # Foreground filter: handle is in front of robot, near pan body
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])
            cz = np.mean(pts[:, 2])
            if not (0.5 < cx < 0.9 and -0.1 < cy < 0.4 and -0.05 < cz < 0.1):
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            return pts, np.array(obb["center"]), prompt
    return None, None, None


def localize_pan(rgb, depth_img, K, E):
    """Localize pan body (returns pts, body_center_xy)."""
    for prompt in ["frying pan", "pan", "skillet"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        m = max(masks, key=lambda d: d["score"])
        pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
        if pts is None or len(pts) < 200:
            continue
        # Body has the lower median Y (handle in +Y direction)
        med_y = np.median(pts[:, 1])
        body = pts[pts[:, 1] < med_y + 0.05]
        if len(body) < 50:
            body = pts
        obb_b = get_oriented_bounding_box_from_3d_points(body)
        return pts, body, np.array(obb_b["center"]), m["mask"]
    return None, None, None, None


# ===== START =====
print(f"Task: {env.handle.task_language}", flush=True)
goto_home_joint_position()
open_gripper()

obs0 = get_observation()
cam0 = obs0["agentview"]
rgb0 = cam0["images"]["rgb"]
depth0 = cam0["images"]["depth"]
depth_img0 = depth0[:, :, 0] if len(depth0.shape) == 3 else depth0
K0 = cam0["intrinsics"]
E0 = cam0["pose_mat"]

# 1. Cabinet (BEFORE arm motion)
cab_pts, cab_center, cab_prompt = localize_cabinet(rgb0, depth_img0, K0, E0)
if cab_pts is None:
    raise RuntimeError("Cabinet not found")
cab_top_z = float(cab_pts[:, 2].max())
top_pts = cab_pts[cab_pts[:, 2] >= cab_top_z - 0.005]
target_x = float(np.median(top_pts[:, 0]))
target_y = float(np.median(top_pts[:, 1]))
y_min = float(top_pts[:, 1].min())
y_max = float(top_pts[:, 1].max())
target_y = float(np.clip(target_y, y_min + 0.08, y_max - 0.08))
print(f"Cabinet ({cab_prompt}): top_z={cab_top_z:.4f}, target=({target_x:.3f},{target_y:.3f}), Y∈({y_min:.3f},{y_max:.3f})", flush=True)

# 2. Pan handle
handle_pts, handle_center, handle_prompt = localize_handle(rgb0, depth_img0, K0, E0)
if handle_center is None:
    raise RuntimeError("Pan handle not found")
hx, hy, hz = float(handle_center[0]), float(handle_center[1]), float(handle_center[2])
handle_top_z = float(handle_pts[:, 2].max())
handle_y_min = float(handle_pts[:, 1].min())
handle_y_max = float(handle_pts[:, 1].max())
print(f"Handle ({handle_prompt}): center=({hx:.3f},{hy:.3f},{hz:.3f}), top_z={handle_top_z:.4f}, Y∈({handle_y_min:.3f},{handle_y_max:.3f})", flush=True)

# Pan body for placement-XY computation
pan_full, pan_body, pan_body_center, pan_mask = localize_pan(rgb0, depth_img0, K0, E0)
pbx = float(pan_body_center[0]) if pan_body_center is not None else hx
pby = float(pan_body_center[1]) if pan_body_center is not None else hy - 0.18
print(f"Pan body: center=({pbx:.3f},{pby:.3f})", flush=True)

# 3. Plan handle grasp:
# - Grasp center: handle_center, but bias toward body side for stability
#   Handle Y range ~[0.10, 0.26]; grasp at Y = handle_y_min + 0.04 (closer to body) for less leverage on pan.
# - Grip Z: just below handle top to ensure fingers wrap around handle wall.
grasp_x = hx
grasp_y = float(np.clip(handle_y_min + 0.04, hy - 0.04, hy + 0.02))  # near body end of handle
grasp_z = max(handle_top_z - 0.020, 0.005)  # 2cm below top — within handle wall
grasp_quat = HANDLE_QUAT.copy()
print(f"Grasp plan: pos=({grasp_x:.3f},{grasp_y:.3f},{grasp_z:.3f}), yaw=90°", flush=True)

# 4. Approach + grasp
hover_z = 0.20
j = solve_ik([grasp_x, grasp_y, hover_z], grasp_quat.tolist())
if j is not None:
    move_to_joints(j)
# Descend in 2 steps
for z_step in [0.10, grasp_z]:
    j = solve_ik([grasp_x, grasp_y, z_step], grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)
close_gripper()

obs_g = get_observation()
gw_after = float(obs_g["robot_cartesian_pos"][-1])
print(f"After close: gw={gw_after:.3f}", flush=True)

# Retry with adjusted z if grasp failed
if gw_after < 0.020:
    print("Air grasp — retrying", flush=True)
    open_gripper()
    for z_try in [grasp_z + 0.005, grasp_z - 0.005, grasp_z + 0.010]:
        z_t = max(z_try, 0.003)
        j = solve_ik([grasp_x, grasp_y, hover_z], grasp_quat.tolist())
        if j is not None:
            move_to_joints(j)
        j = solve_ik([grasp_x, grasp_y, z_t], grasp_quat.tolist())
        if j is not None:
            move_to_joints(j)
        close_gripper()
        obs_r = get_observation()
        gw_after = float(obs_r["robot_cartesian_pos"][-1])
        print(f"  retry z={z_t:.3f}: gw={gw_after:.3f}", flush=True)
        if gw_after >= 0.020:
            grasp_z = z_t
            break
        open_gripper()

# 5. Lift HIGH for transit
# Pan handle is fairly long; the pan body hangs from handle ~10cm in -Y direction below handle grasp axis.
# Lifting transports the pan body too. Need clearance over cabinet.
lift_z = max(grasp_z + 0.30, cab_top_z + 0.30, 0.50)
for step_z in [grasp_z + 0.05, grasp_z + 0.15, lift_z]:
    j = solve_ik([grasp_x, grasp_y, step_z], grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)
obs_l = get_observation()
gw_lift = float(obs_l["robot_cartesian_pos"][-1])
print(f"After lift to z={lift_z:.3f}: gw={gw_lift:.3f}", flush=True)

# 6. Lateral transit to above cabinet center.
# When holding handle, the pan body hangs ~ -Y direction relative to grasp_y by ~(grasp_y - pan_body_center_y) offset.
# So if we want pan BODY centered over cabinet target, arm should go to (target_x, target_y + body_offset_y).
body_offset_y = grasp_y - pby   # how much arm-Y is in front of body-Y (e.g. 0.16 - (-0.001) = 0.16)
arm_target_y = target_y + body_offset_y
# But arm_target_y must be feasible. Also must keep arm_x near target_x; pan body is roughly at the same X
# as arm (the handle is roughly straight-out-Y from body, so X same).
arm_target_x = target_x
print(f"Body offset_y={body_offset_y:.3f}, arm_target=({arm_target_x:.3f},{arm_target_y:.3f})", flush=True)

# Interpolate transit
n_steps = 4
for i in range(1, n_steps + 1):
    t = i / n_steps
    wx = grasp_x + t * (arm_target_x - grasp_x)
    wy = grasp_y + t * (arm_target_y - grasp_y)
    j = solve_ik([wx, wy, lift_z], grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)
obs_t = get_observation()
gw_t = float(obs_t["robot_cartesian_pos"][-1])
print(f"At above-target ({arm_target_x:.3f},{arm_target_y:.3f},{lift_z:.3f}): gw={gw_t:.3f}", flush=True)

# 7. Descent — release pan body just above cabinet top.
# Arm must be at z = cab_top_z + (handle_z_above_table_when_held). When pan body is on cabinet top,
# the handle is at cab_top_z + handle_top_z = 0.218 + 0.04 = 0.258. So arm wrist (TCP+handle thickness)
# must be at ~0.258 + 0.05 (gripper depth above handle top) = ~0.30. Use a few cm extra clearance.
release_z = cab_top_z + 0.10  # arm Z; with TCP offset ~0.10, fingertips at cab_top_z + 0.00 = exactly at top
# Use z = cab_top_z + 0.10 → fingers at cab_top_z (touching), pan body sits firmly on top.
# Actually, since gripper grips handle (which is at handle_top_z above table = 0.04), arm needs to be:
#   cab_top_z + handle_thickness_held (~0.04) + small margin = 0.218 + 0.04 + 0.04 ≈ 0.30
release_z = cab_top_z + 0.09  # ≈ 0.31 — pan body sits on cabinet top, handle still gripped
n_d = 3
for i in range(1, n_d + 1):
    t = i / n_d
    wz = lift_z + t * (release_z - lift_z)
    j = solve_ik([arm_target_x, arm_target_y, wz], grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)
obs_p = get_observation()
gw_p = float(obs_p["robot_cartesian_pos"][-1])
print(f"At release ({arm_target_x:.3f},{arm_target_y:.3f},{release_z:.3f}): gw={gw_p:.3f}", flush=True)

# 8. Release & settle
open_gripper()
for _ in range(8):
    get_observation()

# 9. Retreat up
j = solve_ik([arm_target_x, arm_target_y, lift_z], grasp_quat.tolist())
if j is not None:
    move_to_joints(j)
goto_home_joint_position()
print("Done", flush=True)
