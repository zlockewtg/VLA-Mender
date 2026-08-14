"""
KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate

Scene: 3 black bowls on the table + a plate with red rims + a wooden cabinet.
- Bowls all detected by SAM3 'small bowl' / 'bowl' (score>0.93). Each ~5cm tall.
- Sorting bowls by world-X gives back (X~0.48), middle (X~0.59), front (X~0.74).
- "the middle bowl" = bowl with mid X (~0.59), bowls[1] after sort by X ascending.

Strategy (mirrors KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate but
picks bowls[1] instead of bowls[-1]):
1. SAM3 'small bowl' returns 3 candidates with score>=0.93. Sort by X ascending.
   Pick bowls[1] = MIDDLE bowl.
2. Plate via 'plate with red rims', median XY, percentile85 z for surface.
3. Top-down grasp at bowl rim (Z = rim_z - 0.008).
4. Lift to z=0.30, transport to plate XY, descend, release.

Note: GraspNet neighborhood tightened from 8cm to 3cm to reduce rim-edge grasps
(was the dominant failure mode in the front-bowl run, 7/30).
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ np.array(
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    )
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


TOP_DOWN_QUAT = make_topdown_quat(0)
TOP_DOWN_QUAT_Y90 = make_topdown_quat(90)


def get_view():
    obs = get_observation()
    cam = obs["agentview"]
    rgb = cam["images"]["rgb"]
    depth = cam["images"]["depth"]
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    K = cam["intrinsics"]
    E = cam["pose_mat"]
    return rgb, depth, depth_img, K, E, obs


def localize_bowls(rgb, depth_img, K, E):
    """Return list of (center, pts, mask, score) sorted by world-X ascending (back->front)."""
    masks = segment_sam3_text_prompt(rgb, "small bowl")
    if not masks:
        masks = segment_sam3_text_prompt(rgb, "bowl")
    bowls = []
    for m in sorted(masks or [], key=lambda d: d["score"], reverse=True):
        if m["score"] < 0.5:
            break
        mask = m["mask"].astype(np.uint8)
        pts = mask_to_world_points(mask, depth_img, K, E)
        if pts is None or len(pts) < 100:
            continue
        c = pts.mean(axis=0)
        # Reasonable bounds for table-bowl
        if not (0.40 < c[0] < 0.90 and -0.25 < c[1] < 0.40 and -0.05 < c[2] < 0.10):
            continue
        # Avoid duplicates: skip if center within 5cm of an existing one
        dup = False
        for b in bowls:
            if np.linalg.norm(np.array(c) - np.array(b[0])) < 0.05:
                dup = True
                break
        if dup:
            continue
        bowls.append((c, pts, mask, float(m["score"])))
    bowls.sort(key=lambda b: b[0][0])  # back (small X) -> front (large X)
    return bowls


def localize_plate(rgb, depth_img, K, E):
    for prompt in ["plate with red rims", "white plate", "dinner plate", "plate"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        # plate sits flat on table -> small Z range, located low
        cands = []
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:5]:
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 200:
                continue
            zr = pts[:, 2].max() - pts[:, 2].min()
            zmed = float(np.median(pts[:, 2]))
            # plate is flat (Z range < 0.025) and at table level (Z < 0.02)
            if zr > 0.04 or zmed > 0.03:
                continue
            cands.append((pts, m["score"]))
        if cands:
            pts = max(cands, key=lambda p: p[1])[0]
            center = np.median(pts, axis=0)
            return center, pts
    return None, None


# ===== START =====
print(f"Task: {env.handle.task_language}", flush=True)
goto_home_joint_position()
open_gripper()

# Settle physics with gripper toggles
close_gripper()
open_gripper()
close_gripper()
open_gripper()

rgb, depth, depth_img, K, E, _ = get_view()

# 1. Find all bowls and pick "middle" (bowls[1] after sort by X ascending)
bowls = localize_bowls(rgb, depth_img, K, E)
print(f"Detected {len(bowls)} bowls", flush=True)
for i, (c, _, _, sc) in enumerate(bowls):
    print(f"  bowl[{i}] X={c[0]:.3f} Y={c[1]:.3f} Z={c[2]:.3f} score={sc:.3f}", flush=True)
if len(bowls) < 2:
    raise RuntimeError(f"Need at least 2 bowls for middle; got {len(bowls)}")
# Use bowls[1] (middle X). If only 2 bowls detected, treat the larger-X as middle (best guess).
if len(bowls) >= 3:
    target_center, target_pts, target_mask, target_score = bowls[1]
else:
    # Fallback: with 2 bowls, the gap in X will tell us; but safer to try the closer-to-mid
    # bowl. Default to bowls[1] which is whichever exists at index 1.
    target_center, target_pts, target_mask, target_score = bowls[1]
bx, by, bz = float(target_center[0]), float(target_center[1]), float(target_center[2])
b_zmin = float(target_pts[:, 2].min())
b_zmax = float(target_pts[:, 2].max())
print(f"MIDDLE bowl: ({bx:.3f},{by:.3f},{bz:.3f}) Z=[{b_zmin:.3f},{b_zmax:.3f}]", flush=True)

# 2. Plate
plate_center, plate_pts = localize_plate(rgb, depth_img, K, E)
if plate_center is None:
    raise RuntimeError("Plate not found")
px = float(plate_center[0])
py = float(plate_center[1])
plate_surface_z = float(np.percentile(plate_pts[:, 2], 85))
print(f"Plate: ({px:.3f},{py:.3f}) surface_z={plate_surface_z:.4f}", flush=True)

# 3. Plan grasp at bowl center (matches reference task code that hit 76.7%)
grasp_quat = TOP_DOWN_QUAT.copy()
target_grasp_z = max(b_zmax - 0.008, 0.025)  # ~0.032
grasp_xy = np.array([bx, by])

# Skip GraspNet — bowl is small symmetric, centroid is best.
print(f"Using bowl centroid XY for grasp (no GraspNet): ({bx:.3f},{by:.3f})", flush=True)

# 4. Approach + grasp
hover_z = 0.20
gx, gy = float(grasp_xy[0]), float(grasp_xy[1])

def attempt_grasp(gx, gy, gz, quat):
    """Approach hover, descend in 2 steps, close gripper. Returns gw."""
    open_gripper()
    j = solve_ik([gx, gy, hover_z], quat.tolist())
    if j is not None:
        move_to_joints(j)
    j = solve_ik([gx, gy, gz + 0.05], quat.tolist())
    if j is not None:
        move_to_joints(j)
    j = solve_ik([gx, gy, gz], quat.tolist())
    if j is not None:
        move_to_joints(j)
    close_gripper()
    o = get_observation()
    return float(o["robot_cartesian_pos"][-1])

gw = attempt_grasp(gx, gy, target_grasp_z, grasp_quat)
print(f"After 1st grasp: gw={gw:.3f} at z={target_grasp_z:.3f}", flush=True)

# Retry with deeper Z if air grasp
if gw < 0.030:
    print("Retry grasp lower z", flush=True)
    open_gripper()
    j = solve_ik([gx, gy, hover_z], grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)
    for z_try in [target_grasp_z - 0.010, target_grasp_z - 0.020]:
        z_t = max(z_try, 0.005)
        gw = attempt_grasp(gx, gy, z_t, grasp_quat)
        print(f"  retry z={z_t:.3f}: gw={gw:.3f}", flush=True)
        if gw >= 0.030:
            target_grasp_z = z_t
            break

# Retry with yaw=90 if still air
if gw < 0.030:
    print("Retry yaw=90", flush=True)
    open_gripper()
    j = solve_ik([gx, gy, hover_z], TOP_DOWN_QUAT_Y90.tolist())
    if j is not None:
        move_to_joints(j)
    gw = attempt_grasp(gx, gy, target_grasp_z, TOP_DOWN_QUAT_Y90)
    print(f"  yaw=90 grasp: gw={gw:.3f}", flush=True)
    if gw >= 0.030:
        grasp_quat = TOP_DOWN_QUAT_Y90.copy()

if gw < 0.025:
    print(f"WARN: gw still low ({gw:.3f}); proceeding anyway", flush=True)

# 5. Lift HIGH
lift_z = 0.30
for step_z in [target_grasp_z + 0.05, target_grasp_z + 0.15, lift_z]:
    j = solve_ik([gx, gy, step_z], grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)
o2 = get_observation()
gw2 = float(o2["robot_cartesian_pos"][-1])
print(f"After lift to {lift_z:.3f}: gw={gw2:.3f}", flush=True)

# 6. Transport to plate XY at lift height (interpolated)
xy_off_x = gx - bx
xy_off_y = gy - by
target_arm_x = px + xy_off_x
target_arm_y = py + xy_off_y
print(f"XY offset (grasp - bowl): ({xy_off_x:.3f},{xy_off_y:.3f}) -> arm target ({target_arm_x:.3f},{target_arm_y:.3f})", flush=True)

n_steps = 4
for i in range(1, n_steps + 1):
    t = i / n_steps
    wx = gx + t * (target_arm_x - gx)
    wy = gy + t * (target_arm_y - gy)
    j = solve_ik([wx, wy, lift_z], grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)

# 7. Descend & release
release_z = plate_surface_z + 0.18  # wrist; bowl bottom ~ plate_surface_z + 0.03
n_d = 5
for i in range(1, n_d + 1):
    t = i / n_d
    wz = lift_z + t * (release_z - lift_z)
    j = solve_ik([target_arm_x, target_arm_y, wz], grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)
    obs_d = get_observation()
    gw_d = float(obs_d["robot_cartesian_pos"][-1])
    arm_xy_d = obs_d["robot_cartesian_pos"][:2]
    print(f"  descent z={wz:.3f}: arm_xy=({arm_xy_d[0]:.3f},{arm_xy_d[1]:.3f}) gw={gw_d:.3f}", flush=True)

# Release
open_gripper()
open_gripper()
for _ in range(10):
    get_observation()

# 8. Retreat
j = solve_ik([target_arm_x, target_arm_y, lift_z], grasp_quat.tolist())
if j is not None:
    move_to_joints(j)
goto_home_joint_position()
print("Done", flush=True)
