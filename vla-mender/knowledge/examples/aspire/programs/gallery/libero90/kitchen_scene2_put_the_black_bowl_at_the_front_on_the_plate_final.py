"""
KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate

Scene: 3 black bowls on the table + a plate with red rims + a wooden cabinet.
- Bowls all detected by SAM3 'small bowl' / 'bowl' (score>0.93). Each ~5cm tall,
  rim at z~0.040, bottom at z~-0.011, centroid at z~0.017.
- Sorting bowls by world-X gives back (X~0.48), middle (X~0.59), front (X~0.74).
- Plate: 'plate with red rims' (score>0.97). Median XY ≈ (0.64–0.68, 0.0±0.02).
  Surface z (85th-pctile) ≈ 0.004.
- "the bowl at the front" = bowl with largest X (~0.74, closest to camera).

Strategy:
1. SAM3 'small bowl' returns 3 candidates with score≥0.93. Filter by reasonable
   X/Y/Z bounds, sort by X ascending. Pick LAST (bowls[-1]) = FRONT bowl.
2. Plate via 'plate with red rims', median XY, percentile85 z for surface.
3. Top-down grasp at bowl rim (Z = rim_z - 0.005 ≈ 0.030).
   Bowl rim is ~10cm diameter, gripper fits comfortably. Yaw=0 standard.
4. Lift to z=0.30, transport to plate XY, descend, release.
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

# 1. Find all bowls and pick "front" (largest X)
bowls = localize_bowls(rgb, depth_img, K, E)
print(f"Detected {len(bowls)} bowls", flush=True)
for i, (c, _, _, sc) in enumerate(bowls):
    print(f"  bowl[{i}] X={c[0]:.3f} Y={c[1]:.3f} Z={c[2]:.3f} score={sc:.3f}", flush=True)
if len(bowls) < 1:
    raise RuntimeError("No bowls detected")
# Use the LAST (largest X = visual bottom = front) bowl.
# In LIBERO KITCHEN_SCENE2, "front" means large X (closest to camera = visual bottom of image).
target_center, target_pts, target_mask, target_score = bowls[-1]
bx, by, bz = float(target_center[0]), float(target_center[1]), float(target_center[2])
b_zmin = float(target_pts[:, 2].min())
b_zmax = float(target_pts[:, 2].max())
print(f"FRONT bowl: ({bx:.3f},{by:.3f},{bz:.3f}) Z=[{b_zmin:.3f},{b_zmax:.3f}]", flush=True)

# 2. Plate
plate_center, plate_pts = localize_plate(rgb, depth_img, K, E)
if plate_center is None:
    raise RuntimeError("Plate not found")
px = float(plate_center[0])
py = float(plate_center[1])
plate_surface_z = float(np.percentile(plate_pts[:, 2], 85))
print(f"Plate: ({px:.3f},{py:.3f}) surface_z={plate_surface_z:.4f}", flush=True)

# 3. Plan grasp — try GraspNet first; fallback to top-down centroid
grasp_quat = TOP_DOWN_QUAT.copy()
# Bowl Z=0.04 rim. Aim for grasp at rim (Z just under rim top).
target_grasp_z = max(b_zmax - 0.008, 0.025)  # ~0.032
grasp_xy = np.array([bx, by])

# Try GraspNet for better XY
try:
    grasps, scores = plan_grasp(depth, K, target_mask)
    if grasps is not None and len(grasps) > 0:
        # find highest-verticality grasp near the bowl XY
        best_idx = -1
        best_score = -np.inf
        for i in range(len(grasps)):
            gw = E @ grasps[i]
            gxy = gw[:2, 3]
            if np.linalg.norm(gxy - np.array([bx, by])) > 0.08:
                continue
            vert = abs(gw[2, 2])
            sc = float(scores[i]) * (vert ** 2)
            if sc > best_score:
                best_score = sc
                best_idx = i
        if best_idx >= 0:
            gw = E @ grasps[best_idx]
            grasp_xy = gw[:2, 3]
            print(f"GraspNet XY: {grasp_xy}", flush=True)
except Exception as ex:
    print(f"GraspNet skipped: {ex}", flush=True)

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
# XY offset correction: bowl hangs offset from TCP by (grasp_xy - bowl_centroid_xy)
xy_off_x = gx - bx  # how much grasp X is offset from bowl center
xy_off_y = gy - by
# Place bowl center at plate center -> arm goes to plate_center + offset
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
# TCP offset: wrist Z = fingertip Z + 0.10. Bowl is ~5cm tall, gripped at rim.
# So bowl bottom = fingertip_Z - 0.05.
# To gently set bowl bottom on plate (plate_surface_z ≈ 0.004), aim fingertip Z = plate_surface_z + 0.06.
# That's wrist Z = plate_surface_z + 0.16 ≈ 0.164.
# But bowl is only loosely held; release needs ~2cm clearance for bowl-bottom-just-above-plate.
release_z = plate_surface_z + 0.18  # wrist; bowl bottom ≈ plate_surface_z + 0.03
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
