"""
LIVING_ROOM_SCENE5: put the red mug on the left plate.

Approach (combining best findings from v10 + baseline pattern):
- Localize red mug (SAM3 'red mug' score ~0.95).
- Identify TWO plates; pick LEFT (smallest world-Y) — confirmed: y=-0.295 in seed 51.
- Grasp red mug at OBB centroid with TOP_DOWN_QUAT (works: gw≈0.10 stable).
- Lift to safe height.
- Transit to plate center (no body-offset compensation — drop is forgiving).
- Descend to fingertip ~0.10 above plate, open gripper, let mug fall.
- Allow mug to settle.

Geometry:
- Red mug: tall ~15cm, body ~5cm dia, flared rim ~9cm.
- Plate: thin ceramic ~13.5cm dia, top z ≈ 0.033.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


TOP_DOWN_QUAT = make_topdown_quat(0)


def localize_filtered(rgb, depth_img, K, E, prompts, *,
                      z_min=0.005, z_max=0.30, min_pts=20,
                      extent_max=None, min_score=0.30):
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:5]:
            if m["score"] < min_score:
                continue
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < min_pts:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c = obb["center"]
            if c[2] < z_min or c[2] > z_max:
                continue
            if extent_max is not None and max(obb["extent"]) > extent_max:
                continue
            return c, pts, mask, obb
    return None, None, None, None


def find_plates(rgb, depth_img, K, E, *, min_score=0.40):
    detections = []
    for prompt in ["plate", "white plate", "ceramic plate", "round plate"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:5]:
            if m["score"] < min_score:
                continue
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 50:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c = obb["center"]
            ext = obb["extent"]
            if ext[0] > 0.30 or ext[1] > 0.30:
                continue
            if c[2] < 0.005 or c[2] > 0.10:
                continue
            duplicate = False
            for prev in detections:
                if np.linalg.norm(prev["center"][:2] - c[:2]) < 0.10:
                    duplicate = True
                    break
            if duplicate:
                continue
            detections.append({
                "center": c, "pts": pts, "mask": mask, "obb": obb,
                "score": m["score"], "prompt": prompt,
            })
        if len(detections) >= 2:
            break
    return detections


# ── Observe ──────────────────────────────────────────────────────────────────
goto_home_joint_position()
open_gripper()

obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]
print(f"Task: {env.handle.task_language}", flush=True)


# ── Localize red mug ─────────────────────────────────────────────────────────
mug_center, mug_pts, mug_mask, mug_obb = localize_filtered(
    rgb, depth_img, K, E,
    ["red mug", "red ceramic mug", "red coffee mug", "red cup"],
    z_min=0.020, z_max=0.20, min_pts=200, min_score=0.40, extent_max=0.20,
)
if mug_center is None:
    raise RuntimeError("Red mug not found")
mug_z_min = float(mug_pts[:, 2].min())
mug_z_max = float(mug_pts[:, 2].max())
mug_height = mug_z_max - mug_z_min
print(f"Mug center={mug_center.round(3)} z=[{mug_z_min:.3f},{mug_z_max:.3f}] h={mug_height:.3f}",
      flush=True)


# ── Localize plates, pick LEFT (smallest world-Y) ────────────────────────────
plates = find_plates(rgb, depth_img, K, E)
if len(plates) < 1:
    raise RuntimeError("No plates found")
print(f"Found {len(plates)} plates:", flush=True)
for p in plates:
    print(f"  prompt='{p['prompt']}' score={p['score']:.3f} center={p['center'].round(3)}",
          flush=True)
plates_sorted = sorted(plates, key=lambda p: p["center"][1])
left_plate = plates_sorted[0]
plate_pts = left_plate["pts"]
plate_x_mid = (np.percentile(plate_pts[:, 0], 10) + np.percentile(plate_pts[:, 0], 90)) / 2
plate_y_mid = (np.percentile(plate_pts[:, 1], 10) + np.percentile(plate_pts[:, 1], 90)) / 2
plate_top_z = float(np.percentile(plate_pts[:, 2], 90))
plate_center = np.array([plate_x_mid, plate_y_mid, plate_top_z])
print(f"LEFT plate center={plate_center.round(3)} top_z={plate_top_z:.3f}", flush=True)


# ── Grasp red mug at OBB centroid with TOP_DOWN_QUAT ────────────────────────
grasp_x, grasp_y = float(mug_center[0]), float(mug_center[1])
grasp_quat = TOP_DOWN_QUAT
print(f"Grasp XY=({grasp_x:.3f},{grasp_y:.3f}) TOP_DOWN", flush=True)

# Approach high
goto_pose([grasp_x, grasp_y, 0.30], grasp_quat.tolist())

# Descend in steps to fingertip ~0.075 (mid-body of mug)
fingertip_grasp_z = 0.075
for step_finger_z in [0.20, 0.15, 0.12, 0.09, fingertip_grasp_z]:
    goto_pose([grasp_x, grasp_y, step_finger_z], grasp_quat.tolist())

obs_pre = get_observation()
rcp_pre = obs_pre.get("robot_cartesian_pos", [])
ee_z_pre = float(rcp_pre[2]) if len(rcp_pre) >= 3 else None
print(f"Wrist Z before close: {ee_z_pre}", flush=True)

close_gripper()
for _ in range(5):
    get_observation()
close_gripper()

obs_g = get_observation()
rcp_g = obs_g.get("robot_cartesian_pos", [])
gw = float(rcp_g[-1]) if len(rcp_g) >= 4 else 0.0
print(f"After grasp: gw={gw:.4f}", flush=True)


# Air-grasp retry: shift Y by small amounts
if gw < 0.04:
    for shift_y in [-0.020, +0.020, -0.030, +0.030]:
        open_gripper()
        rx = grasp_x
        ry = grasp_y + shift_y
        goto_pose([rx, ry, 0.30], grasp_quat.tolist())
        for sz in [0.20, 0.15, 0.12, 0.09, fingertip_grasp_z]:
            goto_pose([rx, ry, sz], grasp_quat.tolist())
        close_gripper()
        for _ in range(5):
            get_observation()
        close_gripper()
        obs_r = get_observation()
        rcp_r = obs_r.get("robot_cartesian_pos", [])
        gw_r = float(rcp_r[-1]) if len(rcp_r) >= 4 else 0.0
        print(f"  retry shift_y={shift_y:+.3f}: gw={gw_r:.4f}", flush=True)
        if gw_r >= 0.04:
            grasp_x, grasp_y = rx, ry
            gw = gw_r
            break


# ── Lift ────────────────────────────────────────────────────────────────────
# Incremental lift to keep mug stable
for step_finger_z in [fingertip_grasp_z + 0.05, fingertip_grasp_z + 0.12,
                       fingertip_grasp_z + 0.20, 0.30]:
    goto_pose([grasp_x, grasp_y, step_finger_z], grasp_quat.tolist())
    close_gripper()

obs_l = get_observation()
rcp_l = obs_l.get("robot_cartesian_pos", [])
gw_l = float(rcp_l[-1]) if len(rcp_l) >= 4 else 0.0
print(f"After lift: gw={gw_l:.4f}", flush=True)


# ── Transit to plate + descend + drop ───────────────────────────────────────
# Compensate for body-vs-grip offset.
# Use the OBSERVED grasp position (after grasp) and mug centroid difference at grasp time.
# Mug body axis was at mug_center initially. Gripper grasped at (grasp_x, grasp_y).
# Body offset from gripper TCP: assume body hung at original position relative to mug centroid.
# But mug_center might bias toward handle. Instead, just place fingertip at plate center
# but with a heuristic offset to put body INTO plate (toward plate's interior from the handle side).
# Heuristic: shift release TOWARD plate center (which is where the body should land).
# Empirically v14 mug landed on camera-LEFT of plate (lower world-Y or higher world-Y?).
# Plate y=-0.298. Mug at y=-0.298+offset where offset≈+0.01. So mug is at y≈-0.288.
# That's CLOSER to camera (= +Y direction visible) — so we need to push release FURTHER from camera.
# Try shifting release by -0.020 in Y (toward plate's far side).

release_x = float(plate_center[0]) + 0.005   # tiny shift toward plate's far side in X
release_y = float(plate_center[1]) - 0.010   # tiny shift toward plate's far side in Y
print(f"Release XY=({release_x:.3f},{release_y:.3f}) plate_top={plate_top_z:.3f}", flush=True)

# Approach above plate in 3 lateral steps at finger_z=0.30
for frac in [0.33, 0.67, 1.0]:
    wpt_x = grasp_x + frac * (release_x - grasp_x)
    wpt_y = grasp_y + frac * (release_y - grasp_y)
    goto_pose([wpt_x, wpt_y, 0.30], grasp_quat.tolist())
    close_gripper()

# Descend to plate + 0.10 (fingertip 10cm above plate top → mug bottom 3-4cm above plate)
place_finger_z = plate_top_z + 0.10
for sz in [0.25, 0.20, 0.15, place_finger_z]:
    goto_pose([release_x, release_y, sz], grasp_quat.tolist())

# Open and let mug fall
open_gripper()
for _ in range(20):  # let mug settle
    get_observation()

# Retract straight up
for sz in [place_finger_z + 0.10, 0.30, 0.40]:
    goto_pose([release_x, release_y, sz], grasp_quat.tolist())

goto_home_joint_position()
for _ in range(15):  # final settle
    get_observation()
print("Done", flush=True)
