"""
KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it

Combined task:
1. PHASE 1: Turn on stove knob via top-down grip + j6 wrist rotation (vertical-axis knob).
2. PHASE 2: Pick frying pan by handle + place on stove burner (KS3_put_pan approach).

Scene (seed 51):
- Stove knob: c=(0.471, 0.200, 0.021), z_max=0.048 — vertical-axis knob/dial.
- Stove burner: c=(0.618, 0.204, 0.018), z_max=0.020 (flush with table)
- Frying pan: c=(0.589, -0.241, 0.007), z_max=0.122 (rim height)
- Pan handle: c=(0.607, -0.091, 0.034), z_max=0.043 (low).
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0.0):
    R = (Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix()
         @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def safe_solve_ik(pos, quat):
    try:
        return solve_ik(np.array(pos), quat)
    except Exception:
        return None


def safe_move(j):
    if j is None:
        return False
    try:
        move_to_joints(j)
        return True
    except Exception:
        return False


def localize_knob(rgb, depth_img, K, E):
    """Find stove knob.
    KS3 knob is at (~0.47, +0.20, ~0.02) — front of stove, near burner.
    Filter: 0.40<x<0.55, 0.10<y<0.30, z_max in [0.015, 0.10].
    """
    candidates = []
    for prompt in ["black stove knob", "stove knob", "knob", "stove switch"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in masks[:8]:
            if m["score"] < 0.10:
                continue
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 50:
                continue
            c = pts.mean(axis=0)
            if not (0.40 < c[0] < 0.55):
                continue
            if not (0.10 < c[1] < 0.30):
                continue
            z_top = pts[:, 2].max()
            if z_top < 0.015 or z_top > 0.10:
                continue
            candidates.append((m["score"], m["mask"], pts, c, float(z_top)))
        if candidates:
            break
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])


def try_grasp_and_j6_rotate(cx, cy, cz_top, yaw_deg=0.0, descent_offset=-0.005):
    """Top-down grasp at knob, then j6 wrist rotation to turn knob.
    cz_top: knob z_max. Descend to cz_top + descent_offset.
    Returns True if grip width > 0.05 (knob actually grabbed).

    Strategy: After gripping knob, do net +90° CW (3 CW + 2 CCW = +1.57 rad).
    Validated on seeds 51, 52, 54.
    """
    try:
        quat = make_topdown_quat(yaw_deg)
        target_z = cz_top + descent_offset
        j_app = safe_solve_ik([cx, cy, cz_top + 0.10], quat)
        j_tgt = safe_solve_ik([cx, cy, target_z], quat)
        if j_tgt is None or j_app is None:
            print(f"  IK failed at yaw={yaw_deg:.0f}", flush=True)
            return False

        open_gripper()
        safe_move(j_app)
        safe_move(j_tgt)
        close_gripper()

        obs_g = get_observation()
        grip = float(obs_g['robot_cartesian_pos'][-1])
        print(f"  yaw={yaw_deg:.0f} z_off={descent_offset:+.3f}: grip={grip:.3f}", flush=True)

        if grip <= 0.05:
            open_gripper()
            return False

        # CW rotations: j6 += 1.57. Net rotation = +90° CW (3 CW + 2 CCW).
        # Validated on seeds 51, 52, 54.
        for _ in range(3):
            j_cur = list(get_observation()['robot_joint_pos'][:7])
            j_cur[6] += 1.57
            safe_move(j_cur)
        for _ in range(2):
            j_cur = list(get_observation()['robot_joint_pos'][:7])
            j_cur[6] -= 1.57
            safe_move(j_cur)
        # Try a second net +90° CW for seeds where 90° wasn't enough.
        for _ in range(2):
            j_cur = list(get_observation()['robot_joint_pos'][:7])
            j_cur[6] += 1.57
            safe_move(j_cur)
        for _ in range(1):
            j_cur = list(get_observation()['robot_joint_pos'][:7])
            j_cur[6] -= 1.57
            safe_move(j_cur)

        open_gripper()
        return True
    except Exception as e:
        print(f"  try_grasp error: {e}", flush=True)
        try:
            open_gripper()
        except Exception:
            pass
        return False


def localize_pan_and_handle(rgb, depth_img, K, E):
    pan_pts = None
    for prompt in ["frying pan", "pan", "skillet"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:3]:
            if m["score"] < 0.3:
                continue
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 500:
                continue
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            if cx < 0.4 or cx > 0.85:
                continue
            pan_pts = pts
            print(f"  Pan via '{prompt}' s3={m['score']:.3f}: pts={len(pts)} c=({cx:.3f},{cy:.3f})", flush=True)
            break
        if pan_pts is not None:
            break

    if pan_pts is None:
        return None, None, None, None, None, None

    med_y = float(np.median(pan_pts[:, 1]))
    pan_y_max = float(pan_pts[:, 1].max())
    pan_y_min = float(pan_pts[:, 1].min())
    pan_y_span = pan_y_max - pan_y_min
    handle_mask_g = (pan_pts[:, 1] > pan_y_max - 0.30 * pan_y_span) & (pan_pts[:, 2] < 0.05)
    body_mask = pan_pts[:, 1] < med_y
    body_pts = pan_pts[body_mask]
    handle_pts_low = pan_pts[handle_mask_g] if handle_mask_g.sum() > 50 else None

    pan_y_max = float(pan_pts[:, 1].max())
    pan_y_min = float(pan_pts[:, 1].min())
    best_handle = None
    best_score = -1
    best_meta = None
    for hprompt in ["frying pan handle", "pan handle"]:
        masks = segment_sam3_text_prompt(rgb, hprompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:6]:
            if m["score"] < 0.20:
                continue
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 30:
                continue
            pts_low = pts[pts[:, 2] < 0.05]
            if len(pts_low) < 100:
                continue
            cx = float(np.mean(pts_low[:, 0]))
            cy = float(np.mean(pts_low[:, 1]))
            if not (pan_pts[:, 0].min() - 0.02 < cx < pan_pts[:, 0].max() + 0.02):
                continue
            if not (pan_y_min - 0.02 < cy < pan_y_max + 0.05):
                continue
            x_extent = float(pts_low[:, 0].max() - pts_low[:, 0].min())
            y_extent = float(pts_low[:, 1].max() - pts_low[:, 1].min())
            top_z_cand = float(pts_low[:, 2].max())
            if x_extent > 0.10:
                continue
            if top_z_cand > 0.052:
                continue
            quality = len(pts_low) - 1000 * x_extent + 100 * y_extent - 500 * max(0, top_z_cand - 0.045)
            if quality > best_score:
                best_score = quality
                best_handle = pts_low
                best_meta = (hprompt, m["score"])
    if best_handle is not None:
        handle_pts_low = best_handle
        print(f"  Handle picked via '{best_meta[0]}' s3={best_meta[1]:.3f}: pts={len(handle_pts_low)}", flush=True)

    if handle_pts_low is None or len(handle_pts_low) < 30:
        return pan_pts, body_pts, None, None, None, None

    handle_center = handle_pts_low.mean(axis=0)
    handle_top_z = float(handle_pts_low[:, 2].max())
    body_center = body_pts.mean(axis=0) if len(body_pts) > 0 else None
    return pan_pts, body_pts, body_center, handle_pts_low, handle_center, handle_top_z


def localize_stove(rgb, depth_img, K, E):
    for prompt in ["stove top", "stove", "stovetop", "burner", "stove burner"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:3]:
            if m["score"] < 0.4:
                continue
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 200:
                continue
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            cz = float(np.mean(pts[:, 2]))
            if not (0.4 < cx < 0.85 and 0.05 < cy < 0.40 and -0.05 < cz < 0.10):
                continue
            return pts, np.array([cx, cy, cz]), float(pts[:, 2].max()), prompt
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

# ============ PHASE 1: Turn on stove knob ============
print("=" * 50, flush=True)
print("PHASE 1: TURN ON STOVE KNOB (j6 wrist rotation)", flush=True)
print("=" * 50, flush=True)

knob_res = localize_knob(rgb0, depth_img0, K0, E0)
if knob_res is None:
    print("WARN: knob not found, fallback to (0.471, 0.200, 0.048)", flush=True)
    kx, ky, kz_top = 0.471, 0.200, 0.048
else:
    score, kmask, kpts, kcentroid, kz_top = knob_res
    kx, ky = float(kcentroid[0]), float(kcentroid[1])
    print(f"Knob: c=({kx:.3f},{ky:.3f}) z_top={kz_top:.3f} score={score:.3f}", flush=True)

# Single attempt at yaw=0 with default descent (validated on 3/5 in v2; trying with
# extended +180° net rotation to see if knob requires more turn).
print(f"--- Attempt yaw=0° z_off=-0.005 ---", flush=True)
success = try_grasp_and_j6_rotate(kx, ky, kz_top, yaw_deg=0.0, descent_offset=-0.005)
if success:
    print(f"  Knob rotation cycle done", flush=True)

# Reset to home before phase 2
try:
    open_gripper()
    goto_home_joint_position()
except Exception:
    pass

# ============ PHASE 2: Pick pan + place on stove ============
print("=" * 50, flush=True)
print("PHASE 2: PICK PAN + PLACE ON STOVE", flush=True)
print("=" * 50, flush=True)

obs1 = get_observation()
cam1 = obs1["agentview"]
rgb1 = cam1["images"]["rgb"]
depth1 = cam1["images"]["depth"]
depth_img1 = depth1[:, :, 0] if len(depth1.shape) == 3 else depth1
K1 = cam1["intrinsics"]
E1 = cam1["pose_mat"]

pan_pts, body_pts, body_center, handle_pts, handle_center, handle_top_z = localize_pan_and_handle(rgb1, depth_img1, K1, E1)
if handle_center is None:
    print("ERROR: pan handle not found, cannot place pan", flush=True)
    print("Done", flush=True)
else:
    hx, hy, hz = float(handle_center[0]), float(handle_center[1]), float(handle_center[2])
    handle_y_min = float(handle_pts[:, 1].min())
    handle_y_max = float(handle_pts[:, 1].max())
    pbx = float(body_center[0]) if body_center is not None else hx
    pby = float(body_center[1]) if body_center is not None else hy - 0.18
    pbz = float(body_center[2]) if body_center is not None else 0.0
    print(f"Handle: c=({hx:.3f},{hy:.3f},{hz:.3f}) top_z={handle_top_z:.3f}", flush=True)
    print(f"Pan body: c=({pbx:.3f},{pby:.3f},{pbz:.3f})", flush=True)

    stove_pts, stove_center, stove_top_z, stove_prompt = localize_stove(rgb1, depth_img1, K1, E1)
    if stove_center is None:
        print("ERROR: stove not found", flush=True)
    else:
        sx, sy, sz = float(stove_center[0]), float(stove_center[1]), float(stove_center[2])
        print(f"Stove ({stove_prompt}): c=({sx:.3f},{sy:.3f},{sz:.3f}) top_z={stove_top_z:.3f}", flush=True)

        hx_ext = float(handle_pts[:, 0].max() - handle_pts[:, 0].min())
        hy_ext = float(handle_pts[:, 1].max() - handle_pts[:, 1].min())
        long_axis_xy = None
        long_extent_xy = 0.05
        try:
            xy = handle_pts[:, :2] - handle_pts[:, :2].mean(axis=0)
            cov = np.cov(xy.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            long_axis_xy = eigvecs[:, -1]
            long_axis_xy = long_axis_xy / np.linalg.norm(long_axis_xy)
            long_extent_xy = 4.0 * float(np.sqrt(max(eigvals[-1], 1e-9)))
            angle_long = np.degrees(np.arctan2(long_axis_xy[1], long_axis_xy[0]))
            yaw_deg = -angle_long
            while yaw_deg > 90:
                yaw_deg -= 180
            while yaw_deg <= -90:
                yaw_deg += 180
            print(f"Handle PCA: angle={angle_long:.1f} → yaw={yaw_deg:.1f}", flush=True)
        except Exception as ex:
            print(f"PCA failed: {ex}, fallback", flush=True)
            yaw_deg = 90.0 if hy_ext > hx_ext else 0.0

        if abs(hx_ext - hy_ext) < 0.015:
            yaw_deg = 90.0

        grasp_quat = make_topdown_quat(yaw_deg=yaw_deg)

        toward_body = np.array([pbx - hx, pby - hy])
        tn = np.linalg.norm(toward_body)
        if tn > 1e-3:
            toward_body = toward_body / tn
        else:
            toward_body = np.array([0.0, -1.0])

        if long_axis_xy is not None:
            proj = float(np.dot(toward_body, long_axis_xy))
            bias_dir = proj * long_axis_xy
        else:
            bias_dir = toward_body * 0.5

        bias_mag = min(0.035, long_extent_xy * 0.30)
        bdn = bias_dir / max(np.linalg.norm(bias_dir), 1e-6)
        grasp_x = float(hx + bdn[0] * bias_mag)
        grasp_y = float(hy + bdn[1] * bias_mag)
        grasp_z = max(handle_top_z - 0.020, 0.005)

        print(f"Grasp plan: pos=({grasp_x:.3f},{grasp_y:.3f},{grasp_z:.3f}) yaw={yaw_deg:.1f}", flush=True)

        hover_z = 0.20
        j = solve_ik([grasp_x, grasp_y, hover_z], grasp_quat.tolist())
        if j is not None:
            move_to_joints(j)
        for z_step in [0.10, grasp_z]:
            j = solve_ik([grasp_x, grasp_y, z_step], grasp_quat.tolist())
            if j is not None:
                move_to_joints(j)
        close_gripper()

        obs_g = get_observation()
        gw_after = float(obs_g["robot_cartesian_pos"][-1])
        print(f"After close: gw={gw_after:.3f}", flush=True)

        if gw_after < 0.020:
            print("Air grasp — retrying", flush=True)
            alt_grasps = [
                (grasp_x, grasp_y, grasp_z + 0.005, yaw_deg),
                (grasp_x, grasp_y, grasp_z - 0.005, yaw_deg),
                (grasp_x, grasp_y, grasp_z + 0.010, yaw_deg),
                (hx, hy, grasp_z, yaw_deg),
                (hx, float(np.clip(handle_y_min + 0.04, hy - 0.04, hy + 0.02)), max(handle_top_z - 0.020, 0.005), 90.0),
                (hx, hy, grasp_z, 0.0),
            ]
            for (rx, ry, rz, ryaw) in alt_grasps:
                rz = max(rz, 0.003)
                rquat = make_topdown_quat(yaw_deg=ryaw)
                open_gripper()
                j = solve_ik([rx, ry, hover_z], rquat.tolist())
                if j is not None:
                    move_to_joints(j)
                j = solve_ik([rx, ry, rz], rquat.tolist())
                if j is not None:
                    move_to_joints(j)
                close_gripper()
                obs_r = get_observation()
                gw_after = float(obs_r["robot_cartesian_pos"][-1])
                print(f"  retry pos=({rx:.3f},{ry:.3f},{rz:.3f}) yaw={ryaw:.1f}: gw={gw_after:.3f}", flush=True)
                if gw_after >= 0.020:
                    grasp_x, grasp_y, grasp_z = rx, ry, rz
                    grasp_quat = rquat
                    yaw_deg = ryaw
                    break

        # Lift
        lift_z = max(grasp_z + 0.30, 0.50)
        for step_z in [grasp_z + 0.05, grasp_z + 0.15, lift_z]:
            j = solve_ik([grasp_x, grasp_y, step_z], grasp_quat.tolist())
            if j is not None:
                move_to_joints(j)
        obs_l = get_observation()
        gw_lift = float(obs_l["robot_cartesian_pos"][-1])
        print(f"After lift: gw={gw_lift:.3f}", flush=True)

        body_offset_x = grasp_x - pbx
        body_offset_y = grasp_y - pby
        target_x_stove = sx
        target_y_stove = sy
        raw_arm_target_x = target_x_stove + body_offset_x
        raw_arm_target_y = target_y_stove + body_offset_y
        arm_target_x = float(np.clip(raw_arm_target_x, 0.40, 0.75))
        arm_target_y = min(raw_arm_target_y, 0.30)
        print(f"Body offset=({body_offset_x:.3f},{body_offset_y:.3f}), arm=({arm_target_x:.3f},{arm_target_y:.3f})", flush=True)

        n_steps = 8
        for i in range(1, n_steps + 1):
            t = i / n_steps
            wx = grasp_x + t * (arm_target_x - grasp_x)
            wy = grasp_y + t * (arm_target_y - grasp_y)
            j = solve_ik([wx, wy, lift_z], grasp_quat.tolist())
            if j is not None:
                move_to_joints(j)
        obs_t = get_observation()
        print(f"At above-target: gw={float(obs_t['robot_cartesian_pos'][-1]):.3f}", flush=True)

        release_z_wrist = stove_top_z + 0.135 + 0.020
        print(f"Release wrist z = {release_z_wrist:.3f}", flush=True)

        n_d = 6
        for i in range(1, n_d + 1):
            t = i / n_d
            wz = lift_z + t * (release_z_wrist - lift_z)
            j = solve_ik([arm_target_x, arm_target_y, wz], grasp_quat.tolist())
            if j is not None:
                move_to_joints(j)
            if i == 3:
                obs_mid = get_observation()
                gw_mid = float(obs_mid["robot_cartesian_pos"][-1])
                if gw_mid < 0.020:
                    print(f"  WARN: pan slipped mid-descent", flush=True)
                    break
        obs_p = get_observation()
        ee_p = obs_p["robot_cartesian_pos"][:3]
        print(f"At release pos=({ee_p[0]:.3f},{ee_p[1]:.3f},{ee_p[2]:.3f}): gw={float(obs_p['robot_cartesian_pos'][-1]):.3f}", flush=True)

        open_gripper()
        for _ in range(10):
            get_observation()

        j = solve_ik([arm_target_x, arm_target_y, lift_z], grasp_quat.tolist())
        if j is not None:
            move_to_joints(j)
        try:
            goto_home_joint_position()
        except Exception:
            pass

print("Done", flush=True)
