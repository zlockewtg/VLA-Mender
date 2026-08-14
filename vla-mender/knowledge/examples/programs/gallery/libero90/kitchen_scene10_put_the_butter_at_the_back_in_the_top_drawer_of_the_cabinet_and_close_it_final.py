"""
Back-butter variant of KS10 task. Place butter at BACK of open top drawer,
then close drawer. Identical pipeline to the FRONT variant (v6) except
placement target Y is the BACK quarter of drawer interior:
    drawer_y_back = y_min + 0.25 * (y_max - y_min)

(BACK = more negative Y in this scene since drawer extends in -Y direction
from cabinet face.)
"""
import numpy as np
from scipy.spatial.transform import Rotation


TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
TRANSIT_Z = 0.40
TABLE_Z = -0.011

DRAWER_X_FALLBACK = 0.66
DRAWER_Y_BACK_FALLBACK = -0.21  # back quarter (more negative) of drawer
DRAWER_FLOOR_Z_FALLBACK = 0.155
DRAWER_TOP_Z_FALLBACK = 0.213
HANDLE_X_FALLBACK = 0.664
HANDLE_Y_FALLBACK = -0.063
HANDLE_Z_FALLBACK = 0.186


def make_topdown_quat(yaw_deg=0.0):
    R = (Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix()
         @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def localize_butters_all(rgb, depth, K, E, y_min_filter=-0.10, z_min_filter=-0.05, z_max_filter=0.06):
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    prompts = ["butter package", "small box", "rectangular box", "yellow box",
               "orange box", "flat box"]
    candidates = []
    seen_xy = []
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in sorted(masks, key=lambda x: -x['score'])[:8]:
            if m['score'] < 0.30:
                continue
            mask_arr = m['mask'].astype(np.uint8)
            npix = int(mask_arr.sum())
            if npix < 50 or npix > 6000:
                continue
            pts = mask_to_world_points(mask_arr, depth, K, E)
            if pts is None or len(pts) < 30:
                continue
            cx = float(np.median(pts[:, 0]))
            cy = float(np.median(pts[:, 1]))
            cz = float(np.median(pts[:, 2]))
            if not (0.45 <= cx <= 0.85):
                continue
            if not (y_min_filter <= cy <= 0.30):
                continue
            if not (z_min_filter <= cz <= z_max_filter):
                continue
            is_dup = any(abs(cx - sx) < 0.03 and abs(cy - sy) < 0.03 for sx, sy in seen_xy)
            if is_dup:
                continue
            seen_xy.append((cx, cy))
            ys, xs = np.where(mask_arr > 0)
            mc = rgb[ys, xs].mean(axis=0)
            cr = (mc[0] + mc[1]) / max(mc[2], 1.0)
            top_z = float(np.percentile(pts[:, 2], 95))
            bot_z = float(np.percentile(pts[:, 2], 5))
            try:
                obb = get_oriented_bounding_box_from_3d_points(pts)
                obb_center = np.array(obb['center'])
                obb_extent = np.array(obb['extent'])
            except Exception:
                obb_center = np.array([cx, cy, cz])
                obb_extent = np.array([0.05, 0.04, 0.02])
            candidates.append({
                'cx': cx, 'cy': cy, 'cz': cz,
                'pts': pts, 'mask': mask_arr,
                'top_z': top_z, 'bot_z': bot_z,
                'color_ratio': cr, 'score': m['score'], 'prompt': prompt,
                'obb_center': obb_center, 'obb_extent': obb_extent,
            })
    butters = [c for c in candidates if c['color_ratio'] > 3.5]
    return butters


def localize_drawer_interior(rgb, depth, K, E):
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    prompts = ["drawer interior", "open drawer", "drawer opening", "inside the drawer"]
    best = None
    best_extent = -1.0
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in sorted(masks, key=lambda x: -x['score'])[:5]:
            if m['score'] < 0.10:
                continue
            mask_arr = m['mask'].astype(np.uint8)
            pts = mask_to_world_points(mask_arr, depth, K, E)
            if pts is None or len(pts) < 200:
                continue
            interior = pts[pts[:, 2] > 0.10]
            if len(interior) < 100:
                continue
            cx = float(np.median(interior[:, 0]))
            cy = float(np.median(interior[:, 1]))
            cz = float(np.median(interior[:, 2]))
            if not (0.50 <= cx <= 0.80):
                continue
            if not (-0.30 <= cy <= -0.05):
                continue
            if interior[:, 2].max() < 0.18:
                continue
            x_min = float(interior[:, 0].min())
            x_max = float(interior[:, 0].max())
            y_min = float(interior[:, 1].min())
            y_max = float(interior[:, 1].max())
            if (x_max - x_min) > 0.30 or (y_max - y_min) > 0.30:
                continue
            z20 = float(np.percentile(interior[:, 2], 20))
            floor_pts = interior[interior[:, 2] <= z20 + 0.005]
            floor_z = float(floor_pts[:, 2].mean()) if len(floor_pts) > 0 else float(interior[:, 2].min())
            top_z = float(interior[:, 2].max())
            extent = (x_max - x_min) + (y_max - y_min)
            cand = {
                'cx': cx, 'cy': cy, 'cz': cz,
                'x_range': (x_min, x_max), 'y_range': (y_min, y_max),
                'floor_z': floor_z, 'top_z': top_z,
                'score': m['score'], 'prompt': prompt,
            }
            if extent > best_extent:
                best_extent = extent
                best = cand
    return best


def localize_drawer_handle(rgb, depth, K, E):
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    prompts = ["metal handle", "drawer handle", "handle", "drawer pull"]
    candidates = []
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in sorted(masks, key=lambda x: -x['score'])[:6]:
            if m['score'] < 0.30:
                continue
            mask_arr = m['mask'].astype(np.uint8)
            pts = mask_to_world_points(mask_arr, depth, K, E)
            if pts is None or len(pts) < 20:
                continue
            c = pts.mean(axis=0)
            if not (0.55 <= c[0] <= 0.80):
                continue
            if not (-0.30 <= c[1] <= 0.05):
                continue
            if not (0.10 <= c[2] <= 0.25):
                continue
            z85 = np.percentile(pts[:, 2], 85)
            bar_pts = pts[pts[:, 2] >= z85]
            if len(bar_pts) < 10:
                continue
            handle_center = bar_pts.mean(axis=0)
            candidates.append({
                'score': m['score'], 'center': handle_center,
                'bar_pts': bar_pts, 'all_pts': pts, 'prompt': prompt,
            })
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x['center'][2])
    return candidates[0]


def attempt_pick(grasp_xy, grasp_z, gquat, label):
    open_gripper()
    try:
        j_warm = solve_ik([grasp_xy[0], grasp_xy[1], 0.05], gquat.tolist())
        if j_warm is not None: move_to_joints(j_warm)
    except Exception:
        pass
    descent_zs = [0.20, 0.12, 0.08, 0.04, 0.02, 0.00, grasp_z]
    for tz in descent_zs:
        j = solve_ik([grasp_xy[0], grasp_xy[1], tz], gquat.tolist())
        if j is not None:
            try: move_to_joints(j)
            except Exception: pass
    for _ in range(4):
        j = solve_ik([grasp_xy[0], grasp_xy[1], grasp_z], gquat.tolist())
        if j is not None:
            try: move_to_joints(j)
            except Exception: pass
    obs_d = get_observation()
    wrist_d = obs_d['robot_cartesian_pos'][:3]
    print(f"  [{label}] At grasp: wrist=[{wrist_d[0]:.3f},{wrist_d[1]:.3f},{wrist_d[2]:.3f}]", flush=True)
    close_gripper()
    obs_g = get_observation()
    gw_close = float(obs_g['robot_cartesian_pos'][-1])
    lift_pos = [grasp_xy[0], grasp_xy[1], TRANSIT_Z]
    try:
        j = solve_ik(lift_pos, gquat.tolist())
        if j is not None:
            move_to_joints(j)
    except Exception as e:
        print(f"  [{label}] lift failed: {e}", flush=True)
    obs_l = get_observation()
    gw_lifted = float(obs_l['robot_cartesian_pos'][-1])
    wrist_l = obs_l['robot_cartesian_pos'][:3]
    print(f"  [{label}] gw_close={gw_close:.3f} gw_lifted={gw_lifted:.3f}", flush=True)
    return gw_lifted


print(f"\n=== TASK: {env.handle.task_language} ===", flush=True)

goto_home_joint_position()
open_gripper(); close_gripper(); open_gripper(); close_gripper(); open_gripper()

obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if depth.ndim == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]

butters = localize_butters_all(rgb, depth_img, K, E)
if not butters:
    raise RuntimeError("No butter detected")
# "back" task targets the BACK (more negative X = smaller X = further from camera) butter.
# BDDL tracks the back butter specifically. Sort ascending = smallest X first = back butter.
butters.sort(key=lambda b: b['cx'])
butter = butters[0]
print(f"BACK butter (target — picked): xy=[{butter['cx']:.3f},{butter['cy']:.3f}]", flush=True)

drawer = localize_drawer_interior(rgb, depth_img, K, E)
if drawer is None:
    drawer_x = DRAWER_X_FALLBACK
    drawer_y_back = DRAWER_Y_BACK_FALLBACK
    drawer_floor_z = DRAWER_FLOOR_Z_FALLBACK
else:
    drawer_x = (drawer['x_range'][0] + drawer['x_range'][1]) / 2.0
    y_min, y_max = drawer['y_range']
    # BACK = more negative Y (deeper into cabinet). Use 1/3 from back rather than 1/4 to keep
    # within IK reach. drawer cavity y_range typically ~[-0.227, -0.106], 1/3 from back ≈ -0.187
    drawer_y_back = y_min + 0.33 * (y_max - y_min)
    drawer_floor_z = drawer['floor_z']
print(f"PLACE (BACK): x={drawer_x:.3f} y={drawer_y_back:.3f} floor={drawer_floor_z:.3f}", flush=True)

obb_xy = (float(butter['obb_center'][0]), float(butter['obb_center'][1]))
base_grasp_z = max(butter['bot_z'] + 0.005, butter['top_z'] - 0.012)

quat_yaw0 = make_topdown_quat(0.0)

GW_OK = 0.05
attempts = [
    (obb_xy, base_grasp_z - 0.020, quat_yaw0, "obb_y0_deep020"),
    (obb_xy, base_grasp_z, quat_yaw0, "obb_y0_z0"),
    (obb_xy, base_grasp_z - 0.010, quat_yaw0, "obb_y0_deep010"),
]

best_gw_l = 0.0
chosen_quat = quat_yaw0
chosen_xy = obb_xy
for attempt_xy, attempt_z, attempt_quat, label in attempts:
    gw_l = attempt_pick(attempt_xy, attempt_z, attempt_quat, label)
    if gw_l > best_gw_l:
        best_gw_l = gw_l
        chosen_xy = attempt_xy
        chosen_quat = attempt_quat
    if gw_l >= GW_OK:
        print(f"GRASP SUCCESS [{label}]: gw_l={gw_l:.3f}", flush=True)
        break
    else:
        try:
            goto_home_joint_position()
            open_gripper()
        except Exception:
            pass

if best_gw_l < GW_OK:
    print(f"  WARNING: all grasp attempts failed (best gw={best_gw_l:.3f})", flush=True)

# Re-tighten grip — call close_gripper again to ensure full force
close_gripper()
obs_pre_transit = get_observation()
gw_pre_transit = float(obs_pre_transit['robot_cartesian_pos'][-1])
print(f"After re-close: gw={gw_pre_transit:.3f}", flush=True)

# Transport with intermediate waypoints to maintain grip
butter_x, butter_y = chosen_xy[0], chosen_xy[1]

# WP1: above butter at TRANSIT_Z
# WP2: midpoint between butter and drawer at TRANSIT_Z
# WP3: above drawer (BACK target) at TRANSIT_Z
mid_xy = ((butter_x + drawer_x) / 2.0, (butter_y + drawer_y_back) / 2.0)

print(f"Transport waypoints (BACK target):", flush=True)
# WP1: midpoint at TRANSIT_Z; WP2: above drawer BACK
for tag, wx, wy in [("mid", mid_xy[0], mid_xy[1]), ("above_drawer_back", drawer_x, drawer_y_back)]:
    j = solve_ik([wx, wy, TRANSIT_Z], chosen_quat.tolist())
    if j is not None:
        try:
            move_to_joints(j)
            close_gripper()  # re-tighten
            obs_wp = get_observation()
            gw_wp = float(obs_wp['robot_cartesian_pos'][-1])
            print(f"  WP[{tag}] xy=({wx:.3f},{wy:.3f}) gw={gw_wp:.3f}", flush=True)
        except Exception as e:
            print(f"  WP[{tag}] failed: {e}", flush=True)

# Place: multi-pass descent at BACK to push wrist as low as IK allows
place_z = drawer_floor_z + 0.05
# More descent levels to push IK clamp lower (mirror grasp's multi-pass approach)
descent_tcp = [0.32, 0.30, 0.28, 0.25, 0.22, 0.20, 0.18, 0.15, place_z]
for tz in descent_tcp:
    j = solve_ik([drawer_x, drawer_y_back, tz], chosen_quat.tolist())
    if j is not None:
        try: move_to_joints(j)
        except Exception: break
# Multi-pass at final z to settle
for _ in range(4):
    j = solve_ik([drawer_x, drawer_y_back, place_z], chosen_quat.tolist())
    if j is not None:
        try: move_to_joints(j)
        except Exception: pass

obs_pp = get_observation()
print(f"At place (BACK): wrist=[{obs_pp['robot_cartesian_pos'][0]:.3f},{obs_pp['robot_cartesian_pos'][1]:.3f},{obs_pp['robot_cartesian_pos'][2]:.3f}] gw={obs_pp['robot_cartesian_pos'][-1]:.3f}", flush=True)

open_gripper()
print("Released at BACK.", flush=True)
for _ in range(6):
    get_observation()

# Lift away — go in +Y direction first (toward camera) to avoid pushing drawer
try:
    j = solve_ik([drawer_x, drawer_y_back + 0.05, place_z + 0.03], chosen_quat.tolist())
    if j is not None: move_to_joints(j)
except Exception: pass
# Then lift up
try:
    j = solve_ik([drawer_x, drawer_y_back + 0.10, TRANSIT_Z], TOP_DOWN_QUAT.tolist())
    if j is not None: move_to_joints(j)
except Exception: pass

# Retreat further in +Y direction before closing drawer
try:
    j = solve_ik([drawer_x, drawer_y_back + 0.25, TRANSIT_Z], TOP_DOWN_QUAT.tolist())
    if j is not None: move_to_joints(j)
except Exception: pass

# Close drawer
print("\n=== PHASE 6: CLOSE DRAWER ===", flush=True)
close_gripper()

obs2 = get_observation()
rgb2 = obs2["agentview"]["images"]["rgb"]
depth2 = obs2["agentview"]["images"]["depth"]
depth2_img = depth2[:, :, 0] if depth2.ndim == 3 else depth2
K2 = obs2["agentview"]["intrinsics"]
E2 = obs2["agentview"]["pose_mat"]

handle = localize_drawer_handle(rgb2, depth2_img, K2, E2)
if handle is None:
    handle_x = HANDLE_X_FALLBACK
    handle_y = HANDLE_Y_FALLBACK
    handle_z = HANDLE_Z_FALLBACK
    bar_yaw = 0.0
else:
    handle_center = handle['center']
    handle_x = float(handle_center[0])
    handle_y = float(handle_center[1])
    handle_z = float(handle_center[2])
    bar_pts = handle['bar_pts']
    bar_dx = bar_pts[:, 0].max() - bar_pts[:, 0].min()
    bar_dy = bar_pts[:, 1].max() - bar_pts[:, 1].min()
    bar_yaw = float(np.degrees(np.arctan2(bar_dy, bar_dx)))
print(f"HANDLE: c=[{handle_x:.3f},{handle_y:.3f},{handle_z:.3f}] yaw={bar_yaw:.1f}", flush=True)

push_dir = np.array([0.0, -1.0, 0.0])
if drawer is not None:
    diff = np.array([drawer['cx'] - handle_x, drawer['cy'] - handle_y])
    norm = np.linalg.norm(diff)
    if norm > 1e-3:
        push_dir = np.array([float(diff[0] / norm), float(diff[1] / norm), 0.0])

push_quat = make_topdown_quat(bar_yaw + 90)
PRE_OFFSET = 0.06
pre_pos = [handle_x - push_dir[0] * PRE_OFFSET, handle_y - push_dir[1] * PRE_OFFSET, handle_z + 0.10]
j = solve_ik(pre_pos, push_quat.tolist())
if j is not None:
    try: move_to_joints(j)
    except Exception: pass
push_z = handle_z - 0.04
contact_pos = np.array([handle_x - push_dir[0] * 0.03, handle_y - push_dir[1] * 0.03, push_z])
j = solve_ik(contact_pos.tolist(), push_quat.tolist())
if j is not None:
    try: move_to_joints(j)
    except Exception: pass
PUSH_DISTANCE = 0.25
for i in range(1, 9):
    wp = contact_pos + push_dir * (PUSH_DISTANCE * (i / 8.0))
    j = solve_ik(wp.tolist(), push_quat.tolist())
    if j is not None:
        try: move_to_joints(j)
        except Exception: break

obs_end = get_observation()
print(f"\nDone. Final wrist=[{obs_end['robot_cartesian_pos'][0]:.3f},{obs_end['robot_cartesian_pos'][1]:.3f},{obs_end['robot_cartesian_pos'][2]:.3f}]", flush=True)
