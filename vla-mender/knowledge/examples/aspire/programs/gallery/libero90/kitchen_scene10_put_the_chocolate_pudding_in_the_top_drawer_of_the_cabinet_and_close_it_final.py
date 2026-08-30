"""
Fix code v1: libero_90 / KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it

Compound task:
  Phase 1: Pick chocolate pudding (small brown box) and place inside the OPEN top drawer.
  Phase 2: Close the top drawer using closed-gripper push-paddle technique.

Scene findings (seed 51):
- Top drawer is OPEN at init.
- Drawer interior cavity: x=[0.55,0.626], y=[-0.253,-0.099], floor_z≈0.155, top_z≈0.213.
- Pudding box (flat): SAM3 "small box"(0.918) "brown box"(0.730) "chocolate"(0.65).
  Centroid ~ [0.692, 0.057, z=[-0.011,0.017]].  Object is ~3cm tall.
- Drawer handle ("metal handle" 0.898): centroid [0.666, -0.067, 0.18].
- Cabinet on -Y side (relative to robot at Y=0). Pull-direction = +Y, close-direction = -Y.

Strategy:
  1. Localize pudding via SAM3 (multi-prompt). Filter to workspace x∈[0.45,0.85], y∈[-0.10,0.30], z<0.06.
  2. Localize drawer cavity via "drawer interior" / "open drawer" → place midpoint inside.
  3. Pick pudding using GraspNet top-down grasp; flat box → use TOP_DOWN_QUAT directly.
  4. Lift, transit above drawer interior midpoint, descend, release.
  5. Re-localize handle. Push-paddle close: closed gripper pushes drawer face toward -Y.
"""
import numpy as np
from scipy.spatial.transform import Rotation


TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
TRANSIT_Z = 0.40
TABLE_Z = -0.011

# Fallback constants (from seed 51 probe) — used if SAM3 localization fails.
DRAWER_X_FALLBACK = 0.59
DRAWER_Y_FALLBACK = -0.18
DRAWER_FLOOR_Z_FALLBACK = 0.182
DRAWER_TOP_Z_FALLBACK = 0.213
HANDLE_X_FALLBACK = 0.666
HANDLE_Y_FALLBACK = -0.067
HANDLE_Z_FALLBACK = 0.180


def make_topdown_quat(yaw_deg=0.0):
    """Top-down gripper orientation, rotated yaw_deg around world Z."""
    R = (Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix()
         @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def localize_pudding(rgb, depth, K, E):
    """Find chocolate pudding among 3 brown boxes on the table.
    Strategy: collect all candidate boxes, pick the THICKEST one (chocolate pudding has
    a distinct ~2-3cm height vs other distractor boxes at ~1cm). If no thick box,
    fall back to highest score."""
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    prompts = ["small box", "brown box", "chocolate", "small brown container",
               "flat brown box", "pudding box", "chocolate pudding"]
    candidates = []  # all boxes that pass workspace filter
    seen_xy = []  # dedupe by XY (within 3cm)
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in sorted(masks, key=lambda x: -x['score'])[:6]:
            if m['score'] < 0.20:
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
            if not (-0.10 <= cy <= 0.30):
                continue
            if not (-0.03 <= cz <= 0.06):
                continue
            # Dedupe: skip if a candidate already exists within 3cm XY
            is_dup = False
            for sx, sy in seen_xy:
                if abs(cx - sx) < 0.03 and abs(cy - sy) < 0.03:
                    is_dup = True
                    break
            if is_dup:
                continue
            seen_xy.append((cx, cy))
            top_z = float(np.percentile(pts[:, 2], 95))
            base_z = float(np.percentile(pts[:, 2], 10))
            thickness = top_z - base_z
            score = m['score']
            candidates.append({
                'cx': cx, 'cy': cy, 'cz': cz,
                'pts': pts, 'mask': mask_arr,
                'top_z': top_z, 'base_z': base_z, 'thickness': thickness,
                'score': score, 'prompt': prompt,
            })
    if not candidates:
        return None
    # Print all candidates for visibility
    for c in sorted(candidates, key=lambda x: -x['thickness']):
        print(f"  CAND: xy=[{c['cx']:.3f},{c['cy']:.3f}] top_z={c['top_z']:.3f} thickness={c['thickness']:.3f} score={c['score']:.3f}", flush=True)
    # Prefer the THICKEST box (chocolate pudding is taller than distractor flat boxes).
    # Tiebreaker: highest score.
    candidates.sort(key=lambda c: (-c['thickness'], -c['score']))
    return candidates[0]


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
    """Returns dict with handle center + bar pts. Filters phantoms by workspace bounds."""
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
            # Workspace filter: kill phantom reflections
            if not (0.55 <= c[0] <= 0.80):
                continue
            if not (-0.30 <= c[1] <= 0.05):
                continue
            if not (0.10 <= c[2] <= 0.25):
                continue
            # Top drawer handle is highest Z handle in the cluster
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
    # Pick the candidate with highest center Z (top drawer handle)
    candidates.sort(key=lambda x: -x['center'][2])
    return candidates[0]


print(f"\n=== TASK: {env.handle.task_language} ===", flush=True)

# ===== Phase 0: Initial observation =====
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if depth.ndim == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]

pudding = localize_pudding(rgb, depth_img, K, E)
if pudding is None:
    raise RuntimeError("PUDDING: not detected")
print(f"PUDDING: c=[{pudding['cx']:.3f},{pudding['cy']:.3f},{pudding['cz']:.3f}] base_z={pudding['base_z']:.3f} top_z={pudding['top_z']:.3f} prompt='{pudding['prompt']}' score={pudding['score']:.3f}", flush=True)

drawer = localize_drawer_interior(rgb, depth_img, K, E)
if drawer is None:
    drawer_x = DRAWER_X_FALLBACK
    drawer_y = DRAWER_Y_FALLBACK
    drawer_floor_z = DRAWER_FLOOR_Z_FALLBACK
    drawer_top_z = DRAWER_TOP_Z_FALLBACK
    print(f"DRAWER: not detected, fallback x={drawer_x:.3f} y={drawer_y:.3f}", flush=True)
else:
    # Place TOWARD THE BACK of the drawer (y_min side, away from front handle).
    # This avoids dropping pudding on top of any object near the front (e.g., the bowl
    # in the same scene that's already inside the drawer near front).
    drawer_x = (drawer['x_range'][0] + drawer['x_range'][1]) / 2.0
    # Bias toward y_min (back of drawer): y_min + 25% of range
    y_min, y_max = drawer['y_range']
    drawer_y = y_min + 0.25 * (y_max - y_min)
    drawer_floor_z = drawer['floor_z']
    drawer_top_z = drawer['top_z']
    print(f"DRAWER: c=[{drawer['cx']:.3f},{drawer['cy']:.3f},{drawer['cz']:.3f}] x={drawer['x_range']} y={drawer['y_range']} floor_z={drawer_floor_z:.3f} top_z={drawer_top_z:.3f}", flush=True)

print(f"PLACE TARGET: x={drawer_x:.3f} y={drawer_y:.3f} drawer_top_z={drawer_top_z:.3f}", flush=True)

# ===== Phase 1: Plan grasp =====
grasps, scores = plan_grasp(depth, K, pudding['mask'])
print(f"GraspNet: {len(grasps)} grasps", flush=True)

# Pick best top-down grasp (TCP Z-axis pointing down)
chosen_xy = None
chosen_quat = TOP_DOWN_QUAT
chosen_z = pudding['top_z']

# Use simple top-down quaternion. Pudding box is small (4-7cm) — fingers can grip
# along either short axis. Avoid OBB-derived yaw which sometimes finds bad arm config.
grasp_quat = make_topdown_quat(0.0)
pud_yaw = 0.0

grasp_pose_z = None  # Z from GraspNet pose if available
if len(grasps) > 0:
    try:
        best_pose_world, best_score = select_top_down_grasp(grasps, scores, E)
    except Exception as e:
        print(f"select_top_down_grasp: {e}", flush=True)
        best_pose_world = None
    if best_pose_world is not None and isinstance(best_pose_world, np.ndarray) and best_pose_world.shape == (4, 4):
        gpos, gquat = decompose_transform(best_pose_world)
        chosen_xy = (float(gpos[0]), float(gpos[1]))
        grasp_pose_z = float(gpos[2])
        # Filter sanity: grasp XY within ~4cm of pudding XY
        if abs(chosen_xy[0] - pudding['cx']) > 0.04 or abs(chosen_xy[1] - pudding['cy']) > 0.04:
            print(f"  GraspNet pose far from pudding ({chosen_xy} vs {pudding['cx']:.3f},{pudding['cy']:.3f}) → falling back to centroid", flush=True)
            chosen_xy = None
            grasp_pose_z = None
        else:
            print(f"  GraspNet top-down grasp: xy={chosen_xy} z={grasp_pose_z:.3f} score={best_score:.3f}", flush=True)
    else:
        # No top-down grasp found, fall back to GraspNet best by score (XY only)
        scores_arr = np.array(scores)
        best_i = int(np.argmax(scores_arr))
        g_world = E @ grasps[best_i]
        gp = g_world[:3, 3]
        if abs(gp[0] - pudding['cx']) <= 0.04 and abs(gp[1] - pudding['cy']) <= 0.04:
            chosen_xy = (float(gp[0]), float(gp[1]))
            grasp_pose_z = float(gp[2])
            print(f"  GraspNet best-by-score grasp (non-topdown): xy={chosen_xy} z={grasp_pose_z:.3f} score={scores_arr[best_i]:.3f}", flush=True)

if chosen_xy is None:
    chosen_xy = (pudding['cx'], pudding['cy'])
    print(f"  Using centroid grasp XY: {chosen_xy}", flush=True)

# Grasp Z target — for very flat pudding, use a slightly higher pinch Z
# so fingers close around the box rather than colliding with the table.
# Empirical: TCP target z=-0.003 → wrist z=0.118. Fingertip ≈ TCP target.
thickness = pudding['top_z'] - pudding['base_z']
if thickness > 0.012:
    grasp_z = max(pudding['top_z'] - 0.020, TABLE_Z + 0.005)
else:
    # Very flat pudding: pinch midway through the box.
    grasp_z = max(pudding['top_z'] - 0.005, TABLE_Z + 0.005)
print(f"GRASP TARGET: xy={chosen_xy}, z={grasp_z:.3f} (top_z={pudding['top_z']:.3f}, base_z={pudding['base_z']:.3f}, thickness={thickness:.3f})", flush=True)

# ===== Phase 2: Pick (with retry) =====
def attempt_pick(grasp_xy, gz, gquat, label):
    """Attempt: descend → close → lift. Uses solve_ik+move_to_joints sequential descent.
    Uses an explicit warm-up via low-Z reach to prime IK toward LOW-arm branch."""
    open_gripper()
    # Warm-up: explicitly reach a low-Z point to prime IK toward LOW-arm branch.
    # Without this, IK warm-start from home picks HIGH-arm config that can't reach floor.
    try:
        j_warm = solve_ik(np.array([grasp_xy[0], grasp_xy[1], 0.05]), gquat)
        if j_warm is not None: move_to_joints(j_warm)
    except Exception:
        pass
    # Sequential descent
    descent_seq = [0.20, 0.12, 0.08, 0.04, 0.00, gz]
    for tz in descent_seq:
        pos = np.array([grasp_xy[0], grasp_xy[1], tz])
        j = solve_ik(pos, gquat)
        if j is not None:
            try:
                move_to_joints(j)
            except Exception as e:
                print(f"  [{label}] descent tz={tz:.3f} failed: {e}", flush=True)
                break
    obs_d = get_observation()
    wrist_d = obs_d['robot_cartesian_pos'][:3]
    print(f"  [{label}] At grasp: wrist=[{wrist_d[0]:.3f},{wrist_d[1]:.3f},{wrist_d[2]:.3f}]", flush=True)
    close_gripper()
    obs_g = get_observation()
    gw_close = float(obs_g['robot_cartesian_pos'][-1])
    print(f"  [{label}] After close: gw={gw_close:.3f}", flush=True)
    lift_pos = np.array([grasp_xy[0], grasp_xy[1], TRANSIT_Z])
    j = solve_ik(lift_pos, gquat)
    if j is not None:
        try:
            move_to_joints(j)
        except Exception as e:
            print(f"  [{label}] lift failed: {e}", flush=True)
    obs_l = get_observation()
    gw_lifted = float(obs_l['robot_cartesian_pos'][-1])
    wrist_l = obs_l['robot_cartesian_pos'][:3]
    print(f"  [{label}] After lift: gw={gw_lifted:.3f} wrist=[{wrist_l[0]:.3f},{wrist_l[1]:.3f},{wrist_l[2]:.3f}]", flush=True)
    return gw_lifted

GW_OK = 0.020

goto_home_joint_position()

# Build attempts: (xy, gz, gquat, label) — vary yaw and depth to handle floor-limited reach.
quat_yaw0 = make_topdown_quat(0.0)
quat_yaw90 = make_topdown_quat(90.0)
quat_pud = grasp_quat  # OBB-derived yaw

# For VERY flat pudding far from robot (Y > 0.15), workspace reach limits grip.
# Strategy: PRE-NUDGE the pudding toward the robot (smaller Y) using closed gripper as paddle,
# then re-localize and grasp from a better arm config.
THIN_PUDDING = (pudding['top_z'] - pudding['base_z']) < 0.012
FAR_PUDDING = pudding['cy'] > 0.13

if THIN_PUDDING and FAR_PUDDING:
    print(f"PRE-NUDGE: thin pudding far from robot, pushing toward Y=0", flush=True)
    # Approach with closed gripper from +Y side (behind pudding from robot's view)
    close_gripper()
    # Pre-position above pudding +Y (further from robot than pudding by 5cm)
    nudge_pre = np.array([pudding['cx'], pudding['cy'] + 0.05, 0.10])
    try:
        goto_pose(nudge_pre, quat_yaw0, z_approach=0.10)
    except Exception as e:
        print(f"  nudge_pre: {e}", flush=True)
    # Descend behind pudding
    nudge_contact = np.array([pudding['cx'], pudding['cy'] + 0.05, 0.02])
    try:
        goto_pose(nudge_contact, quat_yaw0)
    except Exception as e:
        print(f"  nudge_contact: {e}", flush=True)
    # Push toward robot in -Y direction by 0.10m
    nudge_push = np.array([pudding['cx'], pudding['cy'] - 0.05, 0.02])
    try:
        goto_pose(nudge_push, quat_yaw0)
    except Exception as e:
        print(f"  nudge_push: {e}", flush=True)
    # Lift away
    try:
        goto_pose(np.array([pudding['cx'], pudding['cy'] - 0.05, 0.30]), quat_yaw0)
    except Exception:
        pass
    goto_home_joint_position()
    # Re-localize pudding after nudge
    obs_n = get_observation()
    rgb_n = obs_n["agentview"]["images"]["rgb"]
    depth_n = obs_n["agentview"]["images"]["depth"]
    d_n = depth_n[:, :, 0] if depth_n.ndim == 3 else depth_n
    K_n = obs_n["agentview"]["intrinsics"]
    E_n = obs_n["agentview"]["pose_mat"]
    pudding_new = localize_pudding(rgb_n, d_n, K_n, E_n)
    if pudding_new is not None:
        old_y = pudding['cy']
        pudding = pudding_new
        chosen_xy = (pudding['cx'], pudding['cy'])
        print(f"POST-NUDGE PUDDING: c=[{pudding['cx']:.3f},{pudding['cy']:.3f},{pudding['cz']:.3f}] (old y={old_y:.3f})", flush=True)
        # Recompute grasp_z for new pudding height
        thickness_new = pudding['top_z'] - pudding['base_z']
        if thickness_new > 0.012:
            grasp_z = max(pudding['top_z'] - 0.020, TABLE_Z + 0.005)
        else:
            grasp_z = max(pudding['top_z'] - 0.005, TABLE_Z + 0.005)

# For VERY flat pudding (thickness < 0.012), use deep grasp_z to ensure fingers wrap below box top.
deep_gz = max(pudding['base_z'] - 0.025, -0.030)
attempts = [(chosen_xy, grasp_z, quat_pud, "primary")]
# Retry: pure centroid (often better IK warm-start)
attempts.append(((pudding['cx'], pudding['cy']), grasp_z, quat_pud, "centroid"))
# Retry: centroid + small XY jitter (try perturbed start)
attempts.append(((pudding['cx'] + 0.01, pudding['cy'] - 0.005), grasp_z, quat_pud, "jit_pos"))
# Retry: centroid + yaw=90 (fingers along X)
attempts.append(((pudding['cx'], pudding['cy']), grasp_z, quat_yaw90, "centroid_y90"))

gw_l = 0.0
chosen_quat = quat_pud
for attempt_xy, attempt_z, attempt_quat, label in attempts:
    gw_l = attempt_pick(attempt_xy, attempt_z, attempt_quat, label)
    if gw_l >= GW_OK:
        chosen_xy = attempt_xy
        chosen_quat = attempt_quat
        print(f"GRASP SUCCESS [{label}]: gw_l={gw_l:.3f}", flush=True)
        break
    else:
        # Recovery: lift away and reset gripper for next attempt
        try:
            goto_home_joint_position()
            open_gripper()
        except Exception:
            pass

if gw_l < GW_OK:
    print(f"  WARNING: all grasp attempts failed (best gw={gw_l:.3f}), attempting placement anyway", flush=True)

# ===== Phase 3: Transport above drawer =====
above_drawer = np.array([drawer_x, drawer_y, TRANSIT_Z])
j = solve_ik(above_drawer, chosen_quat)
if j is not None:
    move_to_joints(j)

# ===== Phase 4: Place inside drawer =====
# Descend just above drawer floor, ~5cm above floor
place_z = drawer_floor_z + 0.05
descent_tcp = [0.30, 0.27, 0.24, 0.21, place_z]
for tz in descent_tcp:
    pos = np.array([drawer_x, drawer_y, tz])
    j = solve_ik(pos, chosen_quat)
    if j is not None:
        try:
            move_to_joints(j)
        except Exception as e:
            print(f"  place descent tz={tz:.3f}: {e}", flush=True)
            break

obs_pp = get_observation()
wrist_pp = obs_pp['robot_cartesian_pos'][:3]
gw_pp = float(obs_pp['robot_cartesian_pos'][-1])
print(f"At place: wrist=[{wrist_pp[0]:.3f},{wrist_pp[1]:.3f},{wrist_pp[2]:.3f}] gw={gw_pp:.3f}", flush=True)

# Release
open_gripper()
print("Released.", flush=True)

# Settle a few frames
for _ in range(6):
    get_observation()

# ===== Phase 5: Lift away (avoid sweeping through drawer) =====
# Lift straight up from drop point. Do NOT use goto_home (would sweep through drawer).
lift_away = np.array([drawer_x, drawer_y, TRANSIT_Z])
try:
    j = solve_ik(lift_away, TOP_DOWN_QUAT)
    if j is not None:
        move_to_joints(j)
except Exception as e:
    print(f"Lift away: {e}", flush=True)

# Retreat in +Y direction toward robot to clear drawer interior before approaching handle
retreat_pos = np.array([drawer_x, drawer_y + 0.15, TRANSIT_Z])
try:
    j = solve_ik(retreat_pos, TOP_DOWN_QUAT)
    if j is not None:
        move_to_joints(j)
except Exception as e:
    print(f"Retreat: {e}", flush=True)



# ===== Phase 6: Close drawer (push-paddle technique) =====
print("\n=== PHASE 6: CLOSE DRAWER ===", flush=True)

# Close gripper to use as paddle
close_gripper()

# Re-localize handle (drawer may have shifted slightly with pudding placed)
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
    bar_yaw = 0.0  # bar is mostly along X
    print(f"HANDLE: not detected, fallback [{handle_x:.3f},{handle_y:.3f},{handle_z:.3f}]", flush=True)
else:
    handle_center = handle['center']
    handle_x = float(handle_center[0])
    handle_y = float(handle_center[1])
    handle_z = float(handle_center[2])
    bar_pts = handle['bar_pts']
    bar_dx = bar_pts[:, 0].max() - bar_pts[:, 0].min()
    bar_dy = bar_pts[:, 1].max() - bar_pts[:, 1].min()
    # bar orientation in XY plane (degrees)
    bar_yaw = float(np.degrees(np.arctan2(bar_dy, bar_dx)))
    print(f"HANDLE: c=[{handle_x:.3f},{handle_y:.3f},{handle_z:.3f}] bar_dx={bar_dx:.3f} bar_dy={bar_dy:.3f} yaw={bar_yaw:.1f}° prompt='{handle['prompt']}' score={handle['score']:.3f}", flush=True)

# Push direction = from handle toward cabinet body (-Y for this scene where drawer pulls +Y)
# Compute from drawer geometry if available, otherwise use -Y direction
push_dir = np.array([0.0, -1.0, 0.0])
if drawer is not None:
    # cabinet body is at smaller Y than handle (drawer interior centroid)
    drawer_xy = np.array([drawer['cx'], drawer['cy']])
    handle_xy = np.array([handle_x, handle_y])
    diff = drawer_xy - handle_xy  # from handle toward drawer interior (= toward cabinet)
    norm = np.linalg.norm(diff)
    if norm > 1e-3:
        push_dir = np.array([float(diff[0] / norm), float(diff[1] / norm), 0.0])
print(f"PUSH DIR: {push_dir}", flush=True)

# Gripper orientation: fingers perpendicular to handle bar
push_quat = make_topdown_quat(bar_yaw + 90)

# Pre-approach 15cm above handle on the OPPOSITE side from push_dir (i.e., on robot side)
PRE_OFFSET = 0.06  # 6cm in front of handle (opposite to push_dir)
pre_pos = np.array([
    handle_x - push_dir[0] * PRE_OFFSET,
    handle_y - push_dir[1] * PRE_OFFSET,
    handle_z + 0.10,
])
print(f"PRE-APPROACH: {pre_pos}", flush=True)
j = solve_ik(pre_pos, push_quat)
if j is not None:
    try:
        move_to_joints(j)
    except Exception as e:
        print(f"  pre-approach failed: {e}", flush=True)

# Descend to push height (slightly below handle bar so paddle contacts drawer face below handle)
push_z = handle_z - 0.04
contact_pos = np.array([
    handle_x - push_dir[0] * 0.03,  # 3cm in front of handle
    handle_y - push_dir[1] * 0.03,
    push_z,
])
print(f"CONTACT: {contact_pos}", flush=True)
j = solve_ik(contact_pos, push_quat)
if j is not None:
    try:
        move_to_joints(j)
    except Exception as e:
        print(f"  contact descent failed: {e}", flush=True)

# Push toward cabinet (8 waypoints, total 0.25m)
PUSH_DISTANCE = 0.25
for i in range(1, 9):
    wp = contact_pos + push_dir * (PUSH_DISTANCE * (i / 8.0))
    j = solve_ik(wp, push_quat)
    if j is not None:
        try:
            move_to_joints(j)
        except Exception as e:
            print(f"  push wp {i} failed: {e}", flush=True)
            break

# Final state
obs_end = get_observation()
print(f"\nDone. Final wrist=[{obs_end['robot_cartesian_pos'][0]:.3f},{obs_end['robot_cartesian_pos'][1]:.3f},{obs_end['robot_cartesian_pos'][2]:.3f}]", flush=True)
