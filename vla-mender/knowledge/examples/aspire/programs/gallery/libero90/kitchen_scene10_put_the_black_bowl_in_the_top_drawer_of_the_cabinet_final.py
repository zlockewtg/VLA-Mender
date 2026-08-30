"""
Fix code v4: libero_90 / KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet

Task language: "put the black bowl in the top drawer of the cabinet"

v4 changes vs v3:
- Add grasp retry: if gw < 0.04 after first attempt, try alternative grasp candidates.
- Use goto_pose for grasp approach (z_approach=0.15) — internal Cartesian path is more reliable.
- Try both rim grasps (multiple candidates) and forced near-side rim fallback.
"""
import numpy as np
from scipy.spatial.transform import Rotation


TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
TRANSIT_Z = 0.40
TABLE_Z = -0.011

DRAWER_X_FALLBACK = 0.66
DRAWER_Y_FALLBACK = -0.16
DRAWER_FLOOR_Z_FALLBACK = 0.155
DRAWER_TOP_Z_FALLBACK = 0.213


def localize_bowl(rgb, depth, K, E):
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    prompts = ["small bowl", "metal bowl", "silver bowl", "akita black bowl",
               "black bowl", "bowl"]
    best_score = -1.0
    best = None
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in sorted(masks, key=lambda x: -x['score'])[:5]:
            if m['score'] < 0.20:
                continue
            mask_arr = m['mask'].astype(np.uint8)
            npix = int(mask_arr.sum())
            if npix < 50 or npix > 12000:
                continue
            pts = mask_to_world_points(mask_arr, depth, K, E)
            if pts is None or len(pts) < 30:
                continue
            cx = float(np.median(pts[:, 0]))
            cy = float(np.median(pts[:, 1]))
            cz = float(np.median(pts[:, 2]))
            if not (0.40 <= cx <= 0.75):
                continue
            if not (-0.05 <= cy <= 0.30):
                continue
            if not (-0.02 <= cz <= 0.10):
                continue
            score = m['score']
            if score > best_score:
                best_score = score
                d_xy = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
                near_pts = pts[d_xy < 0.08]
                top_z = float(np.percentile(near_pts[:, 2], 95)) if len(near_pts) > 5 else float(np.percentile(pts[:, 2], 95))
                base_z = float(np.percentile(near_pts[:, 2], 10)) if len(near_pts) > 5 else float(np.percentile(pts[:, 2], 10))
                best = {
                    'cx': cx, 'cy': cy, 'cz': cz,
                    'pts': pts, 'mask': mask_arr,
                    'top_z': top_z, 'base_z': base_z,
                    'score': score, 'prompt': prompt,
                }
        if best is not None and best_score > 0.85:
            break
    return best


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
            if not (0.55 <= cx <= 0.80):
                continue
            if not (-0.27 <= cy <= -0.05):
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


def get_rim_grasp_candidates(grasps, scores, E, bowl_cx, bowl_cy):
    """Return list of (score, xy_pos) for near-side rim grasps, sorted by score."""
    candidates = []
    for i in range(len(grasps)):
        g_world = E @ grasps[i]
        gp = g_world[:3, 3]
        y_off = gp[1] - bowl_cy
        x_off = gp[0] - bowl_cx
        # Near-side rim: Y_off in [0.025, 0.07], |X_off| < 0.07
        if 0.025 <= y_off <= 0.07 and abs(x_off) < 0.07:
            candidates.append((float(scores[i]), gp))
    candidates.sort(key=lambda x: -x[0])
    return candidates


def attempt_grasp(grasp_xy, top_z, label):
    """Attempt to grasp at given XY. Returns gw after lift."""
    open_gripper()

    # Sequential descent via solve_ik+move_to_joints
    grasp_z_target = max(top_z - 0.025, 0.005)  # rim grasp Z (TCP target)
    descent_seq = [0.20, 0.10, 0.05, grasp_z_target]
    for tz in descent_seq:
        pos = np.array([grasp_xy[0], grasp_xy[1], tz])
        j = solve_ik(pos, TOP_DOWN_QUAT)
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
    gw = float(obs_g['robot_cartesian_pos'][-1])
    print(f"  [{label}] After close: gw={gw:.3f}", flush=True)

    # Lift
    lift_pos = np.array([grasp_xy[0], grasp_xy[1], TRANSIT_Z])
    j = solve_ik(lift_pos, TOP_DOWN_QUAT)
    if j is not None:
        try:
            move_to_joints(j)
        except Exception as e:
            print(f"  [{label}] lift failed: {e}", flush=True)

    obs_l = get_observation()
    gw_l = float(obs_l['robot_cartesian_pos'][-1])
    wrist_l = obs_l['robot_cartesian_pos'][:3]
    print(f"  [{label}] After lift: gw={gw_l:.3f} wrist=[{wrist_l[0]:.3f},{wrist_l[1]:.3f},{wrist_l[2]:.3f}]", flush=True)
    return gw_l


print(f"Task: {env.handle.task_language}", flush=True)

# ===== Phase 0: Initial observation =====
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if depth.ndim == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]

bowl = localize_bowl(rgb, depth_img, K, E)
if bowl is None:
    raise RuntimeError("BOWL: not detected")
print(f"BOWL: c=[{bowl['cx']:.3f},{bowl['cy']:.3f},{bowl['cz']:.3f}] base_z={bowl['base_z']:.3f} top_z={bowl['top_z']:.3f} score={bowl['score']:.3f}", flush=True)

drawer = localize_drawer_interior(rgb, depth_img, K, E)
if drawer is None:
    drawer_x = DRAWER_X_FALLBACK
    drawer_y = DRAWER_Y_FALLBACK
    drawer_floor_z = DRAWER_FLOOR_Z_FALLBACK
    drawer_top_z = DRAWER_TOP_Z_FALLBACK
    print(f"DRAWER: not detected, fallback x={drawer_x:.3f} y={drawer_y:.3f}", flush=True)
else:
    drawer_x = (drawer['x_range'][0] + drawer['x_range'][1]) / 2.0
    drawer_y = (drawer['y_range'][0] + drawer['y_range'][1]) / 2.0
    drawer_floor_z = drawer['floor_z']
    drawer_top_z = drawer['top_z']
    print(f"DRAWER: c=[{drawer['cx']:.3f},{drawer['cy']:.3f},{drawer['cz']:.3f}] x={drawer['x_range']} y={drawer['y_range']} floor_z={drawer_floor_z:.3f} top_z={drawer_top_z:.3f}", flush=True)

print(f"PLACE TARGET: x={drawer_x:.3f} y={drawer_y:.3f} drawer_top_z={drawer_top_z:.3f}", flush=True)

# ===== Phase 1: Compute grasp candidates =====
grasps, scores = plan_grasp(depth, K, bowl['mask'])
print(f"GraspNet: {len(grasps)} grasps", flush=True)

candidates = []
if len(grasps) > 0:
    candidates = get_rim_grasp_candidates(grasps, scores, E, bowl['cx'], bowl['cy'])
    print(f"Rim-side candidates: {len(candidates)}", flush=True)

# Always include hardcoded rim fallback
forced_rim_xy = (bowl['cx'], bowl['cy'] + 0.045)
forced_rim_xy_b = (bowl['cx'] + 0.012, bowl['cy'] + 0.048)  # alt: slightly +X

# ===== Phase 2: Pick (try multiple candidates) =====
goto_home_joint_position()

GW_SUCCESS = 0.040
gw_final = 0.0
chosen_xy = None

# Try up to 3 candidates
attempts = []
for sc, gp in candidates[:2]:
    attempts.append((float(gp[0]), float(gp[1]), f"rim{sc:.2f}"))
attempts.append((forced_rim_xy[0], forced_rim_xy[1], "forced_rim"))
attempts.append((forced_rim_xy_b[0], forced_rim_xy_b[1], "forced_rim_b"))

for i, (gx, gy, label) in enumerate(attempts):
    print(f"\n=== Attempt {i+1} [{label}]: XY=[{gx:.3f},{gy:.3f}] ===", flush=True)
    gw = attempt_grasp((gx, gy), bowl['top_z'], label)
    if gw >= GW_SUCCESS:
        gw_final = gw
        chosen_xy = (gx, gy)
        print(f"GRASP SUCCESS at attempt {i+1}: gw={gw:.3f}", flush=True)
        break
    else:
        # Failed, return to home for next attempt
        if i < len(attempts) - 1:
            try:
                goto_home_joint_position()
                open_gripper()
            except Exception:
                pass

if chosen_xy is None:
    raise RuntimeError("Failed to grasp bowl after all attempts")

# ===== Phase 3: Transport above drawer =====
above_drawer = np.array([drawer_x, drawer_y, TRANSIT_Z])
j = solve_ik(above_drawer, TOP_DOWN_QUAT)
if j is not None:
    move_to_joints(j)
obs_t = get_observation()
gw_t = float(obs_t['robot_cartesian_pos'][-1])
wrist_t = obs_t['robot_cartesian_pos'][:3]
print(f"Above drawer: gw={gw_t:.3f} wrist=[{wrist_t[0]:.3f},{wrist_t[1]:.3f},{wrist_t[2]:.3f}]", flush=True)

# ===== Phase 4: Place inside drawer =====
descent_tcp = [0.30, 0.25, 0.22, 0.20, 0.18]
for tz in descent_tcp:
    pos = np.array([drawer_x, drawer_y, tz])
    j = solve_ik(pos, TOP_DOWN_QUAT)
    if j is not None:
        try:
            move_to_joints(j)
        except Exception as e:
            print(f"  descent tz={tz:.3f}: failed {e}", flush=True)
            break

obs_pp = get_observation()
wrist_pp = obs_pp['robot_cartesian_pos'][:3]
gw_pp = float(obs_pp['robot_cartesian_pos'][-1])
print(f"At place: wrist=[{wrist_pp[0]:.3f},{wrist_pp[1]:.3f},{wrist_pp[2]:.3f}] gw={gw_pp:.3f}", flush=True)

# Release
open_gripper()
print("Released.", flush=True)

# Settle
for _ in range(8):
    get_observation()

# Lift away
lift_away = np.array([drawer_x, drawer_y, TRANSIT_Z])
try:
    j = solve_ik(lift_away, TOP_DOWN_QUAT)
    if j is not None:
        move_to_joints(j)
except Exception as e:
    print(f"Lift away: {e}", flush=True)

obs_end = get_observation()
print(f"Done. Final wrist=[{obs_end['robot_cartesian_pos'][0]:.3f},{obs_end['robot_cartesian_pos'][1]:.3f},{obs_end['robot_cartesian_pos'][2]:.3f}]", flush=True)
