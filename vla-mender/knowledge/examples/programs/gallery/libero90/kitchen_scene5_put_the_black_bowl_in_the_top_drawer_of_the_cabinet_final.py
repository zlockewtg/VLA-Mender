"""
libero_90 / KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet

Task language: "put the black bowl in the top drawer of the cabinet"

Scene state on seed 51 (verified by exploration):
- Bowl at world (0.672, -0.033, 0.016), wide (~11cm OBB), renders silver/metal
  - Best SAM3: "metal bowl" 0.906, "small bowl" 0.859, "black bowl" only 0.5
- Top drawer ALREADY OPEN at init: ctr=(0.659, 0.158, 0.172) Z_range=[-0.011, 0.212]
  - drawer_floor_z ≈ 0.155, drawer_top_z ≈ 0.212
- No need to close drawer (task language doesn't mention it).

Approach (adapted from KS10 task_code.py which got 14/15 on KS10):
1. Localize bowl: SAM3 "small bowl"/"metal bowl"/"silver bowl"
2. Localize drawer interior: SAM3 "drawer interior"/"open drawer", filter Z>0.10
3. Wide-bowl rim grasp via GraspNet near-side filter (Y_off in [0.025, 0.07])
4. Forced rim fallbacks if GraspNet candidates fail
5. Transport at z=0.40 to above drawer interior
6. Descend with TCP z=[0.30, 0.25, 0.22, 0.20, 0.18]
7. Open gripper, retreat
"""
import numpy as np


TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
TRANSIT_Z = 0.40

# Fallbacks (verified from seed 51)
DRAWER_X_FALLBACK = 0.66
DRAWER_Y_FALLBACK = 0.158
DRAWER_FLOOR_Z_FALLBACK = 0.155
DRAWER_TOP_Z_FALLBACK = 0.212


def localize_bowl(rgb, depth, K, E):
    """KS5 bowl filter: bowl on -y side (around y=-0.03)."""
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
            if not (0.40 <= cx <= 0.78):
                continue
            # KS5: bowl at y=-0.03 (front of table)
            if not (-0.25 <= cy <= 0.10):
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
    """KS5 drawer is on +y side at world y≈0.158."""
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    prompts = ["drawer interior", "open drawer", "drawer opening", "inside the drawer",
               "white wooden drawer", "top drawer"]
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
            # KS5: drawer at y ≈ +0.158 (back of table)
            if not (0.55 <= cx <= 0.80):
                continue
            if not (0.05 <= cy <= 0.30):
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
    """Return list of (score, xy_pos) for near-side rim grasps, sorted by score.
    Near-side: relative to camera. The drawer is in +y direction from bowl, so we want
    grasp on the -y side of bowl (rim closest to robot at front of table)."""
    candidates_neg_y = []  # near robot side (rim with y_off < 0)
    candidates_pos_y = []  # cabinet side (rim with y_off > 0)
    for i in range(len(grasps)):
        g_world = E @ grasps[i]
        gp = g_world[:3, 3]
        y_off = gp[1] - bowl_cy
        x_off = gp[0] - bowl_cx
        if abs(x_off) >= 0.07:
            continue
        if -0.07 <= y_off <= -0.025:
            candidates_neg_y.append((float(scores[i]), gp))
        elif 0.025 <= y_off <= 0.07:
            candidates_pos_y.append((float(scores[i]), gp))
    candidates_neg_y.sort(key=lambda x: -x[0])
    candidates_pos_y.sort(key=lambda x: -x[0])
    # Prefer near-robot side (y_off<0) but include both
    return candidates_neg_y, candidates_pos_y


def attempt_grasp(grasp_xy, top_z, label):
    """Attempt to grasp at given XY. Returns gw after lift."""
    open_gripper()

    grasp_z_target = max(top_z - 0.025, 0.005)
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


# ===== Main =====
try:
    print(f"Task: {env.handle.task_language}", flush=True)
except Exception:
    pass

# Settle physics
for _ in range(3):
    open_gripper(); close_gripper()
open_gripper()
for _ in range(2):
    get_observation()

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
print(f"BOWL: c=[{bowl['cx']:.3f},{bowl['cy']:.3f},{bowl['cz']:.3f}] base_z={bowl['base_z']:.3f} top_z={bowl['top_z']:.3f} score={bowl['score']:.3f} prompt={bowl['prompt']}", flush=True)

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
    print(f"DRAWER: c=[{drawer['cx']:.3f},{drawer['cy']:.3f},{drawer['cz']:.3f}] x={drawer['x_range']} y={drawer['y_range']} floor_z={drawer_floor_z:.3f} top_z={drawer_top_z:.3f} prompt={drawer['prompt']}", flush=True)

print(f"PLACE TARGET: x={drawer_x:.3f} y={drawer_y:.3f} drawer_top_z={drawer_top_z:.3f}", flush=True)

# ===== Phase 1: Compute grasp candidates =====
grasps, scores = plan_grasp(depth, K, bowl['mask'])
print(f"GraspNet: {len(grasps)} grasps", flush=True)

cand_neg, cand_pos = [], []
if len(grasps) > 0:
    cand_neg, cand_pos = get_rim_grasp_candidates(grasps, scores, E, bowl['cx'], bowl['cy'])
    print(f"Rim-side cand: -y_side={len(cand_neg)} +y_side={len(cand_pos)}", flush=True)

# Forced rim fallbacks: try -y rim (near robot) first since pulling the bowl from the
# near rim tilts the bowl up toward the cabinet on placement.
forced_rim_neg = (bowl['cx'], bowl['cy'] - 0.045)
forced_rim_pos = (bowl['cx'], bowl['cy'] + 0.045)
forced_rim_neg_b = (bowl['cx'] + 0.012, bowl['cy'] - 0.048)
forced_rim_pos_b = (bowl['cx'] + 0.012, bowl['cy'] + 0.048)

# ===== Phase 2: Pick (try multiple candidates) =====
goto_home_joint_position()

GW_SUCCESS = 0.040

# Try near-robot rim first, then far rim
attempts = []
for sc, gp in cand_neg[:2]:
    attempts.append((float(gp[0]), float(gp[1]), f"rim_neg{sc:.2f}"))
for sc, gp in cand_pos[:2]:
    attempts.append((float(gp[0]), float(gp[1]), f"rim_pos{sc:.2f}"))
attempts.append((forced_rim_neg[0], forced_rim_neg[1], "forced_neg"))
attempts.append((forced_rim_pos[0], forced_rim_pos[1], "forced_pos"))
attempts.append((forced_rim_neg_b[0], forced_rim_neg_b[1], "forced_neg_b"))
attempts.append((forced_rim_pos_b[0], forced_rim_pos_b[1], "forced_pos_b"))

chosen_xy = None
gw_final = 0.0

for i, (gx, gy, label) in enumerate(attempts):
    print(f"\n=== Attempt {i+1} [{label}]: XY=[{gx:.3f},{gy:.3f}] ===", flush=True)
    gw = attempt_grasp((gx, gy), bowl['top_z'], label)
    if gw >= GW_SUCCESS:
        gw_final = gw
        chosen_xy = (gx, gy)
        print(f"GRASP SUCCESS at attempt {i+1}: gw={gw:.3f}", flush=True)
        break
    else:
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
