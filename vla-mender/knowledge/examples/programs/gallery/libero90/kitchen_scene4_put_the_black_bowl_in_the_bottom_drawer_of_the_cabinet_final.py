"""
Fix code: libero_90 / KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet

Task language: "put the black bowl in the bottom drawer of the cabinet"

Adapted from KITCHEN_SCENE10 (top drawer) by flipping the cabinet side from -Y to +Y
and using a much lower drawer Z range (cavity Z ~ [0.013, 0.113] vs top drawer
~ [0.155, 0.213]).

Init state (verified on seeds 51, 52):
  - Bottom drawer is OPEN (extending in +Y).
  - "open drawer" SAM3 mask: X=[0.56, 0.78], Y=[0.04, 0.23], Z=[-0.011, 0.113].
  - Bowl: "metal bowl" / "small bowl" >0.9 score, cx~0.66-0.68, cy~-0.05, cz~0.02.

Strategy:
  1. Localize bowl (small/metal/silver/black/bowl prompts).
  2. Localize bottom-drawer interior (X≈0.56-0.78, Y≈0.04-0.23, top Z≈0.10-0.12).
  3. Grasp bowl rim from near (-Y) side using GraspNet candidates.
  4. Lift, transit above drawer cavity center.
  5. Descend to wrist Z just below drawer top edge (~0.18-0.20) then release.
  6. Lift away.
"""
import numpy as np
from scipy.spatial.transform import Rotation


TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
TRANSIT_Z = 0.40
TABLE_Z = -0.011


def make_topdown_quat(yaw_deg=0.0):
    """Top-down orientation with optional yaw rotation about world Z."""
    R = (Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix()
         @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz

# Fallbacks (used if SAM3 misses; values from seed 51 exploration).
DRAWER_X_FALLBACK = 0.68
DRAWER_Y_FALLBACK = 0.13
DRAWER_TOP_Z_FALLBACK = 0.113
DRAWER_FLOOR_Z_FALLBACK = 0.025


def localize_bowl(rgb, depth, K, E):
    """Find the bowl on the kitchen counter (Y < 0.0)."""
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
            # Bowl is on the counter (Y<0) and on the table (Z low).
            if not (0.45 <= cx <= 0.85):
                continue
            if not (-0.30 <= cy <= 0.05):
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
    """Find the bottom drawer interior cavity (cabinet at +Y side).

    Heuristic: a wide blob with X∈[0.55,0.80], Y∈[0.04,0.25], Z up to ~0.11.
    """
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    prompts = ["open drawer", "drawer interior", "drawer opening",
               "bottom drawer", "open bottom drawer"]
    best = None
    best_extent = -1.0
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in sorted(masks, key=lambda x: -x['score'])[:6]:
            if m['score'] < 0.10:
                continue
            mask_arr = m['mask'].astype(np.uint8)
            pts = mask_to_world_points(mask_arr, depth, K, E)
            if pts is None or len(pts) < 100:
                continue
            cx = float(np.median(pts[:, 0]))
            cy = float(np.median(pts[:, 1]))
            cz = float(np.median(pts[:, 2]))
            # Cabinet at +Y. Bottom drawer is wide and low.
            if not (0.50 <= cx <= 0.85):
                continue
            if not (0.03 <= cy <= 0.30):
                continue
            x_min = float(pts[:, 0].min())
            x_max = float(pts[:, 0].max())
            y_min = float(pts[:, 1].min())
            y_max = float(pts[:, 1].max())
            if (x_max - x_min) < 0.08 or (x_max - x_min) > 0.35:
                continue
            if (y_max - y_min) < 0.08 or (y_max - y_min) > 0.30:
                continue
            top_z = float(pts[:, 2].max())
            # Bottom drawer top edge is around 0.10-0.13. Reject very tall blobs.
            if top_z < 0.05 or top_z > 0.20:
                continue
            # Floor estimate from low percentile.
            z20 = float(np.percentile(pts[:, 2], 20))
            floor_pts = pts[pts[:, 2] <= z20 + 0.005]
            floor_z = float(floor_pts[:, 2].mean()) if len(floor_pts) > 0 else float(pts[:, 2].min())
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


def get_rim_grasp_candidates(grasps, scores, E, bowl_cx, bowl_cy, bowl_radius):
    """Return list of (score, xy_pos) for rim grasps, sorted by score.

    Rim radius = bowl OBB radius. Filter grasps whose distance from bowl center
    is close to bowl_radius. We use a tight window so we don't get grasps that
    miss the rim entirely.
    """
    candidates = []
    for i in range(len(grasps)):
        g_world = E @ grasps[i]
        gp = g_world[:3, 3]
        d = float(np.hypot(gp[0] - bowl_cx, gp[1] - bowl_cy))
        # Tight window around the rim radius.
        if (bowl_radius - 0.020) <= d <= (bowl_radius + 0.010):
            candidates.append((float(scores[i]), gp))
    candidates.sort(key=lambda x: -x[0])
    return candidates


def attempt_grasp(grasp_xy, top_z, label, yaw_deg=0.0):
    """Attempt to grasp at given XY with optional gripper yaw. Returns (gw, achieved_wrist_z)."""
    open_gripper()
    quat = make_topdown_quat(yaw_deg) if yaw_deg != 0.0 else TOP_DOWN_QUAT
    # solve_ik takes the GRIPPER TIP target (TCP offset is applied internally).
    # Target gripper tip BELOW the bowl rim so the jaws straddle the rim wall.
    grasp_z_target = max(top_z - 0.030, 0.005)
    descent_seq = [0.15, 0.10, 0.05, 0.02, grasp_z_target]
    for tz in descent_seq:
        pos = np.array([grasp_xy[0], grasp_xy[1], tz])
        j = solve_ik(pos, quat)
        if j is not None:
            try:
                move_to_joints(j)
            except Exception as e:
                print(f"  [{label}] descent tz={tz:.3f} failed: {e}", flush=True)
                break

    obs_d = get_observation()
    wrist_d = obs_d['robot_cartesian_pos'][:3]
    print(f"  [{label}] yaw={yaw_deg:+.0f} At grasp: wrist=[{wrist_d[0]:.3f},{wrist_d[1]:.3f},{wrist_d[2]:.3f}]", flush=True)

    close_gripper()
    obs_g = get_observation()
    gw = float(obs_g['robot_cartesian_pos'][-1])
    print(f"  [{label}] After close: gw={gw:.3f}", flush=True)

    lift_pos = np.array([grasp_xy[0], grasp_xy[1], TRANSIT_Z])
    j = solve_ik(lift_pos, quat)
    if j is not None:
        try:
            move_to_joints(j)
        except Exception as e:
            print(f"  [{label}] lift failed: {e}", flush=True)

    obs_l = get_observation()
    gw_l = float(obs_l['robot_cartesian_pos'][-1])
    wrist_l = obs_l['robot_cartesian_pos'][:3]
    print(f"  [{label}] After lift: gw={gw_l:.3f} wrist=[{wrist_l[0]:.3f},{wrist_l[1]:.3f},{wrist_l[2]:.3f}]", flush=True)
    return gw_l, wrist_d[2]


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
# Bowl OBB radius (gives the rim's distance from center).
try:
    obb = get_oriented_bounding_box_from_3d_points(bowl['pts'])
    bowl_radius = float(max(obb['extent'][:2]) / 2.0)
except Exception:
    bowl_radius = 0.052  # default ~5cm rim
bowl['radius'] = bowl_radius
print(f"BOWL: c=[{bowl['cx']:.3f},{bowl['cy']:.3f},{bowl['cz']:.3f}] base_z={bowl['base_z']:.3f} top_z={bowl['top_z']:.3f} score={bowl['score']:.3f} prompt={bowl['prompt']!r} radius={bowl_radius:.3f}", flush=True)

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
    print(f"DRAWER: c=[{drawer['cx']:.3f},{drawer['cy']:.3f},{drawer['cz']:.3f}] x={drawer['x_range']} y={drawer['y_range']} floor_z={drawer_floor_z:.3f} top_z={drawer_top_z:.3f} prompt={drawer['prompt']!r}", flush=True)

print(f"PLACE TARGET: x={drawer_x:.3f} y={drawer_y:.3f} drawer_top_z={drawer_top_z:.3f}", flush=True)

# ===== Phase 1: Compute grasp candidates =====
grasps, scores = plan_grasp(depth, K, bowl['mask'])
print(f"GraspNet: {len(grasps)} grasps", flush=True)

candidates = []
if len(grasps) > 0:
    candidates = get_rim_grasp_candidates(grasps, scores, E, bowl['cx'], bowl['cy'], bowl_radius)
    print(f"Rim-side candidates: {len(candidates)}", flush=True)
    for sc, gp in candidates[:5]:
        d = float(np.hypot(gp[0]-bowl['cx'], gp[1]-bowl['cy']))
        print(f"  cand: score={sc:.3f} pos=[{gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}] d_from_center={d:.3f}", flush=True)

# Forced rim fallbacks at the bowl's true rim radius (for x and y axes).
r = bowl_radius * 0.95  # 95% so gripper jaws straddle the rim wall
forced_rim_xy = (bowl['cx'], bowl['cy'] + r)         # +Y rim (cabinet side)
forced_rim_xy_b = (bowl['cx'], bowl['cy'] - r)       # -Y rim (robot side)
forced_rim_xy_c = (bowl['cx'] + r, bowl['cy'])       # +X rim
forced_rim_xy_d = (bowl['cx'] - r, bowl['cy'])       # -X rim

# ===== Phase 2: Pick (try multiple candidates) =====
goto_home_joint_position()

GW_SUCCESS = 0.040
gw_final = 0.0
chosen_xy = None

# Lead with forced rim positions (guaranteed at rim radius) to minimize bowl-push
# events that come from misaligned GraspNet candidates. Then add 1-2 high-score
# GraspNet rim picks as a backup.
attempts = []
attempts.append((forced_rim_xy[0], forced_rim_xy[1], "forced_rim_+Y"))
attempts.append((forced_rim_xy_b[0], forced_rim_xy_b[1], "forced_rim_-Y"))
hi_score_cands = [(sc, gp) for (sc, gp) in candidates if sc > 0.20]
for sc, gp in hi_score_cands[:2]:
    attempts.append((float(gp[0]), float(gp[1]), f"rim{sc:.2f}"))
attempts.append((forced_rim_xy_c[0], forced_rim_xy_c[1], "forced_rim_+X"))
attempts.append((forced_rim_xy_d[0], forced_rim_xy_d[1], "forced_rim_-X"))

def reset_and_relocalize(prev_bowl, attempts_list, current_idx):
    """Reset robot and re-localize bowl. Update attempts_list in-place if bowl moved."""
    try:
        goto_home_joint_position()
        open_gripper()
        obs2 = get_observation()
        rgb2 = obs2["agentview"]["images"]["rgb"]
        d2 = obs2["agentview"]["images"]["depth"]
        d2 = d2[:, :, 0] if d2.ndim == 3 else d2
        K2 = obs2["agentview"]["intrinsics"]
        E2 = obs2["agentview"]["pose_mat"]
        bowl2 = localize_bowl(rgb2, d2, K2, E2)
        if bowl2 is not None:
            dx = bowl2['cx'] - prev_bowl['cx']
            dy = bowl2['cy'] - prev_bowl['cy']
            if abs(dx) > 0.005 or abs(dy) > 0.005:
                print(f"  Bowl moved: dx={dx:.3f} dy={dy:.3f} → shifting future attempts", flush=True)
                for k in range(current_idx + 1, len(attempts_list)):
                    ox, oy, ol = attempts_list[k]
                    attempts_list[k] = (ox + dx, oy + dy, ol)
                prev_bowl['cx'] = bowl2['cx']
                prev_bowl['cy'] = bowl2['cy']
                prev_bowl['top_z'] = bowl2['top_z']
    except Exception as e:
        print(f"  Reset/re-localize failed: {e}", flush=True)


for i, (gx, gy, label) in enumerate(attempts):
    print(f"\n=== Attempt {i+1} [{label}]: XY=[{gx:.3f},{gy:.3f}] ===", flush=True)
    # Try yaw=0 first; if descent doesn't reach low enough or grasp fails, try yaws.
    yaw_seq = [0.0, 45.0, 90.0, -45.0]
    success_yaw = None
    for yaw in yaw_seq:
        gw, achieved_wrist_z = attempt_grasp((gx, gy), bowl['top_z'], label, yaw_deg=yaw)
        if gw >= GW_SUCCESS:
            gw_final = gw
            chosen_xy = (gx, gy)
            success_yaw = yaw
            print(f"GRASP SUCCESS at attempt {i+1} yaw={yaw:+.0f}: gw={gw:.3f}", flush=True)
            break
        # If wrist couldn't descend low enough, try another yaw without re-localize.
        if achieved_wrist_z > 0.18 and yaw != yaw_seq[-1]:
            print(f"  wrist Z={achieved_wrist_z:.3f} too high — try next yaw", flush=True)
            try:
                goto_home_joint_position()
                open_gripper()
            except Exception:
                pass
            continue
        # Otherwise, descent reached low enough but grasp failed — break to re-localize.
        break

    if chosen_xy is not None:
        break

    if i < len(attempts) - 1:
        reset_and_relocalize(bowl, attempts, i)

if chosen_xy is None:
    raise RuntimeError("Failed to grasp bowl after all attempts")

# ===== Phase 2.5: Compute bowl-vs-gripper offset =====
# Gripper grasped at chosen_xy; the bowl center is at bowl['cx'],bowl['cy'].
# When we transport, the gripper xy = goal_xy, and the bowl xy = goal_xy + offset.
offset_x = bowl['cx'] - chosen_xy[0]
offset_y = bowl['cy'] - chosen_xy[1]
print(f"\nBowl offset from gripper: dx={offset_x:.3f} dy={offset_y:.3f}", flush=True)

# ===== Phase 3: Transport above drawer =====
# Compensate for bowl-vs-gripper offset so bowl lands at drawer_x, drawer_y.
gripper_target_x = drawer_x - offset_x
gripper_target_y = drawer_y - offset_y
print(f"Gripper target (compensated): x={gripper_target_x:.3f} y={gripper_target_y:.3f}", flush=True)

above_drawer = np.array([gripper_target_x, gripper_target_y, TRANSIT_Z])
j = solve_ik(above_drawer, TOP_DOWN_QUAT)
if j is not None:
    move_to_joints(j)
obs_t = get_observation()
gw_t = float(obs_t['robot_cartesian_pos'][-1])
wrist_t = obs_t['robot_cartesian_pos'][:3]
print(f"Above drawer: gw={gw_t:.3f} wrist=[{wrist_t[0]:.3f},{wrist_t[1]:.3f},{wrist_t[2]:.3f}]", flush=True)

# ===== Phase 4: Place inside drawer =====
# Bottom-drawer cavity: top edge ~0.11. Cabinet body causes collision so
# wrist often stalls at z~0.30 (gripper tip ~0.20, bowl ~0.16). Push down hard.
descent_tcp = [0.32, 0.28, 0.24, 0.20, 0.16, 0.12]
for tz in descent_tcp:
    pos = np.array([gripper_target_x, gripper_target_y, tz])
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
for _ in range(15):
    get_observation()

# Lift away — gentle vertical lift first so we don't drag the bowl out.
lift_away = np.array([gripper_target_x, gripper_target_y, TRANSIT_Z])
try:
    j = solve_ik(lift_away, TOP_DOWN_QUAT)
    if j is not None:
        move_to_joints(j)
except Exception as e:
    print(f"Lift away: {e}", flush=True)

obs_end = get_observation()
print(f"Done. Final wrist=[{obs_end['robot_cartesian_pos'][0]:.3f},{obs_end['robot_cartesian_pos'][1]:.3f},{obs_end['robot_cartesian_pos'][2]:.3f}]", flush=True)
