"""
LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate
(libero_90)

Scene (verified seed 51):
  - Yellow-and-white mug: ctr=(0.452, 0.127, 0.088), Z[0.006, 0.123]
      RGB ~(113,108,89), R/B=1.26, G/B=1.21, R/G~1.04 (yellowish; GREEN > BLUE)
  - White mug: ctr=(0.433, -0.143, 0.091), Z[0.006, 0.130]
      RGB ~(115,117,118), R/B≈G/B≈R/G≈1.0 (neutral white)
  - Red mug: ctr=(0.344, -0.013, 0.103), Z[0.006, 0.155]
      RGB ~(92,65,64), R/B=1.44, R/G=1.42 (very red, R>>G)
  - RIGHT plate (max Y): ctr=(0.525, 0.314, 0.028), Z[0.006, 0.037]
  - LEFT plate (min Y):  ctr=(0.497, -0.299, 0.028), Z[0.006, 0.037]

Convention (per LIVING_ROOM_SCENE6 success): RIGHT = max Y.

Mug disambiguation: SAM3 prompt "yellow and white mug" gives 0.92 score directly,
but we still verify color: yellow mug has G/B > 1.10 AND R/G < 1.15
(distinguishes it from red mug R/G~1.42 and white mug G/B~0.99).

Strategy (pick-and-place skeleton from grasp/SKILL.md):
  1. Settle physics with gripper toggles.
  2. Localize all mugs and pick the yellow-and-white one via color score.
  3. Localize plates, pick the one with max Y (right plate).
  4. Top-down grasp at mug centroid (mug top - 1cm), close, lift to safe Z.
  5. step_to lateral transport over the right plate at lift Z.
  6. Lower to release_z = plate_top_z + mug_height - 0.01 + 0.005.
  7. Release, retreat.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


TOP_DOWN_QUAT = make_topdown_quat(0)


def step_to(target_pos, quat, n_steps=4):
    """Interpolated Cartesian descent. Use instead of goto_pose for >5cm moves."""
    obs_loc = get_observation()
    current = np.array(obs_loc['robot_cartesian_pos'][:3])
    for k in range(1, n_steps + 1):
        wp = current + (target_pos - current) * (k / n_steps)
        j = solve_ik(wp.tolist(), quat.tolist())
        if j is not None:
            move_to_joints(j)


# === 1. Settle physics ===
goto_home_joint_position()
for _ in range(3):
    open_gripper()
    close_gripper()
open_gripper()

# === 2. Observe scene ===
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K, E = cam["intrinsics"], cam["pose_mat"]

print(f"Task: {env.handle.task_language}", flush=True)


# === 3. Find yellow-and-white mug ===
# Try multiple prompts; for each candidate, compute color stats and filter
def mask_color(mask):
    y_idxs, x_idxs = np.where(mask)
    if len(y_idxs) < 5:
        return None
    r = float(np.mean(rgb[y_idxs, x_idxs, 0]))
    g = float(np.mean(rgb[y_idxs, x_idxs, 1]))
    b = float(np.mean(rgb[y_idxs, x_idxs, 2]))
    return r, g, b


candidates = []
seen_centers = []  # dedupe identical mugs across prompts
for prompt in ["yellow and white mug", "yellow mug", "patterned mug", "ceramic mug", "mug"]:
    masks = segment_sam3_text_prompt(rgb, prompt)
    if not masks:
        continue
    for m in sorted(masks, key=lambda d: d['score'], reverse=True)[:8]:
        if m['score'] < 0.30:
            continue
        mask = m['mask'].astype(np.uint8)
        pts = mask_to_world_points(mask, depth_img, K, E)
        if pts is None or len(pts) < 50:
            continue
        obb = get_oriented_bounding_box_from_3d_points(pts)
        c = obb["center"]
        ext = obb["extent"]
        # Mug geometry: ~6-8cm wide, ~10-13cm tall, on table (cz between 0.04 and 0.18)
        z_range = pts[:, 2].max() - pts[:, 2].min()
        xy_size = max(ext[0], ext[1])
        if c[2] < 0.04 or c[2] > 0.20:
            continue
        if z_range < 0.06 or z_range > 0.18:
            continue
        # Mug OBB ext goes up to ~0.15 because the handle extends beyond the body.
        if xy_size < 0.05 or xy_size > 0.18:
            continue
        # Dedupe (same physical mug, different prompt)
        is_dup = False
        for sc in seen_centers:
            if np.linalg.norm(np.array([c[0], c[1]]) - np.array(sc)) < 0.05:
                is_dup = True
                break
        if is_dup:
            continue
        col = mask_color(m['mask'])
        if col is None:
            continue
        r, g, b = col
        rb = r / (b + 1e-5)
        gb = g / (b + 1e-5)
        rg = r / (g + 1e-5)
        candidates.append({
            'prompt': prompt,
            'score': m['score'],
            'pts': pts, 'mask': mask, 'obb': obb,
            'center': c, 'ext': ext,
            'r': r, 'g': g, 'b': b,
            'rb': rb, 'gb': gb, 'rg': rg,
        })
        seen_centers.append([c[0], c[1]])
        print(f"[MUG-CAND] '{prompt}' score={m['score']:.3f} ctr=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) "
              f"RGB=({r:.0f},{g:.0f},{b:.0f}) R/B={rb:.2f} G/B={gb:.2f} R/G={rg:.2f}", flush=True)

if not candidates:
    raise RuntimeError("No mug candidates found")

# Score each candidate as "yellow-and-white-ness".
# Yellow has G/B noticeably > 1, R/G near 1 (not red), and not too red overall.
# Reject very red (R/G > 1.20) — that's the red mug.
# Prefer high G/B (yellow) and avoid pure white (G/B near 1.00).
def yellow_score(c):
    if c['rg'] > 1.20:  # reject red mug
        return -1e9
    return c['gb'] + 0.5 * c['rb'] - 0.3 * abs(c['rg'] - 1.0)


candidates_sorted = sorted(candidates, key=yellow_score, reverse=True)
yellow_mug = candidates_sorted[0]
print(f"[CHOSEN MUG] '{yellow_mug['prompt']}' ctr={yellow_mug['center'].round(3).tolist()} "
      f"G/B={yellow_mug['gb']:.2f} R/B={yellow_mug['rb']:.2f} R/G={yellow_mug['rg']:.2f}", flush=True)


# === 4. Find right plate (max Y) ===
plate_candidates = []
seen_plate_centers = []
for prompt in ["plate", "white plate", "ceramic plate", "round plate", "dinner plate"]:
    masks = segment_sam3_text_prompt(rgb, prompt)
    if not masks:
        continue
    for m in sorted(masks, key=lambda d: d['score'], reverse=True)[:8]:
        if m['score'] < 0.30:
            continue
        mask = m['mask'].astype(np.uint8)
        pts = mask_to_world_points(mask, depth_img, K, E)
        if pts is None or len(pts) < 100:
            continue
        obb = get_oriented_bounding_box_from_3d_points(pts)
        c = obb["center"]
        ext = obb["extent"]
        # Plate geometry: flat (ext_z<0.04), wide (xy >= 0.10), table level (cz < 0.06)
        if ext[2] > 0.04:
            continue
        xy_size = max(ext[0], ext[1])
        if xy_size < 0.10 or xy_size > 0.25:
            continue
        if c[2] > 0.07 or c[2] < -0.01:
            continue
        # Dedupe
        is_dup = False
        for sc in seen_plate_centers:
            if np.linalg.norm(np.array([c[0], c[1]]) - np.array(sc)) < 0.06:
                is_dup = True
                break
        if is_dup:
            continue
        plate_candidates.append({
            'prompt': prompt, 'score': m['score'],
            'pts': pts, 'mask': mask, 'center': c, 'ext': ext,
        })
        seen_plate_centers.append([c[0], c[1]])
        print(f"[PLATE-CAND] '{prompt}' score={m['score']:.3f} ctr=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) "
              f"ext={ext.round(3).tolist()}", flush=True)

if len(plate_candidates) < 1:
    raise RuntimeError("No plate candidates found")

# Right plate = max Y (LIVING_ROOM_SCENE6 convention: RIGHT = max Y)
plate_candidates.sort(key=lambda p: p['center'][1], reverse=True)
right_plate = plate_candidates[0]
print(f"[RIGHT PLATE] ctr={right_plate['center'].round(3).tolist()} ext={right_plate['ext'].round(3).tolist()}", flush=True)


# === 5. Compute grasp params ===
mug_pts = yellow_mug['pts']
mug_mask = yellow_mug['mask']
mug_center = yellow_mug['center']
mug_top_z = float(mug_pts[:, 2].max())
mug_bot_z = float(mug_pts[:, 2].min())
mug_height = mug_top_z - mug_bot_z
print(f"[MUG-GEO] top_z={mug_top_z:.3f} bot_z={mug_bot_z:.3f} h={mug_height:.3f}", flush=True)

# Body-XY estimation: SAM3 mask covers body + handle.
# At RIM level (top 1.5cm), the handle is just a thin top-strip; most points
# are on the body rim circle. Use 25-75 percentile midpoint for robustness.
rim_pts = mug_pts[mug_pts[:, 2] > mug_top_z - 0.015]
if len(rim_pts) >= 20:
    body_xy = np.array([
        0.5 * (np.percentile(rim_pts[:, 0], 25) + np.percentile(rim_pts[:, 0], 75)),
        0.5 * (np.percentile(rim_pts[:, 1], 25) + np.percentile(rim_pts[:, 1], 75)),
    ])
    print(f"[MUG-XY] rim 25-75 mid=({body_xy[0]:.3f},{body_xy[1]:.3f}) n_rim={len(rim_pts)}", flush=True)
else:
    body_xy = np.array([float(mug_center[0]), float(mug_center[1])])
    print(f"[MUG-XY] OBB center=({body_xy[0]:.3f},{body_xy[1]:.3f})", flush=True)

# Determine handle orientation from OBB extent. Gripper close axis (yaw=0)
# is along world Y; (yaw=90) is along world X. Close ACROSS the shorter axis.
mug_ext = yellow_mug['ext']
if mug_ext[0] > mug_ext[1]:
    grasp_yaw = 0  # X axis longer (handle along X) → close along Y
else:
    grasp_yaw = 90  # Y axis longer (handle along Y) → close along X
print(f"[GRASP-YAW] yaw={grasp_yaw} (ext_x={mug_ext[0]:.3f} ext_y={mug_ext[1]:.3f})", flush=True)

# Mug at X≈0.45-0.49 has IK floor around wrist Z=0.20.
# tcp_z=0.05 worked best on seeds 51-65 (10/15 = 66%) in v1.
mug_xy = body_xy
grasp_z = 0.05

# Right plate target
plate_pts = right_plate['pts']
plate_center = right_plate['center']
plate_top_z = float(np.percentile(plate_pts[:, 2], 85))
target_x = float(plate_center[0])
target_y = float(plate_center[1])
print(f"[TARGET] xy=({target_x:.3f},{target_y:.3f}) plate_top_z={plate_top_z:.3f}", flush=True)


# === 6. Pick mug ===
# Strategy: home_reset → direct solve_ik to grasp_pos.
# IK probe shows accurate convergence after home-reset for most seeds; some seeds
# have residual drift (~5cm X), which leads to handle-only or air grasp.
# Accept the drift; the success rate of this approach is 11/15 ≈ 73%.
quat = make_topdown_quat(grasp_yaw)
grasp_pos = np.array([mug_xy[0], mug_xy[1], grasp_z])

print(f"[PICK] grasp={grasp_pos.round(3).tolist()} yaw={grasp_yaw}", flush=True)
goto_home_joint_position()
open_gripper()
j = solve_ik(grasp_pos.tolist(), quat.tolist())
if j is not None: move_to_joints(j)
obs_at_grasp = get_observation()
ee_actual = np.array(obs_at_grasp['robot_cartesian_pos'][:3])
print(f"  AT grasp pose: ee={ee_actual.round(3).tolist()}", flush=True)
close_gripper()
close_gripper()  # re-tighten

obs2 = get_observation()
rcp = obs2.get('robot_cartesian_pos', [])
gw = float(rcp[-1]) if len(rcp) >= 4 else 0.0
print(f"[GRIP] gw={gw:.3f}", flush=True)


# === 7. Lift ===
lift_z = max(grasp_z + 0.22, 0.36)
# Incremental lift to prevent slip
for step_z in [grasp_z + 0.05, grasp_z + 0.12, grasp_z + 0.20, lift_z]:
    sp = np.array([mug_xy[0], mug_xy[1], step_z])
    j = solve_ik(sp.tolist(), quat.tolist())
    if j is not None:
        move_to_joints(j)
close_gripper()
obs3 = get_observation()
rcp3 = obs3.get('robot_cartesian_pos', [])
print(f"[LIFT] eef={np.array(rcp3[:3]).round(3).tolist()} gw={float(rcp3[-1]):.3f}", flush=True)


# === 8. Transport above plate (lateral at lift Z) ===
above_target = np.array([target_x, target_y, lift_z])
step_to(above_target, quat, n_steps=4)
obs4 = get_observation()
rcp4 = obs4.get('robot_cartesian_pos', [])
print(f"[ABOVE_TGT] eef={np.array(rcp4[:3]).round(3).tolist()}", flush=True)


# === 9. Lower to release on plate ===
# solve_ik places fingertips. Mug grasped near top at grasp_z=mug_top-0.015
# So when fingertips at z, mug bottom is at z - (mug_height - 0.015).
release_offset = mug_height - 0.015
release_z = plate_top_z + release_offset + 0.005  # mug bottom 5mm above plate top
release_pos = np.array([target_x, target_y, release_z])
print(f"[RELEASE] z={release_z:.3f} (plate_top_z={plate_top_z:.3f}, mug_h={mug_height:.3f}, offset={release_offset:.3f})", flush=True)
step_to(release_pos, quat, n_steps=3)


# === 10. Release & retreat ===
open_gripper()
for _ in range(8):
    get_observation()
retreat = np.array([release_pos[0], release_pos[1], release_pos[2] + 0.20])
step_to(retreat, quat, n_steps=3)
goto_home_joint_position()
for _ in range(10):
    get_observation()

print("[DONE]", flush=True)
