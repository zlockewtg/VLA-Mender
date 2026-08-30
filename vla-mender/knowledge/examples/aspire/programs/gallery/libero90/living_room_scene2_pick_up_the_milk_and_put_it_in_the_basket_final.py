"""
LIVING_ROOM_SCENE2: pick up the milk and put it in the basket.

Scene: Cluttered table with orange juice, tomato sauce condiment bottle, alphabet soup
       can, tomato sauce can, milk carton (UPRIGHT, ~15cm tall, "Milk" red label on
       white), cream cheese box (small, front-left, whale label, lying flat), small
       brown/red carton (front-center, lying flat), and basket on the right.

Critical: in the initial agentview the robot ARM occludes the milk carton (it's
behind the arm in the field of view). We MUST move the arm out of view BEFORE
observing — otherwise SAM3 only sees ~4 of the 7+ objects.

Strategy:
- Move arm to a high left side-position to clear the camera view.
- SAM3 "carton of milk": the top-scoring detection (~0.88) is the actual upright
  milk carton at center-back.
- Filter to UPRIGHT cartons (h > 0.10) — rejects small lying-flat cartons.
- Among uprights, pick the most-reddish (milk has red Milk label, R/B ~1.7-2.5).
  The orange juice carton is yellow/orange (R/B ~2.0+) but is at the FAR side
  (smaller x or larger negative y), and we apply a position prior preferring the
  central candidate. Confirmed prompt-driven match: "carton of milk" top score
  is the milk.
- Milk is upright ~15cm tall. Grasp top-down at top - 0.04 = upper-mid body.
- Basket localization: SAM3 "wicker basket"; p10/p90 midpoint for true center.
- Drop with multi-step descent.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


# --- Init: open gripper, move arm OUT of agentview ---
goto_home_joint_position()
for _ in range(2):
    open_gripper()
    close_gripper()
open_gripper()

# Move to high-left position so the arm doesn't block the view.
quat = make_topdown_quat(0)
j = solve_ik([0.30, -0.45, 0.55], quat.tolist())
if j is not None:
    move_to_joints(j)

# --- Observe ---
obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth


# --- Find milk carton (UPRIGHT) ---
candidates = []
seen_centers = []

# Use "carton of milk" — gives milk highest score; also gather "milk" / "milk carton"
for prompt in ["carton of milk", "milk carton", "milk"]:
    masks = segment_sam3_text_prompt(rgb, prompt)
    if not masks:
        continue
    for m in sorted(masks, key=lambda d: d['score'], reverse=True)[:5]:
        if m['score'] < 0.40:
            continue
        pts = mask_to_world_points(m['mask'].astype(np.uint8), depth_img, K, E)
        if pts is None or len(pts) < 30:
            continue
        z_min = float(pts[:, 2].min())
        z_max = float(pts[:, 2].max())
        h_range = z_max - z_min
        # Reject wall pictures (z>0.25), floor noise (z<-0.02)
        if z_max > 0.25 or z_min < -0.02:
            continue
        # Milk is UPRIGHT: h > 0.10 (rejects small lying-flat cartons)
        # but h < 0.22 (rejects scene clutter)
        if h_range < 0.10 or h_range > 0.22:
            continue
        try:
            obb = get_oriented_bounding_box_from_3d_points(pts)
        except Exception:
            continue
        cx, cy, cz = obb['center']
        ext = obb['extent']
        max_xy_ext = max(float(ext[0]), float(ext[1]))
        min_xy_ext = min(float(ext[0]), float(ext[1]))
        # Milk carton: ~6-8cm × 4-6cm footprint
        if min_xy_ext < 0.03 or min_xy_ext > 0.10:
            continue
        if max_xy_ext < 0.05 or max_xy_ext > 0.16:
            continue
        # Skip duplicates
        is_dup = False
        for sc in seen_centers:
            if np.linalg.norm(np.array([cx, cy]) - np.array(sc[:2])) < 0.05:
                is_dup = True
                break
        if is_dup:
            continue
        y_idxs, x_idxs = np.where(m['mask'])
        if len(y_idxs) < 5:
            continue
        r = float(np.mean(rgb[y_idxs, x_idxs, 0]))
        g = float(np.mean(rgb[y_idxs, x_idxs, 1]))
        b = float(np.mean(rgb[y_idxs, x_idxs, 2]))
        rb_ratio = r / (b + 1e-5)
        rg_ratio = r / (g + 1e-5)
        candidates.append({
            'mask': m['mask'],
            'pts': pts,
            'obb': obb,
            'center': np.array([cx, cy, cz]),
            'score': m['score'],
            'rb_ratio': rb_ratio,
            'rg_ratio': rg_ratio,
            'r': r, 'g': g, 'b': b,
            'z_max': z_max,
            'h_range': h_range,
            'prompt': prompt,
        })
        seen_centers.append([cx, cy])
        print(f"[CARTON] '{prompt}' score={m['score']:.3f} ctr=({cx:.3f},{cy:.3f},{cz:.3f}) z=[{z_min:.3f},{z_max:.3f}] h={h_range:.3f} ext=({ext[0]:.3f},{ext[1]:.3f},{ext[2]:.3f}) RGB=({r:.0f},{g:.0f},{b:.0f}) R/B={rb_ratio:.2f}", flush=True)

if not candidates:
    raise RuntimeError("No upright milk carton candidates found")

# Disambiguate: milk has red label on white. Orange juice has yellow/orange labels.
# Both can have R/B > 1.5, but the "carton of milk" prompt's TOP score should be milk.
# Order by SAM3 score within "carton of milk" prompt first, then prefer reddish candidates.
def carton_priority(c):
    # Highest score "carton of milk" detection wins; fall back to RB ratio
    prompt_priority = 0 if c['prompt'] == 'carton of milk' else (1 if c['prompt'] == 'milk carton' else 2)
    return (prompt_priority, -c['score'])

# But we also want to filter R/B: must be reddish (> 1.4); if all available are
# above 1.4, prefer the highest-scoring "carton of milk" detection.
reddish = [c for c in candidates if c['rb_ratio'] > 1.40]
if reddish:
    reddish.sort(key=carton_priority)
    chosen = reddish[0]
else:
    candidates.sort(key=carton_priority)
    chosen = candidates[0]

print(f"[CHOSEN] milk carton ctr={chosen['center']} R/B={chosen['rb_ratio']:.2f} h={chosen['h_range']:.3f} prompt={chosen['prompt']} score={chosen['score']:.3f}", flush=True)

milk_pts = chosen['pts']
milk_obb = chosen['obb']
milk_center = chosen['center']
milk_top_z = chosen['z_max']
milk_height = chosen['h_range']

# --- Localize basket ---
basket_pts = None
basket_score = 0.0
for prompt in ["wicker basket", "basket"]:
    masks = segment_sam3_text_prompt(rgb, prompt)
    if not masks:
        continue
    cand = []
    for m in sorted(masks, key=lambda d: d['score'], reverse=True)[:5]:
        if m['score'] < 0.30:
            continue
        bp = mask_to_world_points(m['mask'].astype(np.uint8), depth_img, K, E)
        if bp is None or len(bp) < 50:
            continue
        h = float(bp[:, 2].max() - bp[:, 2].min())
        if h < 0.08:
            continue
        try:
            obb = get_oriented_bounding_box_from_3d_points(bp)
            mx_ext = max(float(obb['extent'][0]), float(obb['extent'][1]))
            if mx_ext < 0.10:
                continue
        except Exception:
            continue
        cand.append((m['score'], bp))
    if cand:
        cand.sort(key=lambda x: x[0], reverse=True)
        basket_score, basket_pts = cand[0]
        print(f"[BASKET] '{prompt}' score={basket_score:.3f} N={len(basket_pts)}", flush=True)
        break

if basket_pts is None:
    raise RuntimeError("Basket not found")

bx_c = (np.percentile(basket_pts[:, 0], 10) + np.percentile(basket_pts[:, 0], 90)) / 2
by_c = (np.percentile(basket_pts[:, 1], 10) + np.percentile(basket_pts[:, 1], 90)) / 2
basket_floor_z = float(np.percentile(basket_pts[:, 2], 20))
basket_top_z = float(np.percentile(basket_pts[:, 2], 90))
print(f"[BASKET-CTR] xy=({bx_c:.3f},{by_c:.3f}) floor_z={basket_floor_z:.3f} top_z={basket_top_z:.3f}", flush=True)

# --- Grasp milk carton (upright) ---
mx, my = float(milk_center[0]), float(milk_center[1])
# Milk is upright, ~15cm tall. Grasp top-down at top - 4cm (upper third of body).
grasp_z = milk_top_z - 0.040
print(f"[GRASP] xy=({mx:.3f},{my:.3f}) z={grasp_z:.3f} top={milk_top_z:.3f} h={milk_height:.3f}", flush=True)

open_gripper()

# Pre-approach: high above
j = solve_ik([mx, my, grasp_z + 0.18], quat.tolist())
if j is not None:
    move_to_joints(j)

# Descend to grasp
j = solve_ik([mx, my, grasp_z], quat.tolist())
if j is not None:
    move_to_joints(j)

close_gripper()

obs2 = get_observation()
gw = obs2.get('robot_state', {}).get('gripper_open_width', None)
print(f"[GRIP] gw={gw}", flush=True)

# --- Lift ---
lift_z = max(grasp_z + 0.25, 0.35)
j = solve_ik([mx, my, lift_z], quat.tolist())
if j is not None:
    move_to_joints(j)

# --- Transport above basket (3-step lateral at lift height) ---
for frac in [0.33, 0.67, 1.0]:
    wx = mx + frac * (bx_c - mx)
    wy = my + frac * (by_c - my)
    j = solve_ik([wx, wy, lift_z], quat.tolist())
    if j is not None:
        move_to_joints(j)

# --- Lower into basket ---
# Drop z above basket floor, accounting for object height + small clearance.
# For a 15cm upright carton, drop from basket_top + 0.05 to drop bottom-first.
drop_z = max(basket_top_z + 0.05, basket_floor_z + milk_height + 0.05)
print(f"[DROP] xy=({bx_c:.3f},{by_c:.3f}) drop_z={drop_z:.3f}", flush=True)

# Multi-step descent
for step_z in [lift_z, basket_top_z + 0.20, drop_z + 0.06, drop_z]:
    j = solve_ik([bx_c, by_c, step_z], quat.tolist())
    if j is not None:
        move_to_joints(j)

open_gripper()
for _ in range(10):
    get_observation()

print("[DONE]", flush=True)
