"""
LIVING_ROOM_SCENE2: pick up the orange juice and put it in the basket.

Scene: Cluttered LIVING_ROOM_SCENE2 with milk carton (R/B≈1.85, upright), orange
       juice carton (R/B≈3.3, upright, yellow/orange "Orange Juice" label),
       cream cheese (flat, blue), small flat box, alphabet soup can, tomato sauce
       can, wicker basket on the right (~(0.53, 0.27, 0.10)).

Strategy (mirrors milk task — same scene):
- Move arm to high-left side-position to clear agentview occlusion BEFORE observing.
- SAM3 "carton of orange juice" gives OJ top score (~0.938) > milk (~0.852).
- Multi-prompt collection with upright-carton geometry filter (h_range > 0.10).
- Disambiguate OJ from milk by R/B color: OJ R/B≈3.3 (yellow/orange), milk R/B≈1.85.
  Pick the most-yellow upright carton (highest R/B).
- OJ is upright ~15cm tall. Top-down grasp at top - 0.04 = upper-mid body.
- Basket: SAM3 "wicker basket" / "basket"; p10/p90 midpoint for true center.
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


# --- Find orange juice carton (UPRIGHT, yellow/orange) ---
candidates = []
seen_centers = []

# "carton of orange juice" gives OJ highest score (~0.938 > milk ~0.852).
# Add fallback prompts so we still find it if SAM3 wobbles.
for prompt in ["carton of orange juice", "orange juice carton", "orange juice", "yellow carton", "juice carton"]:
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
        try:
            obb = get_oriented_bounding_box_from_3d_points(pts)
        except Exception:
            continue
        cx, cy, cz = obb['center']
        ext = obb['extent']
        ext_x, ext_y, ext_z = float(ext[0]), float(ext[1]), float(ext[2])
        # OJ carton dimensions ~14×6×5cm. Accept either pose:
        #   - Upright: h_range > 0.10 (~0.15), max horizontal ext ~6-8cm
        #   - On side: h_range ~0.06-0.08, longest extent ~14cm, others ~5-8cm
        # The longest dimension overall (max of all 3 extents) should be ~10-16cm
        # — uniquely identifying a standard carton form factor.
        sorted_ext = sorted([ext_x, ext_y, ext_z], reverse=True)
        longest, mid, shortest = sorted_ext
        if longest < 0.10 or longest > 0.18:
            continue
        if mid < 0.04 or mid > 0.10:
            continue
        if shortest < 0.02 or shortest > 0.10:
            continue
        # Reject extremely tall/short overall objects (basket would have h>0.13)
        if h_range > 0.22:
            continue
        # Skip duplicates (within 5cm in xy)
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
            'shortest_ext': shortest,
            'longest_ext': longest,
            'prompt': prompt,
        })
        seen_centers.append([cx, cy])
        print(f"[OJ-CAND] '{prompt}' score={m['score']:.3f} ctr=({cx:.3f},{cy:.3f},{cz:.3f}) z=[{z_min:.3f},{z_max:.3f}] h={h_range:.3f} ext=({ext[0]:.3f},{ext[1]:.3f},{ext[2]:.3f}) RGB=({r:.0f},{g:.0f},{b:.0f}) R/B={rb_ratio:.2f}", flush=True)

if not candidates:
    raise RuntimeError("No upright OJ carton candidates found")

# Disambiguate OJ vs milk: OJ has yellow/orange "OJ" label (R/B > 2.5), milk
# has red label on white (R/B ~1.85). Pick the most-yellow upright carton.
# R/B > 2.5 strictly excludes milk (~1.85) and includes OJ (~3.3).
yellow_cands = [c for c in candidates if c['rb_ratio'] > 2.50]
if yellow_cands:
    # Among yellow cartons, prefer highest SAM3 score on "carton of orange juice"
    def yellow_priority(c):
        prompt_priority = 0 if c['prompt'] == 'carton of orange juice' else (
            1 if c['prompt'] == 'orange juice carton' else 2)
        return (prompt_priority, -c['score'])
    yellow_cands.sort(key=yellow_priority)
    chosen = yellow_cands[0]
else:
    # Fallback: pick highest R/B among all upright cartons
    candidates.sort(key=lambda c: -c['rb_ratio'])
    chosen = candidates[0]

print(f"[CHOSEN] OJ ctr={chosen['center']} R/B={chosen['rb_ratio']:.2f} h={chosen['h_range']:.3f} prompt={chosen['prompt']} score={chosen['score']:.3f}", flush=True)

oj_pts = chosen['pts']
oj_obb = chosen['obb']
oj_center = chosen['center']
oj_top_z = chosen['z_max']
oj_height = chosen['h_range']
oj_is_upright = oj_height > 0.10
print(f"[POSE] upright={oj_is_upright} h={oj_height:.3f}", flush=True)

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

# --- Grasp OJ carton ---
mx, my = float(oj_center[0]), float(oj_center[1])
if oj_is_upright:
    # Upright ~15cm tall: grasp top-down at top - 4cm (upper third of body).
    grasp_z = oj_top_z - 0.040
else:
    # Lying on side: top z ~0.09, height is shortest extent. Grasp centered
    # mid-height to grip across the short axis.
    grasp_z = max(oj_top_z - 0.020, float(oj_center[2]))

# Yaw=0 (default top-down). Proven from milk task. The OJ carton has nearly
# the same dimensions as milk and works with the same orientation.
quat_grasp = make_topdown_quat(0)
print(f"[GRASP] xy=({mx:.3f},{my:.3f}) z={grasp_z:.3f} top={oj_top_z:.3f} h={oj_height:.3f} upright={oj_is_upright}", flush=True)

open_gripper()

# Pre-approach: high above
j = solve_ik([mx, my, grasp_z + 0.18], quat_grasp.tolist())
if j is not None:
    move_to_joints(j)

# Descend to grasp
j = solve_ik([mx, my, grasp_z], quat_grasp.tolist())
if j is not None:
    move_to_joints(j)

close_gripper()

obs2 = get_observation()
gw = obs2.get('robot_state', {}).get('gripper_open_width', None)
print(f"[GRIP] gw={gw}", flush=True)

# --- Lift ---
lift_z = max(grasp_z + 0.25, 0.35)
j = solve_ik([mx, my, lift_z], quat_grasp.tolist())
if j is not None:
    move_to_joints(j)

# --- Transport above basket (3-step lateral at lift height) ---
for frac in [0.33, 0.67, 1.0]:
    wx = mx + frac * (bx_c - mx)
    wy = my + frac * (by_c - my)
    j = solve_ik([wx, wy, lift_z], quat_grasp.tolist())
    if j is not None:
        move_to_joints(j)

# --- Lower into basket ---
# Drop z above basket floor, accounting for object height + small clearance.
# For lying OJ, h is small (~0.07) — use shortest extent for clearance estimate.
effective_obj_h = oj_height if oj_is_upright else max(oj_height, chosen['shortest_ext'])
drop_z = max(basket_top_z + 0.05, basket_floor_z + effective_obj_h + 0.05)
print(f"[DROP] xy=({bx_c:.3f},{by_c:.3f}) drop_z={drop_z:.3f}", flush=True)

# Multi-step descent
for step_z in [lift_z, basket_top_z + 0.20, drop_z + 0.06, drop_z]:
    j = solve_ik([bx_c, by_c, step_z], quat_grasp.tolist())
    if j is not None:
        move_to_joints(j)

open_gripper()
for _ in range(10):
    get_observation()

print("[DONE]", flush=True)
