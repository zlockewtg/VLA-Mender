import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


# --- Init ---
goto_home_joint_position()
for _ in range(3):
    open_gripper()
    close_gripper()
open_gripper()

# --- Observe ---
obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

# --- Identify both cans (SAM3 returns both for any can-related prompt) ---
# The alphabet soup can has a colorful/blue label (R/B ~ 0.98).
# The tomato sauce can has a red label (R/B > 1.20).
candidate_cans = []
seen_centers = []
for prompt in ["soup can", "can of soup", "can", "alphabet soup can"]:
    masks = segment_sam3_text_prompt(rgb, prompt)
    if not masks:
        continue
    for m in sorted(masks, key=lambda d: d['score'], reverse=True)[:5]:
        if m['score'] < 0.50:
            continue
        pts = mask_to_world_points(m['mask'].astype(np.uint8), depth_img, K, E)
        if pts is None or len(pts) < 20:
            continue
        obb = get_oriented_bounding_box_from_3d_points(pts)
        cx, cy, cz = obb['center']
        # Filter by canonical can geometry: ~7cm dia x ~6-9cm tall, on table (z<0.10)
        if cz > 0.12 or cz < 0.02:
            continue
        h_range = pts[:, 2].max() - pts[:, 2].min()
        if h_range < 0.04 or h_range > 0.15:
            continue
        # Skip duplicates (same physical can, different prompt)
        is_dup = False
        for sc in seen_centers:
            if np.linalg.norm(np.array([cx, cy]) - np.array(sc[:2])) < 0.04:
                is_dup = True
                break
        if is_dup:
            continue
        # Color stats
        y_idxs, x_idxs = np.where(m['mask'])
        if len(y_idxs) < 5:
            continue
        r = float(np.mean(rgb[y_idxs, x_idxs, 0]))
        g = float(np.mean(rgb[y_idxs, x_idxs, 1]))
        b = float(np.mean(rgb[y_idxs, x_idxs, 2]))
        rb_ratio = r / (b + 1e-5)
        candidate_cans.append({
            'mask': m['mask'],
            'pts': pts,
            'obb': obb,
            'center': np.array([cx, cy, cz]),
            'score': m['score'],
            'rb_ratio': rb_ratio,
            'r': r, 'g': g, 'b': b,
        })
        seen_centers.append([cx, cy])
        print(f"[CAN] prompt={prompt} ctr=({cx:.3f},{cy:.3f},{cz:.3f}) RGB=({r:.0f},{g:.0f},{b:.0f}) R/B={rb_ratio:.2f}", flush=True)
    if len(candidate_cans) >= 2:
        break

if not candidate_cans:
    raise RuntimeError("No can candidates found")

# Pick the can with lowest R/B ratio (alphabet soup is less red than tomato sauce)
candidate_cans.sort(key=lambda c: c['rb_ratio'])
chosen_can = candidate_cans[0]
print(f"[CHOSEN] alphabet soup can ctr={chosen_can['center']} R/B={chosen_can['rb_ratio']:.2f}", flush=True)

can_pts = chosen_can['pts']
can_obb = chosen_can['obb']
can_center = chosen_can['center']
can_top_z = can_center[2] + can_obb['extent'][2] / 2
can_height = float(can_obb['extent'][2])

# --- Localize basket ---
basket_pts = None
for prompt in ["wicker basket", "basket"]:
    masks = segment_sam3_text_prompt(rgb, prompt)
    if not masks:
        continue
    best = max(masks, key=lambda d: d['score'])
    if best['score'] < 0.30:
        continue
    bp = mask_to_world_points(best['mask'].astype(np.uint8), depth_img, K, E)
    if bp is None or len(bp) < 50:
        continue
    basket_pts = bp
    basket_mask = best['mask']
    basket_score = best['score']
    print(f"[BASKET] prompt={prompt} score={basket_score:.3f} N={len(bp)}", flush=True)
    break

if basket_pts is None:
    raise RuntimeError("Basket not found")

# True basket center (use p10/p90 midpoint to avoid wall-bias)
bx_c = (np.percentile(basket_pts[:, 0], 10) + np.percentile(basket_pts[:, 0], 90)) / 2
by_c = (np.percentile(basket_pts[:, 1], 10) + np.percentile(basket_pts[:, 1], 90)) / 2
basket_floor_z = float(np.percentile(basket_pts[:, 2], 5))
basket_top_z = float(np.percentile(basket_pts[:, 2], 90))
print(f"[BASKET-CTR] xy=({bx_c:.3f},{by_c:.3f}) floor_z={basket_floor_z:.3f} top_z={basket_top_z:.3f}", flush=True)

# --- Grasp can ---
quat = make_topdown_quat(0)
cx, cy = float(can_center[0]), float(can_center[1])
grasp_z = can_top_z - 0.030  # mid-body (top - 3cm)
print(f"[GRASP] xy=({cx:.3f},{cy:.3f}) z={grasp_z:.3f} can_top={can_top_z:.3f} can_h={can_height:.3f}", flush=True)

open_gripper()

# Pre-approach
j = solve_ik([cx, cy, grasp_z + 0.15], quat.tolist())
if j is not None:
    move_to_joints(j)

# Descend to grasp
j = solve_ik([cx, cy, grasp_z], quat.tolist())
if j is not None:
    move_to_joints(j)

close_gripper()

obs2 = get_observation()
gw = obs2.get('robot_state', {}).get('gripper_open_width', None)
print(f"[GRIP] gw={gw}", flush=True)

# --- Lift ---
lift_z = grasp_z + 0.25
j = solve_ik([cx, cy, lift_z], quat.tolist())
if j is not None:
    move_to_joints(j)

# --- Transport above basket (3-step lateral) ---
for frac in [0.33, 0.67, 1.0]:
    wx = cx + frac * (bx_c - cx)
    wy = cy + frac * (by_c - cy)
    j = solve_ik([wx, wy, lift_z], quat.tolist())
    if j is not None:
        move_to_joints(j)

# --- Lower into basket ---
# Use minimal-bounce drop: drop_z = floor + can_height + small margin
drop_z = basket_floor_z + can_height + 0.05
print(f"[DROP] xy=({bx_c:.3f},{by_c:.3f}) drop_z={drop_z:.3f}", flush=True)

# Multi-step descent
for step_z in [lift_z, basket_top_z + 0.15, drop_z + 0.05, drop_z]:
    j = solve_ik([bx_c, by_c, step_z], quat.tolist())
    if j is not None:
        move_to_joints(j)

open_gripper()
for _ in range(8):
    get_observation()

print("[DONE]", flush=True)
