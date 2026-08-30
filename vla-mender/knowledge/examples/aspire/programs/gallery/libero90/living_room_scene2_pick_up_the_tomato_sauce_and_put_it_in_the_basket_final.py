"""
LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket
Task type: pick-and-place

Pick: tomato sauce can (red label, ~7cm dia, ~6cm tall; mean R/B ≈ 1.46).
Place: wicker basket (rim z ≈ 0.16; ~17–20cm wide).

Disambiguation (LR2-specific):
  Scene contains TWO short cans:
    * alphabet soup can (yellow label, R/B ≈ 1.0)  Y < -0.05  — DISTRACTOR
    * tomato sauce can (red label, R/B ≈ 1.46)     Y > 0      — TARGET
  Use BOTH:
    1. Y > 0 geometric filter (alphabet soup at left)
    2. R/B > 1.2 color tiebreak (validated chunk 3 / LR1 30/30)

Basket placement: cream_basket pattern (chunk 3, 30/30).
  - rim_z = p95 of basket pts z.
  - Transit at max(lift_z, rim_z + 0.20) to clear walls.
  - Release at rim_z + can_height + 0.05.

Physics-settling: toggle gripper 3× (LIBERO-Pro multi-object scenes spawn distractors).
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


def find_object(rgb, depth_img, K, E, prompts, ext_z_max=None, ext_z_min=None,
                ext_xy_max=None, ext_xy_min=None, pos_z_max=None, pos_z_min=None,
                pos_y_max=None, pos_y_min=None, pos_x_max=None, pos_x_min=None,
                top=10, min_score=0.10, return_all=False):
    """Try each prompt; collect candidates matching geometry filters."""
    candidates = []
    for p in prompts:
        masks = segment_sam3_text_prompt(rgb, p)
        if not masks:
            continue
        for m in masks[:top]:
            score = m.get("score", 0)
            if score < min_score:
                continue
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 50:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c, ext = obb["center"], obb["extent"]
            ext_z, xy_size = ext[2], max(ext[0], ext[1])
            if ext_z_max is not None and ext_z > ext_z_max:
                continue
            if ext_z_min is not None and ext_z < ext_z_min:
                continue
            if ext_xy_max is not None and xy_size > ext_xy_max:
                continue
            if ext_xy_min is not None and xy_size < ext_xy_min:
                continue
            if pos_z_max is not None and c[2] > pos_z_max:
                continue
            if pos_z_min is not None and c[2] < pos_z_min:
                continue
            if pos_y_max is not None and c[1] > pos_y_max:
                continue
            if pos_y_min is not None and c[1] < pos_y_min:
                continue
            if pos_x_max is not None and c[0] > pos_x_max:
                continue
            if pos_x_min is not None and c[0] < pos_x_min:
                continue
            candidates.append({"score": score, "prompt": p, "center": c, "ext": ext,
                                "pts": pts, "mask": mask})
    if return_all:
        return candidates
    if not candidates:
        return None
    return max(candidates, key=lambda d: d["score"])


def red_can_pick(rgb, candidates, threshold=1.2):
    """From a list of candidates, pick the one with highest R/B ratio (>=threshold)."""
    best = None
    best_rb = -1.0
    for c in candidates:
        ys, xs = np.where(c["mask"] > 0)
        if len(ys) < 20:
            continue
        pixels = rgb[ys, xs].astype(float)
        denom = max(pixels[:, 2].mean(), 1.0)
        rb = pixels[:, 0].mean() / denom
        c["rb"] = rb
        if rb > best_rb:
            best_rb = rb
            best = c
    return best


def dedupe_candidates(cands, radius=0.05):
    """Keep highest-score candidate per cluster (3D XY center within radius)."""
    cands_sorted = sorted(cands, key=lambda d: -d["score"])
    kept = []
    for c in cands_sorted:
        if any(np.linalg.norm(c["center"][:2] - k["center"][:2]) < radius for k in kept):
            continue
        kept.append(c)
    return kept


# === Step 1: Settle physics so distractors finish spawning ===
for _ in range(3):
    open_gripper()
    close_gripper()
open_gripper()

obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

# === Step 2: Localize tomato sauce can ===
# LR2-specific: filter Y>0 to exclude alphabet soup at Y<-0.05 (left side).
# Then R/B tiebreak among remaining candidates.
can_prompts = ["tomato sauce can", "can", "red can"]
can_cands = find_object(rgb, depth_img, K, E, can_prompts,
                        ext_z_min=0.04, ext_z_max=0.10,
                        ext_xy_min=0.04, ext_xy_max=0.12,
                        pos_z_max=0.10, pos_y_min=0.0,
                        min_score=0.30, return_all=True)
can_cands = dedupe_candidates(can_cands, radius=0.05)
print(f"[CAN] {len(can_cands)} can-shaped candidates after dedupe (Y>0 filter)")
for c in can_cands:
    ys, xs = np.where(c["mask"] > 0)
    if len(ys) >= 20:
        pix = rgb[ys, xs].astype(float)
        rb = pix[:, 0].mean() / max(pix[:, 2].mean(), 1.0)
    else:
        rb = -1
    print(f"   prompt='{c['prompt']}' score={c['score']:.3f} center={c['center'].tolist()} R/B={rb:.2f}")

if not can_cands:
    # Fallback: drop Y filter, use color only
    print("[CAN] Y>0 filter rejected all; falling back to color-only", flush=True)
    can_cands = find_object(rgb, depth_img, K, E, can_prompts,
                            ext_z_min=0.04, ext_z_max=0.10,
                            ext_xy_min=0.04, ext_xy_max=0.12,
                            pos_z_max=0.10,
                            min_score=0.30, return_all=True)
    can_cands = dedupe_candidates(can_cands, radius=0.05)

if not can_cands:
    raise RuntimeError("No can-shaped candidates found")

can = red_can_pick(rgb, can_cands, threshold=1.2)
if can is None:
    raise RuntimeError("Could not pick a red can candidate")
print(f"[CAN] picked prompt='{can['prompt']}' score={can['score']:.3f} R/B={can.get('rb',0):.2f}")
print(f"      center={can['center'].tolist()} ext={can['ext'].tolist()}")
if can.get("rb", 0) < 1.2:
    print(f"[CAN WARN] R/B={can.get('rb',0):.2f} below threshold; may have grabbed wrong can")

obj_center = can["center"]
obj_pts = can["pts"]
obj_mask = can["mask"]

# === Step 3: Localize basket ===
basket_prompts = ["wicker basket", "basket"]
basket = find_object(rgb, depth_img, K, E, basket_prompts,
                     ext_xy_min=0.13, ext_z_min=0.10)
if basket is None:
    raise RuntimeError("Basket not found")
basket_center = basket["center"]
basket_pts = basket["pts"]
print(f"[BASKET] prompt='{basket['prompt']}' score={basket['score']:.3f}")
print(f"         center={basket_center.tolist()} ext={basket['ext'].tolist()}")

basket_rim_z = float(np.percentile(basket_pts[:, 2], 95))
print(f"[BASKET] rim_z={basket_rim_z:.3f}")

# === Step 4: Plan grasp ===
quat = make_topdown_quat(0)

# Use OBB center for XY (more robust than median for short cylinders).
obj_obb = get_oriented_bounding_box_from_3d_points(obj_pts)
grasp_x = float(obj_obb["center"][0])
grasp_y = float(obj_obb["center"][1])

# z from "body" pixels only (filter to within 4cm horizontally of OBB center).
xy_dist = np.linalg.norm(obj_pts[:, :2] - obj_obb["center"][:2], axis=1)
body_pts = obj_pts[xy_dist < 0.04]
if len(body_pts) < 20:
    body_pts = obj_pts
body_z_max = float(body_pts[:, 2].max())
body_z_min = float(body_pts[:, 2].min())
print(f"[INFO] body_pts={len(body_pts)}/{len(obj_pts)}  z_range=[{body_z_min:.3f}, {body_z_max:.3f}]")

# Short can (~6cm): grasp 1.5cm below visible top.
target_grasp_z = body_z_max - 0.015
target_grasp_z = max(target_grasp_z, body_z_min + 0.02)
grasp_pos = np.array([grasp_x, grasp_y, target_grasp_z])
print(f"[GRASP] grasp_pos={grasp_pos.tolist()}")

# === Step 5: Execute pick ===
open_gripper()
goto_pose(grasp_pos, quat, z_approach=0.15)
goto_pose(grasp_pos, quat)
close_gripper()

obs2 = get_observation()
rcp = obs2.get("robot_cartesian_pos", [])
gw = rcp[7] if len(rcp) >= 8 else 0
print(f"[GRIP] width={gw:.3f}")

# === Step 6: Lift ===
lift_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.20])
joints = solve_ik(lift_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 7: Move above basket (clear rim during transit) ===
transit_z = max(lift_pos[2], basket_rim_z + 0.20)
above_basket = np.array([basket_center[0], basket_center[1], transit_z])
joints = solve_ik(above_basket.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 8: Lower to release just above rim ===
obj_height = obj_pts[:, 2].max() - obj_pts[:, 2].min()
release_z = basket_rim_z + obj_height + 0.05
release_pos = np.array([basket_center[0], basket_center[1], release_z])
print(f"[RELEASE] z={release_z:.3f} (rim={basket_rim_z:.3f}, obj_h={obj_height:.3f})")
joints = solve_ik(release_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 9: Release ===
open_gripper()

# === Step 10: Retreat upward ===
retreat_pos = np.array([release_pos[0], release_pos[1], release_pos[2] + 0.15])
joints = solve_ik(retreat_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# Settle
for _ in range(5):
    get_observation()
