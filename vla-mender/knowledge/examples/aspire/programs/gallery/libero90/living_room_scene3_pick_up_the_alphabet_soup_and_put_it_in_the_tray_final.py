"""
LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray
Task type: pick-and-place

Pick: alphabet soup CAN — short can (~6cm tall, ~7cm dia, top z≈0.09).
      Label is blue/yellow with neutral mean RGB; mean R/B pixel ratio ≈ 0.99.
Place: wooden tray on the right side of the table (rim z≈0.10).

Scene caveats (LIVING_ROOM_SCENE3, mirror of tomato_tray):
- The robot arm at HOME occludes the cans; deocclude by moving the arm aside first.
- The scene contains TWO similar-shaped cans:
  * tomato sauce can (red label, mean RGB ratio R/B ≈ 1.47) — DISTRACTOR
  * alphabet soup can (blue/yellow label, R/B ≈ 0.99) — TARGET
  Both score ~0.66–0.93 on prompts "alphabet soup can" / "soup can". Their
  positions swap across seeds. To disambiguate reliably, take the top-K SAM3
  hits and pick the one with the **lower red/blue mean RGB ratio**. The
  alphabet soup can wins (R/B ≈ 0.99 < 1.2 < 1.47 ≈ tomato sauce).
- Other distractors: ketchup bottle (taller, far +x), cream cheese box (flat,
  ext_z<0.04), butter (small). The candidate filter (extent + z<0.10) rejects
  them.
- Release height: drop from 5cm above tray rim (proven robust on tomato_tray
  30/30; +0.08 caused bounce-out on short cans).

Validated on seeds 51–80: target ≥3/5 on 51-55.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


def localize_object(rgb, depth, K, E, prompts, min_score=0.0):
    """Top-1 SAM3 localizer; tries prompts in order, returns first viable hit."""
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        best = max(masks, key=lambda d: d["score"])
        if best["score"] < min_score:
            continue
        mask = best["mask"].astype(np.uint8)
        pts = mask_to_world_points(mask, depth_img, K, E)
        if pts is None or len(pts) < 10:
            continue
        center = get_oriented_bounding_box_from_3d_points(pts)["center"]
        return center, pts, mask, best["score"]
    return None, None, None, 0.0


def alphabet_can_localize(rgb, depth, K, E):
    """Pick the LEAST-red top-K SAM3 hit for can prompts.
    Discriminator: mean R/B ratio (alphabet soup ≈0.99, tomato sauce ≈1.47)."""
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

    # Aggregate hits across multiple prompts (both score similarly on each).
    cands = []
    seen_centers = []
    for prompt in ["soup can", "alphabet soup can", "can"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        masks = sorted(masks, key=lambda d: -d["score"])
        for m in masks[:4]:
            if m["score"] < 0.4:
                continue
            mk = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mk, depth_img, K, E)
            if pts is None or len(pts) < 100:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            ext = obb["extent"]
            c = obb["center"]
            # Filter to can-shaped tabletop objects
            if c[2] > 0.10:
                continue
            if not (0.04 <= max(ext[0], ext[1]) <= 0.12):
                continue
            if not (0.04 <= ext[2] <= 0.10):
                continue
            # Dedupe by 3D position (within 4cm)
            if any(np.linalg.norm(c[:2] - sc[:2]) < 0.04 for sc in seen_centers):
                continue
            seen_centers.append(c)
            # Color discriminator (mean R/B ratio over masked pixels)
            bool_mask = m["mask"].astype(bool)
            rgb_pixels = rgb[bool_mask]
            if len(rgb_pixels) == 0:
                continue
            r = float(rgb_pixels[:, 0].mean())
            b = float(rgb_pixels[:, 2].mean())
            rb_ratio = r / (b + 1e-6)
            cands.append({
                "mask": mk, "pts": pts, "center": c, "score": m["score"],
                "rb_ratio": rb_ratio,
            })

    if not cands:
        return None, None, None, 0.0
    # Pick the LEAST red (lowest R/B) — alphabet soup has neutral colors
    cands.sort(key=lambda d: d["rb_ratio"])
    chosen = cands[0]
    print(f"[INFO] alphabet candidates: {[(round(c['rb_ratio'],2), c['center'].tolist()) for c in cands]}")
    return chosen["center"], chosen["pts"], chosen["mask"], chosen["score"]


# === Step 1: Move arm aside to deocclude the scene ===
quat_topdown = make_topdown_quat(0)
side_pos = np.array([0.4, -0.4, 0.3])
joints = solve_ik(side_pos.tolist(), quat_topdown.tolist())
if joints is not None:
    move_to_joints(joints)
else:
    for fallback_pos in [[0.3, -0.4, 0.35], [0.3, 0.4, 0.35], [0.5, -0.3, 0.3]]:
        joints = solve_ik(fallback_pos, quat_topdown.tolist())
        if joints is not None:
            move_to_joints(joints)
            break

for _ in range(3):
    obs = get_observation()

# === Step 2: Localize the alphabet soup can with color disambiguation ===
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]

obj_center, obj_pts, obj_mask, obj_score = alphabet_can_localize(rgb, depth, K, E)
if obj_center is None:
    # Fallback: simple top-1 SAM3
    obj_center, obj_pts, obj_mask, obj_score = localize_object(
        rgb, depth, K, E,
        ["alphabet soup can", "soup can", "alphabet soup"],
        min_score=0.4,
    )
if obj_center is None:
    raise RuntimeError("Alphabet soup can not found")
print(f"[INFO] alphabet soup can score={obj_score:.3f} center={obj_center.tolist()}")

# Localize tray. SAM3 mask sometimes leaks beyond tray (extent X up to 0.44 on seed 51)
# pulling OBB center toward distractors. Filter to RIM points (z>0.07) which are the
# tray walls — gives a robust center for the actual tray box.
tray_prompts = ["wooden tray", "tray", "serving tray"]
tgt_center_full, tgt_pts, _, tgt_score = localize_object(rgb, depth, K, E, tray_prompts, min_score=0.3)
if tgt_center_full is None:
    raise RuntimeError("Tray not found")
tray_rim_z = float(tgt_pts[:, 2].max())
rim_pts = tgt_pts[tgt_pts[:, 2] > 0.07]
if len(rim_pts) >= 50:
    tgt_center = get_oriented_bounding_box_from_3d_points(rim_pts)["center"]
else:
    tgt_center = tgt_center_full
print(f"[INFO] tray score={tgt_score:.3f} center_full={tgt_center_full.tolist()} center_rim={tgt_center.tolist()} rim_z={tray_rim_z:.3f}")

# === Step 3: Decide grasp XY ===
# OBB X is biased ~1-2cm AWAY from the robot due to perspective-elongated depth
# cloud (mask extent X ≈ 0.098m for a 7cm can). plan_grasp uses GraspNet which
# operates on the depth cloud + finds physically-valid grasp pose; it returns
# XY consistently ~1-2cm closer to the robot — i.e. the actual can center.
obj_obb = get_oriented_bounding_box_from_3d_points(obj_pts)

depth_img_local = depth[:, :, 0] if len(depth.shape) == 3 else depth
grasp_x = grasp_y = None
try:
    grasp_poses, grasp_scores = plan_grasp(depth_img_local, K, obj_mask)
    if len(grasp_poses) > 0:
        # Try select_top_down_grasp first
        best_grasp_world, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
        if best_grasp_world is None:
            best_grasp_world = E @ grasp_poses[grasp_scores.argmax()]
        gpos, _ = decompose_transform(best_grasp_world)
        # Use planner XY, but only if it falls within OBB bounds (sanity check)
        d_obb = np.linalg.norm(np.asarray(gpos)[:2] - obj_obb["center"][:2])
        if d_obb < 0.04:  # within 4cm of OBB center
            grasp_x = float(gpos[0])
            grasp_y = float(gpos[1])
            print(f"[INFO] using plan_grasp XY: ({grasp_x:.3f},{grasp_y:.3f}) (OBB diff {d_obb*100:.1f}cm)")
except Exception as e:
    print(f"[WARN] plan_grasp failed: {e}")

if grasp_x is None:
    # Fallback to OBB center
    grasp_x = float(obj_obb["center"][0])
    grasp_y = float(obj_obb["center"][1])
    print(f"[INFO] fallback to OBB XY: ({grasp_x:.3f},{grasp_y:.3f})")

# z from "body" pixels only (drop shadow/contamination)
xy_dist = np.linalg.norm(obj_pts[:, :2] - obj_obb["center"][:2], axis=1)
body_pts = obj_pts[xy_dist < 0.04]
if len(body_pts) < 20:
    body_pts = obj_pts
body_z_max = float(body_pts[:, 2].max())
body_z_min = float(body_pts[:, 2].min())
print(f"[INFO] body_pts={len(body_pts)}/{len(obj_pts)}  z_range=[{body_z_min:.3f}, {body_z_max:.3f}]")

quat = make_topdown_quat(0)

# Short can: grasp 1.5cm below visible top
target_grasp_z = body_z_max - 0.015
target_grasp_z = max(target_grasp_z, body_z_min + 0.02)
grasp_pos = np.array([grasp_x, grasp_y, target_grasp_z])
print(f"[INFO] grasp_pos={grasp_pos.tolist()}")

# === Step 4: Execute pick ===
open_gripper()
goto_pose(grasp_pos, quat, z_approach=0.15)
goto_pose(grasp_pos, quat)
close_gripper()

# Lift
lift_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.20])
joints = solve_ik(lift_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 5: Move above tray ===
above_tgt = np.array([tgt_center[0], tgt_center[1], lift_pos[2]])
joints = solve_ik(above_tgt.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 6: Lower to release (close drop reduces bounce-out) ===
release_pos = np.array([tgt_center[0], tgt_center[1], tray_rim_z + 0.05])
joints = solve_ik(release_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

open_gripper()

# Settle
for _ in range(5):
    get_observation()
