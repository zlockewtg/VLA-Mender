"""
LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket
Task type: pick-and-place

Pick: butter package — small yellow/orange flat rectangular box, ~7.6cm x 4cm x 1.7cm,
      center near (0.55, 0.05, 0.030).
Place: wicker basket (rim ~0.20; ~17–19cm wide; center near (0.53, 0.27, 0.10)).

Disambiguation:
  Scene contains: butter (yellow), cream cheese (blue) — both flat (ext_z < 0.04),
                  alphabet soup can (ext_z ~0.06), tomato sauce can (ext_z ~0.06),
                  basket (ext_xy ~0.18), and a wall picture at z~0.42.

  SAM3 prompts: "butter package" (0.92–0.94 on butter; also fires on cream cheese),
                 "small box" (similar). Cannot distinguish butter from cream cheese by SAM3 alone.

  Color disambiguation: butter has high (R+G)/B ratio (>2.5) due to its yellow/orange label;
                         cream cheese box is dark blue with (R+G)/B ~1.5.
                         Pick the flat-box candidate with the highest (R+G)/B ratio.

  Geometric: ext_z < 0.04 to exclude cans (~0.06); pos_z < 0.06 to exclude wall fixtures (~0.42).

Reuses cream_basket pattern (chunk 3, 30/30) for grasp + basket placement.
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
                top=10, min_score=0.10, return_all=False):
    """Try each prompt, return best (or all) candidate(s) matching geometry filters."""
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
            candidates.append({"score": score, "prompt": p, "center": c, "ext": ext,
                                "pts": pts, "mask": mask})
    if return_all:
        return candidates
    if not candidates:
        return None
    return max(candidates, key=lambda d: d["score"])


def color_score_yellow(rgb, mask):
    """Return (R+G)/B average over masked pixels. Higher = more yellow/orange.
    Butter (yellow): ~4.1.  Cream cheese (blue/teal): ~1.5."""
    ys, xs = np.where(mask > 0)
    if len(ys) < 20:
        return 0.0
    pixels = rgb[ys, xs].astype(float)
    rmean = pixels[:, 0].mean()
    gmean = pixels[:, 1].mean()
    bmean = pixels[:, 2].mean()
    return (rmean + gmean) / max(bmean, 1.0)


def dedupe_candidates(cands, radius=0.05):
    """Remove duplicate candidates whose 3D centers are within `radius` of each other.
    Keeps the highest-scoring one in each cluster."""
    cands_sorted = sorted(cands, key=lambda d: -d["score"])
    kept = []
    for c in cands_sorted:
        if any(np.linalg.norm(c["center"][:2] - k["center"][:2]) < radius for k in kept):
            continue
        kept.append(c)
    return kept


# === Step 1: Observe scene ===
# LIBERO-Pro distractor objects spawn/fall during the first ~50 physics steps. Need to
# advance physics enough for the scene to fully settle before localization.
# goto_home_joint_position() is a long blocking move that steps physics for ~100+ steps.
goto_home_joint_position()
open_gripper()
open_gripper()
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

# === Step 2: Localize butter package ===
# Multi-prompt strategy: SAM3 fires on multiple flat boxes, so we collect ALL candidates that
# match the flat-box geometry, dedupe by 3D position, then pick the one with highest yellow color.
butter_prompts = ["butter package", "butter", "small box", "rectangular box", "flat box",
                  "yellow butter package", "yellow box", "orange box"]
all_cands = find_object(rgb, depth_img, K, E, butter_prompts,
                        ext_z_max=0.04, ext_xy_min=0.04, ext_xy_max=0.12,
                        pos_z_max=0.06, min_score=0.40, return_all=True)
all_cands = dedupe_candidates(all_cands, radius=0.05)
print(f"[BUTTER] {len(all_cands)} flat-box candidates after dedupe")
for c in all_cands:
    cs = color_score_yellow(rgb, c["mask"])
    print(f"   prompt='{c['prompt']}' score={c['score']:.3f} center={c['center'].tolist()} (R+G)/B={cs:.2f}")

# Pick candidate with highest yellow color (butter ≈ 4+, blue cream cheese ≈ 1.5)
if not all_cands:
    raise RuntimeError("No flat-box candidates for butter")
butter = max(all_cands, key=lambda c: color_score_yellow(rgb, c["mask"]))
butter_color = color_score_yellow(rgb, butter["mask"])
butter_center = butter["center"]
butter_pts = butter["pts"]
butter_mask = butter["mask"]
print(f"[BUTTER] picked prompt='{butter['prompt']}' score={butter['score']:.3f} (R+G)/B={butter_color:.2f}")
print(f"         center={butter_center.tolist()} ext={butter['ext'].tolist()}")
if butter_color < 2.0:
    print(f"[BUTTER WARN] color score low ({butter_color:.2f}); proceeding anyway")

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

# Compute basket rim z
basket_rim_z = float(np.percentile(basket_pts[:, 2], 95))
print(f"[BASKET] rim_z={basket_rim_z:.3f}")

# === Step 4: Plan grasp ===
quat = make_topdown_quat(0)

try:
    grasp_poses, grasp_scores = plan_grasp(depth_img, K, butter_mask)
except Exception as e:
    print(f"[GRASP] plan_grasp raised: {e}")
    grasp_poses, grasp_scores = None, None

if grasp_poses is not None and len(grasp_poses) > 0:
    best_grasp_T, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
    if best_grasp_T is None:
        best_grasp_T = E @ grasp_poses[grasp_scores.argmax()]
    grasp_pos, gquat = decompose_transform(best_grasp_T)
    obj_obb = get_oriented_bounding_box_from_3d_points(butter_pts)
    obb_xy = obj_obb["center"][:2]
    dist_xy = np.linalg.norm(grasp_pos[:2] - obb_xy)
    # Flat box symmetric — OBB center is the best grasp point. Snap aggressively (>1cm).
    if dist_xy > 0.010:
        print(f"[GRASP] plan_grasp XY off by {dist_xy:.3f}m → snap to OBB center")
        grasp_pos = np.array([obb_xy[0], obb_xy[1], grasp_pos[2]])
    print(f"[GRASP] plan_grasp grasp_pos={grasp_pos.tolist()}")
else:
    print("[GRASP] plan_grasp empty → fall back to OBB center")
    obj_obb = get_oriented_bounding_box_from_3d_points(butter_pts)
    grasp_pos = np.array([obj_obb["center"][0], obj_obb["center"][1], butter_pts[:, 2].max()])

# Force grasp z = mid-bottom of butter (flat box ~1.7cm thick).
# Aim fingertips slightly ABOVE the table (table_z + 0.005) so fingers reach down past
# the butter top and grip its sides. This is more robust to small object disturbances.
top_z = float(butter_pts[:, 2].max())
bot_z = float(butter_pts[:, 2].min())
table_z = bot_z  # butter sits on the table
grasp_z = max(table_z + 0.005, top_z - 0.012)
grasp_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_z])
print(f"[GRASP] final grasp_pos={grasp_pos.tolist()}, top_z={top_z:.3f}")

# === Step 5: Execute pick (gripper already open from settling step) ===
goto_pose(grasp_pos, quat, z_approach=0.15)
goto_pose(grasp_pos, quat)
close_gripper()

# Check grip width
obs2 = get_observation()
rcp = obs2.get("robot_cartesian_pos", [])
gw = rcp[7] if len(rcp) >= 8 else 0
print(f"[GRIP] width={gw:.3f}")

# === Step 6: Lift ===
lift_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.20])
joints = solve_ik(lift_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 7: Move above basket (clear rim) ===
transit_z = max(lift_pos[2], basket_rim_z + 0.20)
above_basket = np.array([basket_center[0], basket_center[1], transit_z])
joints = solve_ik(above_basket.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 8: Lower to release just above rim ===
obj_height = butter_pts[:, 2].max() - butter_pts[:, 2].min()
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
