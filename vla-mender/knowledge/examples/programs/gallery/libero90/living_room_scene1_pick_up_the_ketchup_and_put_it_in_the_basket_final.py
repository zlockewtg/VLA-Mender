"""
LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket
Task type: pick-and-place

Pick: ketchup bottle. In LIVING_ROOM_SCENE1, the ketchup is LYING FLAT
       (ext_x≈0.15m long axis, ext_y≈0.06m short axis, ext_z≈0.03m thick) at
       c≈(0.335, -0.169, 0.09). NOT an upright tall bottle.
Place: wicker basket at c≈(0.51, 0.26, 0.10), rim_z ≈ 0.16, ext_xy≈0.19m.

Disambiguation:
  - "ketchup bottle" / "orange bottle" / "bottle" all score ~0.91-0.95 on the
    ketchup at x=0.335. Two short cans live at x≈0.585-0.595 (ext_z ~0.06,
    ext_xy ~0.07-0.10) and could be matched by "bottle" if not filtered.
  - Filter: pos_x < 0.50 (ketchup is back-left, far from cans/basket).
  - Other distractors: butter/cream cheese box (ext_z ~0.02, x≈0.37, y≈0.05).
    Filter by ext_xy_min ≈ 0.10 to keep only the elongated ketchup.

Grasp:
  - Lying-flat bottle: top_z ≈ 0.105 (centroid + ext_z/2). Use yaw=0 so fingers
    close along world-Y direction = short axis (6cm < gripper 8cm).
  - Grasp z = top_z - 0.010 (just below the visible top surface).
  - Use OBB center for XY (clean elongated shape, well-localized).

Place: reuses cream_basket pattern (chunk 3, 30/30 in tomato_basket).
  - rim_z = p95 of basket pts z.
  - Transit at max(lift_z, rim_z + 0.20) to clear walls.
  - Release at rim_z + obj_height + 0.05.

Physics-settling: toggle gripper 3× before first observation
  (LIBERO-Pro multi-object scenes spawn distractors during first ~150 steps).
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
                pos_x_max=None, pos_x_min=None, pos_y_max=None, pos_y_min=None,
                top=10, min_score=0.10, return_all=False):
    """Try each prompt, collect candidates matching geometry filters."""
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
            if pos_x_max is not None and c[0] > pos_x_max:
                continue
            if pos_x_min is not None and c[0] < pos_x_min:
                continue
            if pos_y_max is not None and c[1] > pos_y_max:
                continue
            if pos_y_min is not None and c[1] < pos_y_min:
                continue
            candidates.append({"score": score, "prompt": p, "center": c, "ext": ext,
                               "pts": pts, "mask": mask})
    if return_all:
        return candidates
    if not candidates:
        return None
    return max(candidates, key=lambda d: d["score"])


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

# === Step 2: Localize the ketchup bottle (lying flat, back-left of table) ===
ketchup_prompts = ["orange bottle", "ketchup bottle", "tall bottle", "bottle",
                   "red bottle", "ketchup", "tomato ketchup"]
# Ketchup at x≈0.335, y≈-0.17, z≈0.09. ext_x=0.148 (long), so ext_xy_min=0.12
# excludes cans (xy~0.07-0.10) and butter (xy~0.08).
# pos_x_max=0.50 excludes cans at x≈0.585-0.595.
ketchup = find_object(rgb, depth_img, K, E, ketchup_prompts,
                      ext_xy_min=0.12, ext_z_max=0.08,
                      pos_x_max=0.50, pos_z_max=0.15,
                      min_score=0.30)
if ketchup is None:
    raise RuntimeError("Ketchup not found")
obj_center = ketchup["center"]
obj_pts = ketchup["pts"]
obj_mask = ketchup["mask"]
print(f"[KETCHUP] prompt='{ketchup['prompt']}' score={ketchup['score']:.3f}")
print(f"          center={obj_center.tolist()} ext={ketchup['ext'].tolist()}")

# === Step 3: Localize the wicker basket ===
basket_prompts = ["wicker basket", "basket"]
basket = find_object(rgb, depth_img, K, E, basket_prompts,
                     ext_xy_min=0.13, ext_z_min=0.10)
if basket is None:
    raise RuntimeError("Basket not found")
basket_center = basket["center"]
basket_pts = basket["pts"]
print(f"[BASKET]  prompt='{basket['prompt']}' score={basket['score']:.3f}")
print(f"          center={basket_center.tolist()} ext={basket['ext'].tolist()}")

basket_rim_z = float(np.percentile(basket_pts[:, 2], 95))
print(f"[BASKET]  rim_z={basket_rim_z:.3f}")

# === Step 4: Plan grasp ===
# Use yaw=0 → fingers close along world-Y (short axis, ~6cm) of lying-flat bottle.
# This grips the narrower side (gripper opens ~8cm).
quat = make_topdown_quat(0)

# Use OBB center for XY (elongated shape; OBB is robust).
obj_obb = get_oriented_bounding_box_from_3d_points(obj_pts)
grasp_x = float(obj_obb["center"][0])
grasp_y = float(obj_obb["center"][1])

# Body z-range: filter to inliers within 5cm horizontally of OBB center.
xy_dist = np.linalg.norm(obj_pts[:, :2] - obj_obb["center"][:2], axis=1)
body_pts = obj_pts[xy_dist < 0.06]
if len(body_pts) < 50:
    body_pts = obj_pts
body_z_max = float(body_pts[:, 2].max())
body_z_min = float(body_pts[:, 2].min())
print(f"[INFO]    body_pts={len(body_pts)}/{len(obj_pts)}  z_range=[{body_z_min:.3f}, {body_z_max:.3f}]")

# Lying-flat bottle (~3cm tall): grasp 1cm below visible top so fingers wrap upper body.
target_grasp_z = body_z_max - 0.010
grasp_pos = np.array([grasp_x, grasp_y, target_grasp_z])
print(f"[GRASP]   grasp_pos={grasp_pos.tolist()}")

# === Step 5: Execute pick ===
open_gripper()
goto_pose(grasp_pos, quat, z_approach=0.15)
goto_pose(grasp_pos, quat)
close_gripper()

obs2 = get_observation()
rcp = obs2.get("robot_cartesian_pos", [])
gw = rcp[7] if len(rcp) >= 8 else 0
print(f"[GRIP]    width={gw:.3f}")

# === Step 6: Lift ===
lift_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.20])
joints = solve_ik(lift_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 7: Move above basket (transit at safe height) ===
transit_z = max(lift_pos[2], basket_rim_z + 0.20)
above_basket = np.array([basket_center[0], basket_center[1], transit_z])
joints = solve_ik(above_basket.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 8: Lower to release ===
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
