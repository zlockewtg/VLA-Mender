"""
LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket
Task type: pick-and-place

Pick: cream cheese box (small flat blue rectangular box, ~8cm x 4cm x 1.8cm,
       center near (0.37, 0.05, 0.03)).
Place: wicker basket (rim ~0.16, interior ~17cm wide; center near (0.52, 0.25, 0.09)).

Disambiguation: 2 cans on left at world_x~0.59, similar XY-extent but ext_z>=0.06.
                Filter cream_cheese candidates by ext_z < 0.04 (cream cheese is flat).
SAM3 prompts: "small box" 0.69-0.88, "butter package" 0.52-0.78, "rectangular box" 0.46-0.62.
              "wicker basket" 0.84-0.86, "basket" 0.85-0.88.
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
                top=10, min_score=0.10):
    """Try each prompt, return best candidate matching geometry filters."""
    best = None
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
            cand = {"score": score, "prompt": p, "center": c, "ext": ext,
                    "pts": pts, "mask": mask}
            if best is None or score > best["score"]:
                best = cand
    return best


# === Step 1: Observe scene ===
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

# === Step 2: Localize cream cheese box (flat, thin) ===
cream_prompts = ["small box", "butter package", "rectangular box", "blue box",
                 "cream cheese box", "cream cheese"]
# Filter: thin flat (ext_z < 0.04 rules out cans which are 6cm tall)
# AND XY size 0.04-0.12 (rules out tiny noise & big basket)
cream = find_object(rgb, depth_img, K, E, cream_prompts,
                    ext_z_max=0.04, ext_xy_min=0.04, ext_xy_max=0.12,
                    pos_z_max=0.06)
if cream is None:
    raise RuntimeError("Cream cheese not found")
cream_center = cream["center"]
cream_pts = cream["pts"]
cream_mask = cream["mask"]
print(f"[CREAM] prompt='{cream['prompt']}' score={cream['score']:.3f}")
print(f"        center={cream_center.tolist()} ext={cream['ext'].tolist()}")

# === Step 3: Localize basket (large XY, thick walls) ===
basket_prompts = ["wicker basket", "basket"]
# Filter: large XY (>0.12), tall (>0.10)
basket = find_object(rgb, depth_img, K, E, basket_prompts,
                     ext_xy_min=0.13, ext_z_min=0.10)
if basket is None:
    raise RuntimeError("Basket not found")
basket_center = basket["center"]
basket_pts = basket["pts"]
print(f"[BASKET] prompt='{basket['prompt']}' score={basket['score']:.3f}")
print(f"         center={basket_center.tolist()} ext={basket['ext'].tolist()}")

# Compute basket rim & interior XY center
basket_rim_z = float(np.percentile(basket_pts[:, 2], 95))
print(f"[BASKET] rim_z={basket_rim_z:.3f}")

# === Step 4: Plan grasp ===
# Cream cheese: long axis along x (~8cm), short axis along y (~4cm).
# Default top-down quat with yaw=0 closes fingers along y-direction → grips short axis (~4cm) — good.
# Object top is at ~0.036; aim grasp z just at top (gripper descends to surface).

# Use plan_grasp first; fall back to OBB if it returns garbage.
quat = make_topdown_quat(0)

grasp_poses, grasp_scores = plan_grasp(depth_img, K, cream_mask)
best_grasp = None
if grasp_poses is not None and len(grasp_poses) > 0:
    best_grasp_T, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
    if best_grasp_T is None:
        best_grasp_T = E @ grasp_poses[grasp_scores.argmax()]
    grasp_pos, gquat = decompose_transform(best_grasp_T)
    # Use top-down quat (cream cheese is flat — top-down always best)
    # Validate XY against object center: snap if off-center
    obj_obb = get_oriented_bounding_box_from_3d_points(cream_pts)
    obb_xy = obj_obb["center"][:2]
    dist_xy = np.linalg.norm(grasp_pos[:2] - obb_xy)
    if dist_xy > 0.025:  # >2.5cm off
        print(f"[GRASP] plan_grasp XY off by {dist_xy:.3f}m → snap to OBB center")
        grasp_pos = np.array([obb_xy[0], obb_xy[1], grasp_pos[2]])
    print(f"[GRASP] plan_grasp grasp_pos={grasp_pos.tolist()}")
else:
    print("[GRASP] plan_grasp empty → fall back to OBB center")
    grasp_pos = np.array([cream_center[0], cream_center[1], cream_pts[:, 2].max()])

# Force grasp z = object top (cream cheese top z; thin so we want fingertips at top)
# OBB tells us top is roughly cream_center[2] + ext[2]/2. Use pts max as ground truth.
top_z = float(cream_pts[:, 2].max())
# Fingertips at table-top level (object top): solve_ik places tips. To grasp the
# 1.8cm thick box, descend so tips are ~ at z = top_z - half-thickness ~ table+0.005.
# But since flat boxes typically have their flat top accessible, descending to the
# bottom (table) crashes the gripper. Use top_z (fingers will close around the box).
# Actually for a thin flat 1.8cm object, we want fingers to be at z roughly = top_z - 0.01
# so they hug the sides of the box.
grasp_z = top_z - 0.005
grasp_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_z])
print(f"[GRASP] final grasp_pos={grasp_pos.tolist()}, top_z={top_z:.3f}")

# === Step 5: Execute pick ===
open_gripper()
goto_pose(grasp_pos, quat, z_approach=0.15)
goto_pose(grasp_pos, quat)
close_gripper()

# Check grip width
obs2 = get_observation()
gw = obs2.get("robot_cartesian_pos", [0]*8)[7] if len(obs2.get("robot_cartesian_pos", [])) >= 8 else 0
print(f"[GRIP] width={gw:.3f}")

# === Step 6: Lift ===
lift_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.20])
joints = solve_ik(lift_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 7: Move above basket (high enough to clear rim) ===
# basket rim ~0.16; gripper carrying box must clear basket walls
# Move at z = max(lift_pos[2], basket_rim_z + 0.20)
transit_z = max(lift_pos[2], basket_rim_z + 0.20)
above_basket = np.array([basket_center[0], basket_center[1], transit_z])
joints = solve_ik(above_basket.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 8: Lower to release just above rim ===
# Release height: rim + clearance
# Cream cheese is 1.8cm thick. Want fingertips just above rim so box drops in.
# release_z = rim_z + obj_height + margin
obj_height = cream_pts[:, 2].max() - cream_pts[:, 2].min()
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
