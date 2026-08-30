"""
LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate
Task type: pick-and-place with relative spatial placement.

Pick: chocolate pudding box (small flat dark/brown rectangular box, ~8cm x 5cm x 3cm).
Place: to the LEFT of the dinner plate (white plate, ~13.5cm dia, 1.1cm thick).

Placement convention (LIVING_ROOM camera): world +y axis = image-right.
"Left of plate" means image-left → world -y. So:
Target = plate_center + [0, -offset_y, 0] where
   offset_y = plate_radius_y + pudding_half_y + small_gap.

Mirrors the right-of-plate task (28/30); only the Y offset sign is flipped.

KEY FINDING: For this scene, single goto_pose() with z_approach=0.15 doesn't reach
the requested final pose because move_to_joints_blocking has limited interpolation
(120 sim-steps).  Solve by chunking motion into incremental waypoints (3-4 steps).

SAM3 prompts (verified seed 51):
  pudding: "chocolate pudding box" (0.93), "small box" (0.91), "brown box" (0.85)
  plate:   "dinner plate" (0.93), "white plate" (0.92), "plate" (0.91)
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def step_to(target_pos, quat, n_steps=4):
    """Move incrementally to target via interpolated waypoints.
    Required because move_to_joints_blocking has limited steps and
    can't traverse large pose differences in one call."""
    obs = get_observation()
    current = np.array(obs['robot_cartesian_pos'][:3])
    for k in range(1, n_steps + 1):
        wp = current + (target_pos - current) * (k / n_steps)
        j = solve_ik(wp.tolist(), quat.tolist())
        move_to_joints(j)


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

# === Step 2: Localize chocolate pudding (flat thin box) ===
pudding_prompts = ["chocolate pudding box", "small box", "brown box", "dark box",
                   "rectangular box", "pudding box", "chocolate pudding"]
pudding = find_object(rgb, depth_img, K, E, pudding_prompts,
                      ext_z_max=0.045, ext_xy_min=0.04, ext_xy_max=0.13,
                      pos_z_max=0.07)
if pudding is None:
    raise RuntimeError("Chocolate pudding not found")
pud_center = pudding["center"]
pud_pts = pudding["pts"]
pud_mask = pudding["mask"]
pud_ext = pudding["ext"]
print(f"[PUDDING] prompt='{pudding['prompt']}' score={pudding['score']:.3f}")
print(f"          center={pud_center.tolist()} ext={pud_ext.tolist()}")

# === Step 3: Localize plate ===
plate_prompts = ["dinner plate", "white plate", "plate", "round plate"]
plate = find_object(rgb, depth_img, K, E, plate_prompts,
                    ext_xy_min=0.10, ext_z_max=0.04, pos_z_max=0.10)
if plate is None:
    raise RuntimeError("Plate not found")
plate_center = plate["center"]
plate_pts = plate["pts"]
plate_ext = plate["ext"]
print(f"[PLATE] prompt='{plate['prompt']}' score={plate['score']:.3f}")
print(f"        center={plate_center.tolist()} ext={plate_ext.tolist()}")

# === Step 4: Compute target (LEFT of plate) ===
# In LIBERO living-room scene: world +y axis = image-right.
# So "left of plate" = plate_y - offset (negative offset).
plate_half_y = float(plate_ext[1] / 2)
pud_half_y = float(pud_ext[1] / 2)
gap = 0.020
target_x = float(plate_center[0])
target_y = float(plate_center[1]) - plate_half_y - pud_half_y - gap
# Use table z (just below plate top) for placement
table_z = float(np.percentile(plate_pts[:, 2], 5))
target_center = np.array([target_x, target_y, table_z])
print(f"[TARGET] xy=({target_x:.3f}, {target_y:.3f}) z={table_z:.3f}")

# === Step 5: Pick the pudding ===
quat = make_topdown_quat(0)

# Use OBB center / median for robust XY (no plan_grasp needed; tight box has clear top-down grasp)
pud_xy = np.array([float(pud_center[0]), float(pud_center[1])])

top_z = float(pud_pts[:, 2].max())
grasp_z = top_z - 0.005  # tips just below box top → fingers wrap sides

# Pre-grasp ~10cm above
pre_pos = np.array([pud_xy[0], pud_xy[1], top_z + 0.10])
grasp_pos = np.array([pud_xy[0], pud_xy[1], grasp_z])

print(f"[PICK] pre={pre_pos.tolist()} grasp={grasp_pos.tolist()}")
open_gripper()
step_to(pre_pos, quat, n_steps=4)
step_to(grasp_pos, quat, n_steps=3)
close_gripper()

obs2 = get_observation()
gw = float(obs2['robot_cartesian_pos'][7])
print(f"[GRIP] width={gw:.3f}")
if gw < 0.05:
    print("[WARN] empty grasp; aborting")

# === Step 6: Lift ===
lift_pos = np.array([pud_xy[0], pud_xy[1], grasp_z + 0.20])
step_to(lift_pos, quat, n_steps=3)
obs2 = get_observation()
print(f"[LIFT] pos={obs2['robot_cartesian_pos'][:3]} width={obs2['robot_cartesian_pos'][7]:.3f}")

# === Step 7: Move laterally above target ===
above_target = np.array([target_center[0], target_center[1], lift_pos[2]])
step_to(above_target, quat, n_steps=4)
obs2 = get_observation()
print(f"[ABOVE_TGT] pos={obs2['robot_cartesian_pos'][:3]}")

# === Step 8: Lower to release ===
# Place pudding on table next to plate. Box bottom should land on table.
# pud_height ~3cm; surface_z = table_z. solve_ik places finger TIPS (gripping near box top).
# Release at tips_z = surface_z + pud_height + 0.010 so pad-bottom is just above table.
pud_height = float(pud_pts[:, 2].max() - pud_pts[:, 2].min())
release_z = table_z + pud_height + 0.010
release_pos = np.array([target_center[0], target_center[1], release_z])
print(f"[RELEASE] z={release_z:.3f} (table_z={table_z:.3f}, pud_h={pud_height:.3f})")
step_to(release_pos, quat, n_steps=3)

# === Step 9: Release ===
open_gripper()

# Settle
for _ in range(3):
    get_observation()

# === Step 10: Retreat upward ===
retreat = np.array([release_pos[0], release_pos[1], release_pos[2] + 0.20])
step_to(retreat, quat, n_steps=3)

# Settle physics
for _ in range(5):
    get_observation()
