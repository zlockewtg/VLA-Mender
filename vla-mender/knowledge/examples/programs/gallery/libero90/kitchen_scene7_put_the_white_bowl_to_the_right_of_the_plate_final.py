"""
KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate

Task: pick a white bowl (sitting on top of an appliance/cabinet at z~0.22)
and place it to the RIGHT of a plate (on the table at z~0).

Spatial convention (KITCHEN scene, agentview camera):
  Camera at (~1.32, 0, 0.70) looking along -x.
  World +y = image-right.  "right of plate" = plate_y + offset (positive y).

Geometry (verified seed 51):
  Bowl: center=(0.630, -0.264, 0.220), ext=(0.080, 0.082, 0.036) — round, ~8cm dia, ~4cm tall, on appliance top.
  Plate: center=(0.671, -0.004, 0.001), ext=(0.133, 0.135, 0.011) — flat round, ~13.5cm dia.

SAM3 prompts:
  Bowl: "small bowl" (0.90), "white bowl" (0.67), "bowl" (0.67) — but each may also return the plate as a low-score candidate.
  Plate: "round plate" (0.94), "plate" (0.86), "dinner plate" (0.83).

Both bowl and plate appear in many prompts. Use geometric filters:
  Plate: ext_xy_min > 0.10, ext_z < 0.04, pos_z < 0.10
  Bowl:  ext_xy_min in [0.05, 0.10], ext_z in [0.02, 0.07] (bowl is taller than plate)

Strategy:
  1. Settle physics with gripper toggles.
  2. Localize plate (high-confidence first via "round plate").
  3. Localize bowl (filter shape so we don't pick the plate).
  4. Compute target_xy = plate + [0, plate_radius_y + bowl_radius_y + gap, 0].
  5. Top-down grasp at bowl_top_z - 0.010, yaw=0 (bowl is round, yaw doesn't matter).
  6. Lift +0.20 from grasp_z (bowl was high already at z=0.22 on appliance).
  7. Lateral move above target_xy at lift z.
  8. Lower to table_z + bowl_height + 0.005.
  9. Open gripper, retreat +0.20.

Use step_to() for all motions because goto_pose() doesn't reliably descend.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def step_to(target_pos, quat, n_steps=4):
    """Interpolated Cartesian descent. Use instead of goto_pose for >5cm moves."""
    obs_loc = get_observation()
    current = np.array(obs_loc['robot_cartesian_pos'][:3])
    for k in range(1, n_steps + 1):
        wp = current + (target_pos - current) * (k / n_steps)
        j = solve_ik(wp.tolist(), quat.tolist())
        if j is not None:
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


# === Step 1: Settle physics ===
goto_home_joint_position()
for _ in range(3):
    open_gripper()
    close_gripper()
open_gripper()

# === Step 2: Observe scene ===
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K, E = cam["intrinsics"], cam["pose_mat"]

# === Step 3: Localize plate ===
# Plate is flat (ext_z < 0.04), large (ext_xy > 0.10), low (pos_z < 0.10).
plate_prompts = ["round plate", "plate", "dinner plate", "white plate", "ceramic plate"]
plate = find_object(rgb, depth_img, K, E, plate_prompts,
                    ext_xy_min=0.10, ext_z_max=0.04, pos_z_max=0.10, min_score=0.30)
if plate is None:
    raise RuntimeError("Plate not found")
plate_center = plate["center"]
plate_pts = plate["pts"]
plate_ext = plate["ext"]
print(f"[PLATE] prompt='{plate['prompt']}' score={plate['score']:.3f}")
print(f"        center={plate_center.tolist()} ext={plate_ext.tolist()}")

# === Step 4: Localize white bowl ===
# Bowl is small (ext_xy 0.05-0.10), taller than plate (ext_z 0.02-0.07),
# and could be elevated on the appliance (pos_z up to 0.30).
bowl_prompts = ["small bowl", "white bowl", "bowl", "ceramic bowl", "round bowl"]
bowl = find_object(rgb, depth_img, K, E, bowl_prompts,
                   ext_xy_min=0.05, ext_xy_max=0.12, ext_z_min=0.02, ext_z_max=0.07,
                   pos_z_max=0.40, min_score=0.30)
if bowl is None:
    raise RuntimeError("White bowl not found")
bowl_center = bowl["center"]
bowl_pts = bowl["pts"]
bowl_mask = bowl["mask"]
bowl_ext = bowl["ext"]
print(f"[BOWL] prompt='{bowl['prompt']}' score={bowl['score']:.3f}")
print(f"       center={bowl_center.tolist()} ext={bowl_ext.tolist()}")

# === Step 5: Compute target_xy (right of plate) ===
plate_half_y = float(plate_ext[1] / 2)
bowl_half_y = float(bowl_ext[1] / 2)
gap = 0.030  # 3cm gap so bowl is clearly to the right (away from plate edge)
target_x = float(plate_center[0])
target_y = float(plate_center[1]) + plate_half_y + bowl_half_y + gap
# Use plate-top z as table reference (plate sits on table; plate top ~ table_z + 0.011)
table_z = float(np.percentile(plate_pts[:, 2], 5))  # underside of plate ~ table top
print(f"[TARGET] xy=({target_x:.3f}, {target_y:.3f}) table_z={table_z:.3f}")

# === Step 6: Pick bowl ===
quat = make_topdown_quat(0)

# Use bowl rim XY centroid for grasp (robust to depth elongation).
# The bowl rim is the topmost ~1cm of points; rim is symmetric so its mean = true center.
bowl_top_z_now = float(bowl_pts[:, 2].max())
rim_pts = bowl_pts[bowl_pts[:, 2] > bowl_top_z_now - 0.012]
if len(rim_pts) >= 20:
    # Use median of x-range and y-range to find geometric center robust to outliers
    rim_xy_mid = np.array([
        0.5 * (np.percentile(rim_pts[:, 0], 5) + np.percentile(rim_pts[:, 0], 95)),
        0.5 * (np.percentile(rim_pts[:, 1], 5) + np.percentile(rim_pts[:, 1], 95)),
    ])
    bowl_xy = rim_xy_mid
    print(f"[RIM_XY] using rim midpoint xy=({bowl_xy[0]:.3f},{bowl_xy[1]:.3f}) n_rim={len(rim_pts)}")
else:
    bowl_xy = np.array([float(bowl_center[0]), float(bowl_center[1])])
    print(f"[RIM_XY] fallback to OBB center xy=({bowl_xy[0]:.3f},{bowl_xy[1]:.3f})")

bowl_top_z = float(bowl_pts[:, 2].max())
bowl_bot_z = float(bowl_pts[:, 2].min())
bowl_height = bowl_top_z - bowl_bot_z
grasp_z = bowl_top_z - 0.010  # tips just below rim → fingers wrap bowl wall
print(f"[BOWL_GEO] top_z={bowl_top_z:.3f} bot_z={bowl_bot_z:.3f} height={bowl_height:.3f} grasp_z={grasp_z:.3f}")

# Pre-grasp ~12cm above the bowl
pre_pos = np.array([bowl_xy[0], bowl_xy[1], bowl_top_z + 0.12])
grasp_pos = np.array([bowl_xy[0], bowl_xy[1], grasp_z])

print(f"[PICK] pre={pre_pos.tolist()} grasp={grasp_pos.tolist()}")
step_to(pre_pos, quat, n_steps=4)
step_to(grasp_pos, quat, n_steps=3)
close_gripper()

obs2 = get_observation()
gw = float(obs2['robot_cartesian_pos'][7])
print(f"[GRIP] width={gw:.3f}")

# === Step 7: Lift ===
# Bowl was at z=0.22 (on appliance). Need lift well above appliance to clear it.
# Appliance top is at ~0.22. Lift to grasp_z + 0.20 = ~0.42.
lift_z = max(grasp_z + 0.20, 0.40)  # ensure clearance over appliance
lift_pos = np.array([bowl_xy[0], bowl_xy[1], lift_z])
step_to(lift_pos, quat, n_steps=3)
obs2 = get_observation()
print(f"[LIFT] pos={obs2['robot_cartesian_pos'][:3]} width={obs2['robot_cartesian_pos'][7]:.3f}")

# === Step 8: Move laterally above target ===
above_target = np.array([target_x, target_y, lift_z])
step_to(above_target, quat, n_steps=4)
obs2 = get_observation()
print(f"[ABOVE_TGT] pos={obs2['robot_cartesian_pos'][:3]}")

# === Step 9: Lower to release ===
# solve_ik places finger TIPS. Bowl is gripped near top (grasp_z = top - 0.010),
# so when finger tips are at z, bowl bottom is at z - (bowl_height - 0.010).
# We want bowl bottom = table_z + 0.005 → tips_z = table_z + 0.005 + (bowl_height - 0.010)
release_offset = bowl_height - 0.010  # how high gripper tips are above bowl bottom at grasp
release_z = table_z + 0.005 + release_offset
release_pos = np.array([target_x, target_y, release_z])
print(f"[RELEASE] z={release_z:.3f} (table_z={table_z:.3f}, bowl_h={bowl_height:.3f}, offset={release_offset:.3f})")
step_to(release_pos, quat, n_steps=3)

# === Step 10: Release ===
open_gripper()
for _ in range(3):
    get_observation()

# === Step 11: Retreat upward ===
retreat = np.array([release_pos[0], release_pos[1], release_pos[2] + 0.20])
step_to(retreat, quat, n_steps=3)
for _ in range(5):
    get_observation()
