# Code block 0
import numpy as np

###############################################################################
# Helper functions
###############################################################################

def get_cam(obs):
    cam = obs["robot0_robotview"]
    return cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]


def move_safe(pos, quat):
    joints = solve_ik(np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64))
    move_to_joints(joints)


def pixel_to_3d_safe(u, v, depth, K, E):
    u, v = int(u), int(v)
    z = float(depth[v, u])
    if np.isfinite(z) and z > 0:
        return pixel_to_world_point(u, v, z, K, E)
    for r in range(1, 15):
        for vv in range(max(0, v - r), min(depth.shape[0], v + r + 1)):
            for uu in range(max(0, u - r), min(depth.shape[1], u + r + 1)):
                zz = float(depth[vv, uu])
                if np.isfinite(zz) and zz > 0:
                    return pixel_to_world_point(uu, vv, zz, K, E)
    raise RuntimeError(f"No valid depth near ({u},{v})")


def find_peg(obs):
    """Find square peg. Must have z_max > 0 (protrudes above table)."""
    rgb, depth, K, E = get_cam(obs)
    for prompt in ["brown square peg", "square dowel", "brown square post"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d.get("score", 0.0), reverse=True)[:10]:
            area = m["mask"].sum()
            if area > 1500 or area < 50:
                continue
            pts = mask_to_world_points(m["mask"], depth, K, E)
            if len(pts) < 5:
                continue
            z_max = pts[:, 2].max()
            if z_max > 0.0:
                obb = get_oriented_bounding_box_from_3d_points(pts)
                return obb['center'], z_max
    # Molmo fallback
    for prompt in ["small square peg on table", "square peg"]:
        try:
            result = point_prompt_molmo(rgb, prompt)
            uv = list(result.values())[0]
            if uv[0] is not None:
                wp = pixel_to_3d_safe(int(uv[0]), int(uv[1]), depth, K, E)
                if wp[2] > 0.0:
                    return wp, wp[2]
        except:
            continue
    raise RuntimeError("Cannot find peg")


###############################################################################
# Main execution
###############################################################################

print("=== NUT ASSEMBLY ===", flush=True)

obs = get_observation()
rgb, depth, K, E = get_cam(obs)

# 1. Find peg
peg_center, peg_top_z = find_peg(obs)
print(f"Peg: center={peg_center}, top_z={peg_top_z:.4f}", flush=True)

# 2. Segment nut, get OBB for geometry
nut_masks = segment_sam3_text_prompt(rgb, "brown square nut")
best_nut = max(nut_masks, key=lambda d: d.get("score", 0.0))
nut_pts = mask_to_world_points(best_nut["mask"], depth, K, E)
nut_obb = get_oriented_bounding_box_from_3d_points(nut_pts)
nut_z_surface = nut_pts[:, 2].max()

# 3. Get hole center via Molmo
hole_result = point_prompt_molmo(rgb, "white hollow center of the brown square nut")
hole_uv = list(hole_result.values())[0]
hole_world = pixel_to_3d_safe(int(hole_uv[0]), int(hole_uv[1]), depth, K, E)
print(f"Hole center: {hole_world}", flush=True)

# 4. Determine handle axis from OBB
# The nut+handle longest axis is the handle direction
nut_R = nut_obb['R']
nut_extent = nut_obb['extent']
axis0_2d = nut_R[:2, 0]
axis1_2d = nut_R[:2, 1]
proj0 = nut_extent[0] * np.linalg.norm(axis0_2d)
proj1 = nut_extent[1] * np.linalg.norm(axis1_2d)
if proj0 >= proj1:
    handle_axis = axis0_2d / np.linalg.norm(axis0_2d)
    handle_ext = nut_extent[0]
else:
    handle_axis = axis1_2d / np.linalg.norm(axis1_2d)
    handle_ext = nut_extent[1]

# Find which end of axis is the handle (farther from hole)
obb_center = nut_obb['center']
end1 = obb_center[:2] + handle_axis * (handle_ext / 2)
end2 = obb_center[:2] - handle_axis * (handle_ext / 2)
if np.linalg.norm(end1 - hole_world[:2]) > np.linalg.norm(end2 - hole_world[:2]):
    handle_end = end1
    handle_dir = handle_axis
else:
    handle_end = end2
    handle_dir = -handle_axis

# 5. Grasp position: 15mm from handle end toward body
grasp_xy = handle_end - handle_dir * 0.015
grasp_z = nut_z_surface - 0.015
grasp_pos = np.array([grasp_xy[0], grasp_xy[1], grasp_z])

# 6. Gripper orientation: y-axis along handle, z-axis down
z_grip = np.array([0, 0, -1], dtype=np.float64)
y_grip = np.array([handle_dir[0], handle_dir[1], 0], dtype=np.float64)
x_grip = np.cross(y_grip, z_grip)
x_grip = x_grip / np.linalg.norm(x_grip)
R_grip = np.column_stack([x_grip, y_grip, z_grip])
grip_quat = rotation_matrix_to_quaternion(R_grip)

# 7. Offset from grasp to hole center
offset = hole_world - grasp_pos
print(f"Grasp: {grasp_pos}, offset->hole: {offset}", flush=True)

# 8. Grasp
open_gripper()
pre = grasp_pos.copy()
pre[2] = nut_z_surface + 0.10
move_safe(pre, grip_quat)
move_safe(grasp_pos, grip_quat)
close_gripper()

# 9. Lift
lift_z = nut_z_surface + 0.16
move_safe(np.array([grasp_pos[0], grasp_pos[1], lift_z]), grip_quat)

# 10. Re-find peg
obs2 = get_observation()
try:
    peg2, peg_top2 = find_peg(obs2)
except:
    peg2, peg_top2 = peg_center, peg_top_z

# 11. Align: move hole center above peg center
target_x = peg2[0] - offset[0]
target_y = peg2[1] - offset[1]
move_safe(np.array([target_x, target_y, lift_z]), grip_quat)
print(f"Above peg: ({target_x:.4f}, {target_y:.4f})", flush=True)

# 12. Insert: descend slowly, go deep
# peg_top_z is ~0.028. The hole center should be at peg_top2 level.
# EE z = peg_top2 - offset[2]
target_z = peg_top2 - offset[2]
# Go deeper to ensure full insertion
for dz in [0.06, 0.04, 0.02, 0.01, 0.0, -0.01, -0.02, -0.03,
           -0.04, -0.05, -0.06, -0.07, -0.08]:
    p = np.array([target_x, target_y, target_z + dz])
    try:
        move_safe(p, grip_quat)
    except:
        break

# 13. Release
open_gripper()

# 14. Retreat
move_safe(np.array([target_x, target_y, peg_top2 + 0.20]), grip_quat)

print("=== DONE ===", flush=True)