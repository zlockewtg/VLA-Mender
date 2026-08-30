import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


# Run the full task
def localize_object(rgb, depth, K, E, prompts):
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks: continue
        m = max(masks, key=lambda d: d["score"])
        pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
        if pts is None or len(pts) < 10: continue
        return get_oriented_bounding_box_from_3d_points(pts)["center"], pts, m["mask"]
    return None, None, None


# CLOSE DRAWER
obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, "open drawer")
m = masks[0]
pts = mask_to_world_points(m["mask"].astype(np.uint8), depth, K, E)
y_max = np.percentile(pts[:, 1], 95); y_min = np.percentile(pts[:, 1], 5)
x_c = np.percentile(pts[:, 0], 50)
z_top = np.percentile(pts[:, 2], 95); z_mid = (z_top + np.percentile(pts[:, 2], 5)) / 2
quat = make_topdown_quat(0)
close_gripper()
joints = solve_ik([x_c, y_max + 0.04, z_mid], quat.tolist())
if joints is not None: move_to_joints(joints)
joints = solve_ik([x_c, y_max - max(0.25, y_max - y_min + 0.10), z_mid], quat.tolist())
if joints is not None: move_to_joints(joints)
joints = solve_ik([x_c, y_max - max(0.25, y_max - y_min + 0.10) - 0.05, z_mid], quat.tolist())
if joints is not None: move_to_joints(joints)
goto_home_joint_position()
for _ in range(3):
    obs = get_observation()

# PICK BOWL AND PLACE
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]

bowl_c, bowl_pts, bowl_mask = localize_object(rgb, depth, K, E, ["small bowl", "black bowl"])
cab_c, cab_pts, _ = localize_object(rgb, depth, K, E, ["cabinet top surface", "cabinet top"])

print(f"Initial bowl center: {bowl_c.round(3)}", flush=True)
print(f"Cab top center: {cab_c.round(3)}", flush=True)

# Pick bowl
bowl_top_z = np.percentile(bowl_pts[:, 2], 95)
g_quat = make_topdown_quat(90)
g_pos = np.array([bowl_c[0], bowl_c[1], bowl_top_z - 0.01])
open_gripper()
goto_pose(g_pos, g_quat, z_approach=0.15)
goto_pose(g_pos, g_quat)
close_gripper()

# Lift, then check bowl position via observation
obs2 = get_observation()
cam2 = obs2["agentview"]
rgb2 = cam2["images"]["rgb"]; depth2 = cam2["images"]["depth"]
K2 = cam2["intrinsics"]; E2 = cam2["pose_mat"]

# Look for bowl now
masks = segment_sam3_text_prompt(rgb2, "small bowl")
if masks:
    m = max(masks, key=lambda d: d["score"])
    pts2 = mask_to_world_points(m["mask"].astype(np.uint8), depth2, K2, E2)
    if pts2 is not None:
        c2 = get_oriented_bounding_box_from_3d_points(pts2)["center"]
        print(f"Bowl AFTER grasp (in gripper): {c2.round(3)}", flush=True)

# Lift up
lift_z = max(g_pos[2] + 0.25, 0.45)
joints = solve_ik([g_pos[0], g_pos[1], lift_z], g_quat.tolist())
if joints is not None: move_to_joints(joints)

# Move above cabinet center
cab_x_lo, cab_x_hi = np.percentile(cab_pts[:, 0], [5, 95])
cab_y_lo, cab_y_hi = np.percentile(cab_pts[:, 1], [5, 95])
target_x = (cab_x_lo + cab_x_hi) / 2
target_y = (cab_y_lo + cab_y_hi) / 2
print(f"Target: ({target_x:.3f}, {target_y:.3f})", flush=True)

joints = solve_ik([target_x, target_y, lift_z], g_quat.tolist())
if joints is not None: move_to_joints(joints)

# Lower and release
surface_z = cab_pts[:, 2].max()
joints = solve_ik([target_x, target_y, surface_z + 0.04], g_quat.tolist())
if joints is not None: move_to_joints(joints)

# Check bowl pos before release
obs3 = get_observation()
cam3 = obs3["agentview"]
rgb3 = cam3["images"]["rgb"]; depth3 = cam3["images"]["depth"]
K3 = cam3["intrinsics"]; E3 = cam3["pose_mat"]
masks = segment_sam3_text_prompt(rgb3, "small bowl")
if masks:
    m = max(masks, key=lambda d: d["score"])
    pts3 = mask_to_world_points(m["mask"].astype(np.uint8), depth3, K3, E3)
    if pts3 is not None:
        c3 = get_oriented_bounding_box_from_3d_points(pts3)["center"]
        print(f"Bowl BEFORE release (above cabinet): {c3.round(3)}", flush=True)

open_gripper()
for _ in range(5):
    get_observation()

obs4 = get_observation()
cam4 = obs4["agentview"]
rgb4 = cam4["images"]["rgb"]; depth4 = cam4["images"]["depth"]
K4 = cam4["intrinsics"]; E4 = cam4["pose_mat"]
masks = segment_sam3_text_prompt(rgb4, "small bowl")
if masks:
    m = max(masks, key=lambda d: d["score"])
    pts4 = mask_to_world_points(m["mask"].astype(np.uint8), depth4, K4, E4)
    if pts4 is not None:
        c4 = get_oriented_bounding_box_from_3d_points(pts4)["center"]
        print(f"Bowl AFTER release (settled): {c4.round(3)}", flush=True)

# Retreat
joints = solve_ik([target_x, target_y, surface_z + 0.20], g_quat.tolist())
if joints is not None: move_to_joints(joints)

for _ in range(3):
    get_observation()

obs5 = get_observation()
cam5 = obs5["agentview"]
rgb5 = cam5["images"]["rgb"]; depth5 = cam5["images"]["depth"]
K5 = cam5["intrinsics"]; E5 = cam5["pose_mat"]
masks = segment_sam3_text_prompt(rgb5, "small bowl")
if masks:
    m = max(masks, key=lambda d: d["score"])
    pts5 = mask_to_world_points(m["mask"].astype(np.uint8), depth5, K5, E5)
    if pts5 is not None:
        c5 = get_oriented_bounding_box_from_3d_points(pts5)["center"]
        print(f"Bowl FINAL: {c5.round(3)}", flush=True)

# Where's the cabinet top now?
masks = segment_sam3_text_prompt(rgb5, "cabinet top")
if masks:
    m = max(masks, key=lambda d: d["score"])
    pts_c = mask_to_world_points(m["mask"].astype(np.uint8), depth5, K5, E5)
    if pts_c is not None:
        c = get_oriented_bounding_box_from_3d_points(pts_c)["center"]
        x_lo, x_hi = np.percentile(pts_c[:, 0], [5, 95])
        y_lo, y_hi = np.percentile(pts_c[:, 1], [5, 95])
        z_lo, z_hi = np.percentile(pts_c[:, 2], [5, 95])
        print(f"Cab top FINAL: ctr={c.round(3)} x=[{x_lo:.3f},{x_hi:.3f}] y=[{y_lo:.3f},{y_hi:.3f}] z=[{z_lo:.3f},{z_hi:.3f}]", flush=True)
