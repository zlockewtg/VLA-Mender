import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


# CLOSE TOP DRAWER (KITCHEN_SCENE5)
# Adapted from KITCHEN_SCENE10 close-drawer code.
# KS5 difference: cabinet body is on the +Y side; drawer protrudes outward in
# the -Y direction (toward smaller Y). Drawer point cloud y in [0.08, 0.25],
# where y_min is the protruding (open) face and y_max is the back/cabinet side.
# Push direction is +Y (opposite of KS10 which pushed -Y).
# SAM3 workspace Y filter: (-0.20, 0.50) to capture +Y drawer.

print(f"Task: {env.handle.task_language}", flush=True)
obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K, E = cam["intrinsics"], cam["pose_mat"]

# Localize the open drawer.
masks = segment_sam3_text_prompt(rgb, "open drawer")
if not masks:
    raise RuntimeError("No 'open drawer' masks returned")

# Filter by workspace (reject phantom background masks).
# KS5: drawer in +Y region, y filter is (-0.20, 0.50).
drawer_pts = None
for m in masks[:5]:
    pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
    if pts is None or len(pts) < 100:
        continue
    c = pts.mean(0)
    if 0.40 < c[0] < 0.85 and -0.20 < c[1] < 0.50 and 0.05 < c[2] < 0.30:
        drawer_pts = pts
        print(f"Selected open-drawer mask score={m['score']:.3f} center={c.round(3)}", flush=True)
        break

if drawer_pts is None:
    for prompt in ["top drawer", "cabinet drawer", "drawer"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:5]:
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 100:
                continue
            c = pts.mean(0)
            if 0.40 < c[0] < 0.85 and -0.20 < c[1] < 0.50 and 0.05 < c[2] < 0.30:
                drawer_pts = pts
                print(f"Fallback prompt='{prompt}' mask score={m['score']:.3f} center={c.round(3)}", flush=True)
                break
        if drawer_pts is not None:
            break

if drawer_pts is None:
    raise RuntimeError("Could not localize the open top drawer")

y_max = float(np.percentile(drawer_pts[:, 1], 95))
y_min = float(np.percentile(drawer_pts[:, 1], 5))
x_c = float(np.percentile(drawer_pts[:, 0], 50))
z_top = float(np.percentile(drawer_pts[:, 2], 95))
z_bot = float(np.percentile(drawer_pts[:, 2], 5))
z_mid = (z_top + z_bot) / 2
drawer_length = y_max - y_min

print(f"Drawer x_c={x_c:.3f}, y=[{y_min:.3f},{y_max:.3f}], len={drawer_length:.3f}, z_mid={z_mid:.3f}", flush=True)

# KS5: cabinet at +Y, drawer protrudes to -Y. Open face is at y_min.
# Push direction is +Y (push from y_min-0.04 toward y_min + push_dist).
quat = make_topdown_quat(0)
close_gripper()

# Step 1 — pre-contact: position in front of drawer face (-Y side, y_min - 0.04)
contact_pos = [x_c, y_min - 0.04, z_mid]
joints = solve_ik(contact_pos, quat.tolist())
if joints is not None:
    move_to_joints(joints)
print(f"Contact at {contact_pos}", flush=True)

# Step 2 — push to close: travel +y by max(0.30, drawer_length+0.15)
push_dist = max(0.30, drawer_length + 0.15)
push_target_y = y_min + push_dist
push_pos = [x_c, push_target_y, z_mid]
joints = solve_ik(push_pos, quat.tolist())
if joints is not None:
    move_to_joints(joints)
print(f"Pushed to y={push_target_y:.3f}", flush=True)

# Step 3 — multi-pass deeper push to seat drawer at hard stop
for extra in [0.04, 0.08, 0.12]:
    p = [x_c, push_target_y + extra, z_mid]
    j = solve_ik(p, quat.tolist())
    if j is not None:
        move_to_joints(j)
print(f"Final push to y={push_target_y + 0.12:.3f}", flush=True)

# Step 4 — HOLD at closed position so success predicate samples stable closed state.
for _ in range(10):
    get_observation()

# Retreat: lift VERTICALLY first, then home.
joints = solve_ik([x_c, push_target_y + 0.12, z_mid + 0.20], quat.tolist())
if joints is not None:
    move_to_joints(joints)
goto_home_joint_position()
for _ in range(5):
    get_observation()
print("Done", flush=True)
