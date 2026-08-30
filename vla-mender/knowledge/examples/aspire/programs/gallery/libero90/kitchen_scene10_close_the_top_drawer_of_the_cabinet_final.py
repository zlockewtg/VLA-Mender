import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


# CLOSE TOP DRAWER (KITCHEN_SCENE10)
# Initial state: top drawer is open, protruding in +y direction toward camera.
# We push from the drawer's +y face in the -y direction with closed gripper as flat pusher.

print(f"Task: {env.handle.task_language}", flush=True)
obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K, E = cam["intrinsics"], cam["pose_mat"]

# Localize the open drawer. "open drawer" gives the protruding top drawer with high score.
masks = segment_sam3_text_prompt(rgb, "open drawer")
if not masks:
    raise RuntimeError("No 'open drawer' masks returned")

# Filter by workspace (reject phantom background masks at x<0).
drawer_pts = None
for m in masks[:5]:
    pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
    if pts is None or len(pts) < 100:
        continue
    c = pts.mean(0)
    if 0.40 < c[0] < 0.85 and -0.40 < c[1] < 0.20 and 0.05 < c[2] < 0.30:
        drawer_pts = pts
        print(f"Selected open-drawer mask score={m['score']:.3f} center={c.round(3)}", flush=True)
        break

if drawer_pts is None:
    # Fallback: try "top drawer" or "cabinet drawer" with same workspace filter.
    for prompt in ["top drawer", "cabinet drawer", "drawer"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:5]:
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 100:
                continue
            c = pts.mean(0)
            if 0.40 < c[0] < 0.85 and -0.40 < c[1] < 0.20 and 0.05 < c[2] < 0.30:
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

# Push-to-close: closed gripper as flat pusher.
quat = make_topdown_quat(0)
close_gripper()

# Step 1 — pre-contact: position behind drawer face (+y_max + 0.04)
contact_pos = [x_c, y_max + 0.04, z_mid]
joints = solve_ik(contact_pos, quat.tolist())
if joints is not None:
    move_to_joints(joints)
print(f"Contact at {contact_pos}", flush=True)

# Step 2 — push to close: travel -y by max(0.30, drawer_length+0.15)
# Push deep to guarantee the drawer joint reaches its hard stop.
push_dist = max(0.30, drawer_length + 0.15)
push_target_y = y_max - push_dist
push_pos = [x_c, push_target_y, z_mid]
joints = solve_ik(push_pos, quat.tolist())
if joints is not None:
    move_to_joints(joints)
print(f"Pushed to y={push_target_y:.3f}", flush=True)

# Step 3 — multi-pass slow push to ensure drawer doesn't bounce / wedge
for extra in [0.04, 0.08, 0.12]:
    p = [x_c, push_target_y - extra, z_mid]
    j = solve_ik(p, quat.tolist())
    if j is not None:
        move_to_joints(j)
print(f"Final push to y={push_target_y - 0.12:.3f}", flush=True)

# Step 4 — HOLD at closed position for several physics steps so the success
# predicate samples the drawer in its closed state (prevents bounce-back).
for _ in range(10):
    get_observation()

# Retreat: lift VERTICALLY first (don't drag the drawer back open), then home.
joints = solve_ik([x_c, push_target_y - 0.12, z_mid + 0.20], quat.tolist())
if joints is not None:
    move_to_joints(joints)
goto_home_joint_position()
for _ in range(5):
    get_observation()
print("Done", flush=True)
