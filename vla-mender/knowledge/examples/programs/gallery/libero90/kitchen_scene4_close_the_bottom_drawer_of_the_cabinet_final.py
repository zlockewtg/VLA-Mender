import numpy as np
from scipy.spatial.transform import Rotation

def make_topdown_quat(yaw_deg=0.0):
    R = (Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix()
         @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K, E = cam["intrinsics"], cam["pose_mat"]
print(f"Task: {env.handle.task_language}", flush=True)


def localize_bottom_drawer(rgb, depth_img, K, E):
    """Find the bottom drawer face using SAM3 segmentation."""
    candidates = []
    for prompt in ["open drawer", "wooden drawer", "bottom drawer", "drawer"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:5]:
            if m['score'] < 0.3:
                continue
            pts = mask_to_world_points(m['mask'].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 200:
                continue
            c = pts.mean(0)
            zmax = pts[:, 2].max()
            if 0.3 < c[0] < 0.9 and -0.5 < c[1] < 0.5 and 0.02 < zmax < 0.40:
                candidates.append((c[2], pts, m['score'], prompt))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda x: x[0])
    best_z, best_pts, best_score, best_prompt = candidates[0]
    return best_pts, best_score, best_prompt


def find_bowl_centered_under_descent(rgb, depth_img, K, E, push_x, descent_y):
    """Find a tall bowl/bottle whose center is close to descent column.
    Returns the bowl with smallest |center_x - push_x| if any are within 4cm.
    """
    blockers = []
    for prompt in ["bowl", "metal bowl"]:
        try:
            masks = segment_sam3_text_prompt(rgb, prompt)
        except Exception:
            masks = []
        for m in masks[:3]:
            if m['score'] < 0.30:
                continue
            pts = mask_to_world_points(m['mask'].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 50:
                continue
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            cz = float(pts[:, 2].mean())
            x_min = float(pts[:, 0].min())
            x_max = float(pts[:, 0].max())
            y_min = float(pts[:, 1].min())
            y_max = float(pts[:, 1].max())
            z_max = float(pts[:, 2].max())
            # The bowl must be in the descent region (Y near descent_y)
            if y_min - 0.05 <= descent_y <= y_max + 0.05:
                blockers.append({
                    "prompt": prompt, "score": m["score"],
                    "cx": cx, "cy": cy, "cz": cz,
                    "x_min": x_min, "x_max": x_max,
                    "y_min": y_min, "y_max": y_max, "z_max": z_max,
                })
    return blockers


drawer_pts, _, _ = localize_bottom_drawer(rgb, depth_img, K, E)
if drawer_pts is None:
    drawer_x_min, drawer_x_max = 0.60, 0.75
    push_x = 0.677
    push_y_face = 0.043
else:
    drawer_x_min = float(drawer_pts[:, 0].min())
    drawer_x_max = float(drawer_pts[:, 0].max())
    push_x = float(np.mean(drawer_pts[:, 0]))
    push_y_face = float(drawer_pts[:, 1].min())
    print(f"Drawer: x={push_x:.3f}, x_extent=[{drawer_x_min:.3f},{drawer_x_max:.3f}], face_y={push_y_face:.3f}", flush=True)

DEFAULT_START_Y = -0.05

# Detect bowl directly in descent column. If push_x is within ~4cm of bowl
# center, shift push_x to hit bowl EDGE instead of CENTER (so arm clips
# the bowl and slides past, rather than getting trapped).
blockers = find_bowl_centered_under_descent(rgb, depth_img, K, E, push_x, DEFAULT_START_Y)
for b in blockers:
    print(f"Bowl @ cx={b['cx']:.3f} (push_x={push_x:.3f}, dx={b['cx']-push_x:.3f})", flush=True)

CRITICAL_DX = 0.010  # closer than 1cm = bowl "centered" under descent (rare edge case)
shift_applied = False
for b in blockers:
    if abs(b['cx'] - push_x) < CRITICAL_DX:
        # Need to shift push_x to ±4cm from bowl center
        # Try right side first (toward higher X) - bottle usually on left
        shift_right = b['cx'] + 0.045
        shift_left = b['cx'] - 0.045
        if drawer_x_min + 0.005 <= shift_right <= drawer_x_max - 0.005:
            push_x = shift_right
            shift_applied = True
            print(f"Push X shifted RIGHT to {push_x:.3f} (4.5cm from bowl edge)", flush=True)
            break
        elif drawer_x_min + 0.005 <= shift_left <= drawer_x_max - 0.005:
            push_x = shift_left
            shift_applied = True
            print(f"Push X shifted LEFT to {push_x:.3f} (4.5cm from bowl edge)", flush=True)
            break

if not shift_applied:
    print(f"No shift needed; push_x={push_x:.3f}", flush=True)

start_y = DEFAULT_START_Y
push_quat = make_topdown_quat(yaw_deg=90.0)
close_gripper()
print("Starting multi-pass push strategy", flush=True)

# MULTI-PASS PUSH STRATEGY (original — proven 14/15 baseline)
push_targets = [0.00, 0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25]
z_values = [0.020, 0.030, 0.040, 0.045, 0.048, 0.049, 0.050, 0.055]

done_break = False
for test_z in z_values:
    if done_break:
        break
    try:
        j = solve_ik([push_x, start_y, 0.30], push_quat.tolist())
        if j is not None:
            move_to_joints(j)
    except Exception:
        done_break = True
        break

    try:
        j = solve_ik([push_x, start_y, test_z], push_quat.tolist())
    except Exception:
        done_break = True
        break
    if j is None:
        print(f"  z={test_z:.3f}: start IK failed, skipping", flush=True)
        continue
    try:
        move_to_joints(j)
    except Exception:
        done_break = True
        break

    for ty in push_targets:
        try:
            j2 = solve_ik([push_x, ty, test_z], push_quat.tolist())
            if j2 is not None:
                move_to_joints(j2)
        except Exception:
            done_break = True
            break
    if done_break:
        break

    try:
        obs_e = get_observation()
        pos_e = obs_e.get("robot_cartesian_pos")
        if pos_e is not None:
            print(f"  z={test_z:.3f}: wrist_y={pos_e[1]:.4f}, wrist_z={pos_e[2]:.4f}", flush=True)
    except Exception:
        done_break = True
        break

# Final hold and escape
try:
    j = solve_ik([push_x, 0.25, 0.065], push_quat.tolist())
    if j is not None:
        move_to_joints(j)
    j = solve_ik([push_x, 0.20, 0.40], push_quat.tolist())
    if j is not None:
        move_to_joints(j)
    goto_home_joint_position()
    for _ in range(3):
        get_observation()
except Exception as e:
    print(f"Escape ended: {e}", flush=True)
print("Done", flush=True)
