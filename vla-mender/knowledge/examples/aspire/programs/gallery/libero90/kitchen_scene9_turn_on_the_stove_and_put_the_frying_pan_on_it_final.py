# KITCHEN_SCENE9 / turn_on_the_stove_and_put_the_frying_pan_on_it
# Two-step task: (1) turn the stove knob via top-down grasp + wrist yaw rotation;
# (2) pick up the frying pan by its handle and place body on the burner.
#
# Validation: 19/30 = 63.3% on seeds 51-80. Passes 4/5 on 51-55.
#
# Key design decisions:
# - Knob: single grasp at top_z-0.025, then yaw rotations [+60, +120, -60].
#   Adding more attempts caused the rotation to undo itself.
# - Pan: handle yaw via world-frame mask ptp (yaw=90 if hy_range > hx_range).
# - Pan placement: half-offset (gripper_xy = burner_xy - 0.5*grasp_to_body_xy),
#   then DROP from burner_top + 0.13 (NOT lower) so the pan settles naturally.
#   Lowering further before release caused the pan to slide off as the handle
#   protruded beyond the stove's back edge.
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0.0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def fresh_obs():
    obs = get_observation()
    cam = obs["agentview"]
    rgb = cam["images"]["rgb"]
    depth = cam["images"]["depth"]
    K = cam["intrinsics"]
    E = cam["pose_mat"]
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    return obs, rgb, depth_img, K, E


def localize_object(rgb, depth_img, K, E, prompts):
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        best = max(masks, key=lambda d: d["score"])
        mask = best["mask"].astype(np.uint8)
        pts = mask_to_world_points(mask, depth_img, K, E)
        if pts is None or len(pts) < 10:
            continue
        center = get_oriented_bounding_box_from_3d_points(pts)["center"]
        return center, pts, mask, best["score"]
    return None, None, None, 0.0


def localize_knob(rgb, depth_img, K, E):
    """Locate stove knob with stove-area filter."""
    for prompt in ["black stove knob", "stove knob", "black knob"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        cands = []
        for m in masks[:20]:
            mk = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mk, depth_img, K, E)
            if pts is None or len(pts) < 30:
                continue
            c = pts.mean(axis=0)
            # Knob is in front of stove: x∈[0.35,0.55], y∈[0.20,0.40], z>=0
            if not (0.35 < c[0] < 0.55 and 0.20 < c[1] < 0.40 and -0.02 < c[2] < 0.10):
                continue
            cands.append((m["score"], c, pts))
        if cands:
            cands.sort(key=lambda t: t[0], reverse=True)
            _, c, pts = cands[0]
            return c, pts
    return None, None


# ============ STEP 1: TURN ON STOVE KNOB ============
print("=== STEP 1: Turn on stove knob ===", flush=True)
_, rgb, depth_img, K, E = fresh_obs()

knob_center, knob_pts = localize_knob(rgb, depth_img, K, E)
if knob_center is None:
    knob_center = np.array([0.46, 0.30, 0.02])
    knob_pts = None

kx, ky = float(knob_center[0]), float(knob_center[1])
knob_top_z = float(np.percentile(knob_pts[:, 2], 95)) if knob_pts is not None else 0.05

# Grasp at z = knob_top - 0.025 (mid-knob, where its rotational handle is)
grasp_z = float(np.clip(knob_top_z - 0.025, 0.012, 0.040))
print(f"  knob ctr=({kx:.3f},{ky:.3f}) top_z={knob_top_z:.3f} grasp_z={grasp_z:.3f}", flush=True)

tdq0 = make_topdown_quat(0.0)
goto_home_joint_position()
open_gripper()

# 2-step descent
j = solve_ik(np.array([kx, ky, grasp_z + 0.10]), tdq0.tolist())
if j is not None:
    move_to_joints(j)
j = solve_ik(np.array([kx, ky, grasp_z]), tdq0.tolist())
if j is not None:
    move_to_joints(j)

close_gripper()
g_obs = get_observation()
grip = float(g_obs["robot_cartesian_pos"][7])
print(f"  knob grip = {grip:.3f}", flush=True)

# CW rotation via wrist yaw — three steps: +60, +120, -60 (net +120 CW)
for angle_deg in [60.0, 120.0, -60.0]:
    tdq_rot = make_topdown_quat(yaw_deg=angle_deg)
    j = solve_ik(np.array([kx, ky, grasp_z]), tdq_rot.tolist())
    if j is not None:
        move_to_joints(j)

open_gripper()

# Lift up and retreat home
j = solve_ik(np.array([kx, ky, grasp_z + 0.20]), tdq0.tolist())
if j is not None:
    move_to_joints(j)

goto_home_joint_position()
for _ in range(2):
    get_observation()


# ============ STEP 2: PLACE FRYING PAN ON STOVE ============
print("=== STEP 2: Place pan on stove ===", flush=True)

_, rgb2, depth_img2, K2, E2 = fresh_obs()

# Localize pan handle (for grasp) and pan body (for offset)
handle_center, handle_pts, _, h_score = localize_object(
    rgb2, depth_img2, K2, E2,
    ["frying pan handle", "pan handle"])
body_center, body_pts, _, b_score = localize_object(
    rgb2, depth_img2, K2, E2,
    ["pan body", "round pan", "frying pan"])

if handle_center is None and body_center is None:
    raise RuntimeError("Pan not found")
if body_center is None:
    body_center = handle_center.copy()
    body_pts = handle_pts

# Localize stove burner for placement target
burner_center, burner_pts, _, _ = localize_object(
    rgb2, depth_img2, K2, E2, ["stove burner"])
if burner_center is None:
    raise RuntimeError("Stove burner not found")

burner_top_z = float(np.percentile(burner_pts[:, 2], 90))
top_pts = burner_pts[burner_pts[:, 2] > burner_top_z - 0.01]
if len(top_pts) > 50:
    bx = float((np.percentile(top_pts[:, 0], 5) + np.percentile(top_pts[:, 0], 95)) / 2)
    by = float((np.percentile(top_pts[:, 1], 5) + np.percentile(top_pts[:, 1], 95)) / 2)
else:
    bx, by = float(burner_center[0]), float(burner_center[1])
print(f"  burner=({bx:.3f},{by:.3f},{burner_top_z:.3f})", flush=True)

# Determine handle yaw via world-frame ptp
if handle_pts is not None:
    hx_range = float(handle_pts[:, 0].ptp())
    hy_range = float(handle_pts[:, 1].ptp())
    yaw = 90.0 if hy_range > hx_range else 0.0
    h_top_z = float(handle_pts[:, 2].max())
    grasp_pos = np.array([handle_center[0], handle_center[1], h_top_z - 0.01])
else:
    yaw = 0.0
    grasp_pos = np.array([body_center[0], body_center[1], body_pts[:, 2].max() - 0.01])

# Half-offset: gripper holds handle, body lands at gripper + 0.5 * grasp_to_body
grasp_to_body_xy = np.array([float(body_center[0] - grasp_pos[0]),
                              float(body_center[1] - grasp_pos[1])])
print(f"  grasp_pos={grasp_pos} g2b={grasp_to_body_xy} yaw={yaw}", flush=True)

grasp_quat = make_topdown_quat(yaw)

# Pre-grasp, lower, close
open_gripper()
goto_pose(grasp_pos.tolist(), grasp_quat.tolist(), z_approach=0.15)
goto_pose(grasp_pos.tolist(), grasp_quat.tolist())
close_gripper()
for _ in range(2):
    get_observation()

# Lift well above the stove
lift_z = max(grasp_pos[2] + 0.30, burner_top_z + 0.30)
lift_pos = np.array([grasp_pos[0], grasp_pos[1], lift_z])
j = solve_ik(lift_pos.tolist(), grasp_quat.tolist())
if j is not None:
    move_to_joints(j)

# Half-offset placement target for gripper
gripper_target_xy = np.array([bx - 0.5 * grasp_to_body_xy[0],
                              by - 0.5 * grasp_to_body_xy[1]])
above = np.array([gripper_target_xy[0], gripper_target_xy[1], lift_z])
j = solve_ik(above.tolist(), grasp_quat.tolist())
if j is not None:
    move_to_joints(j)

# Slow descent to release height — KEY: drop FROM HIGHER (burner_top + 0.13), don't go lower.
# Lowering further causes the pan to slide off as the handle protrudes off the stove's back edge.
for z_target in [burner_top_z + 0.20, burner_top_z + 0.13]:
    pos = np.array([gripper_target_xy[0], gripper_target_xy[1], z_target])
    j = solve_ik(pos.tolist(), grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)

open_gripper()
for _ in range(3):
    get_observation()

# Retreat upward immediately to avoid disturbing the falling pan
retreat = np.array([gripper_target_xy[0], gripper_target_xy[1], burner_top_z + 0.40])
j = solve_ik(retreat.tolist(), grasp_quat.tolist())
if j is not None:
    move_to_joints(j)

# Move arm completely away from stove
home_retreat = np.array([0.4, -0.4, 0.6])
j = solve_ik(home_retreat.tolist(), grasp_quat.tolist())
if j is not None:
    move_to_joints(j)

# Long settle for success-predicate firing
for _ in range(15):
    get_observation()

# Verification
obs_v = get_observation()
cam_v = obs_v["agentview"]
rgb_v = cam_v["images"]["rgb"]
depth_v = cam_v["images"]["depth"]
depth_img_v = depth_v[:, :, 0] if len(depth_v.shape) == 3 else depth_v
K_v, E_v = cam_v["intrinsics"], cam_v["pose_mat"]

pan_v_c, _, _, _ = localize_object(rgb_v, depth_img_v, K_v, E_v,
    ["pan body", "round pan", "frying pan"])
if pan_v_c is not None:
    print(f"  [verify] pan body ctr=({pan_v_c[0]:.3f},{pan_v_c[1]:.3f},{pan_v_c[2]:.3f})", flush=True)
    print(f"  [verify] burner=({bx:.3f},{by:.3f},{burner_top_z:.3f}) "
          f"dist={np.hypot(pan_v_c[0]-bx, pan_v_c[1]-by):.3f}", flush=True)

print("=== Done ===", flush=True)
