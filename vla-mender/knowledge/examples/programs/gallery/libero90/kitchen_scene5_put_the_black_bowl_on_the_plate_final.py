import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


def select_best_candidate(rgb, depth_img, K, E, prompts, ext_filter, label="object"):
    cands = []
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:6]:
            mk = m["mask"].astype(np.uint8)
            if mk.sum() < 100:
                continue
            pts = mask_to_world_points(mk, depth_img, K, E)
            if pts is None or len(pts) < 50:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            center, extent = obb["center"], obb["extent"]
            if not ext_filter(center, extent):
                continue
            cands.append((m["score"], center, pts, mk, prompt, extent))
    if not cands:
        return None, None, None
    cands.sort(key=lambda c: -c[0])
    sc, c, p, mk, pr, ex = cands[0]
    print(f"[{label}] picked prompt='{pr}' score={sc:.3f} center={c.round(3)} ext={ex.round(3)}", flush=True)
    return c, p, mk


# ---------------- Settle physics ----------------
goto_home_joint_position()
for _ in range(3):
    open_gripper()
    close_gripper()
open_gripper()
for _ in range(2):
    get_observation()

obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if depth.ndim == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]

# ---------------- Localize bowl ----------------
def bowl_filter(center, extent):
    if center[0] < 0.3 or center[0] > 1.0:
        return False
    if center[2] > 0.15 or center[2] < -0.05:
        return False
    if extent[2] < 0.025:
        return False
    if max(extent[0], extent[1]) > 0.20:
        return False
    if max(extent[0], extent[1]) < 0.04:
        return False
    return True


bowl_center, bowl_pts, bowl_mask = select_best_candidate(
    rgb, depth_img, K, E,
    ["small bowl", "bowl", "dark bowl", "black bowl"],
    bowl_filter,
    label="bowl",
)
if bowl_center is None:
    raise RuntimeError("Black bowl not found")

bowl_obb = get_oriented_bounding_box_from_3d_points(bowl_pts)
bowl_top_z = float(bowl_pts[:, 2].max())
bowl_bottom_z = float(bowl_pts[:, 2].min())
bowl_height = bowl_top_z - bowl_bottom_z
bowl_radius = max(float(bowl_obb["extent"][0]), float(bowl_obb["extent"][1])) / 2.0

# ---------------- Localize plate ----------------
def plate_filter(center, extent):
    if center[0] < 0.3 or center[0] > 1.0:
        return False
    if center[2] > 0.10 or center[2] < -0.05:
        return False
    if extent[2] >= 0.025:
        return False
    if min(extent[0], extent[1]) < 0.10:
        return False
    return True


tgt_center, tgt_pts, _ = select_best_candidate(
    rgb, depth_img, K, E,
    ["plate", "dinner plate", "white plate"],
    plate_filter,
    label="plate",
)
if tgt_center is None:
    raise RuntimeError("Plate not found")

surface_z = float(tgt_pts[:, 2].max())

print(f"bowl_center={bowl_center.round(3)} top_z={bowl_top_z:.3f} h={bowl_height:.3f} r={bowl_radius:.3f}", flush=True)
print(f"plate_center={tgt_center.round(3)} surface_z={surface_z:.3f}", flush=True)

# ---------------- Grasp pose: bowl center + topdown ----------------
# Strategy: descend with finger tips inside the bowl rim region.
# Use yaw=90 to avoid cabinet collision constraint.
grasp_pos = np.array([bowl_center[0], bowl_center[1], bowl_top_z - 0.020])
grasp_quat = make_topdown_quat(90)

print(f"[grasp] final pos={grasp_pos.round(3)} quat={grasp_quat.round(3)}", flush=True)


def step_to(target_pos, quat, n_steps=6):
    """Interpolated descent — avoids goto_pose's max_steps convergence failure."""
    obs_now = get_observation()
    current = np.array(obs_now['robot_cartesian_pos'][:3])
    print(f"[step_to] start ee_pos={current.round(3)} -> target={np.array(target_pos).round(3)}", flush=True)
    for k in range(1, n_steps + 1):
        wp = current + (np.array(target_pos) - current) * (k / n_steps)
        j = solve_ik(wp.tolist(), quat.tolist())
        if j is not None:
            move_to_joints(j)
            obs_now = get_observation()
            actual = np.array(obs_now['robot_cartesian_pos'][:3])
            print(f"[step_to] step {k} target={wp.round(3)} actual={actual.round(3)}", flush=True)
        else:
            print(f"[step_to] step {k} solve_ik returned None for {wp.round(3)}", flush=True)


# ---------------- Pick ----------------
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.15)
obs_pre = get_observation()
print(f"[grasp] after pre-grasp ee_pos={np.array(obs_pre['robot_cartesian_pos'][:3]).round(3)}", flush=True)
# Use step_to instead of plain goto_pose for the descent
step_to(grasp_pos, grasp_quat, n_steps=4)
close_gripper()

obs2 = get_observation()
grip_w = float(obs2["robot_cartesian_pos"][7]) if len(obs2["robot_cartesian_pos"]) > 7 else -1
ee_pos = obs2["robot_cartesian_pos"][:3]
print(f"[grasp] grip_width={grip_w:.3f} ee_pos={np.array(ee_pos).round(3)}", flush=True)

# ---------------- Lift ----------------
lift_z = grasp_pos[2] + 0.20
lift_pos = np.array([grasp_pos[0], grasp_pos[1], lift_z])
joints = solve_ik(lift_pos.tolist(), grasp_quat.tolist())
if joints is not None:
    move_to_joints(joints)

# After lift, re-localize bowl to detect offset from gripper
obs_lift = get_observation()
ee_lifted = np.array(obs_lift['robot_cartesian_pos'][:3])
rgb_l = obs_lift["agentview"]["images"]["rgb"]
depth_l = obs_lift["agentview"]["images"]["depth"]
depth_img_l = depth_l[:, :, 0] if depth_l.ndim == 3 else depth_l
K_l = obs_lift["agentview"]["intrinsics"]
E_l = obs_lift["agentview"]["pose_mat"]
bowl_lift_center, _, _ = select_best_candidate(
    rgb_l, depth_img_l, K_l, E_l,
    ["small bowl", "bowl", "dark bowl", "black bowl"],
    bowl_filter,
    label="bowl_lifted",
)
bowl_offset = np.array([0.0, 0.0])
if bowl_lift_center is not None:
    # Bowl xy relative to gripper xy: bowl - gripper
    bowl_offset = bowl_lift_center[:2] - ee_lifted[:2]
    print(f"[lift] bowl_lifted_xy={bowl_lift_center[:2].round(3)} ee_xy={ee_lifted[:2].round(3)} offset={bowl_offset.round(3)}", flush=True)

# ---------------- Move above plate ----------------
# Compensate for bowl-pendant offset: when the wide-bowl is grasped at rim
# with yaw=90 (gripper closes along world-x), the bowl tends to swing toward
# +x and -y from the gripper after the asymmetric rim-grip.
# Empirical compensation: gripper xy = plate_center + (0.027, -0.017)
# so bowl lands centered.
COMPENSATE = np.array([0.030, -0.020])
target_xy = tgt_center[:2] + COMPENSATE
above = np.array([target_xy[0], target_xy[1], lift_z])
joints = solve_ik(above.tolist(), grasp_quat.tolist())
if joints is not None:
    move_to_joints(joints)

# ---------------- Lower to release ----------------
release_z = surface_z + bowl_height - 0.005
release_pos = np.array([target_xy[0], target_xy[1], release_z])
print(f"[place] release_pos={release_pos.round(3)} (target_xy compensated by {COMPENSATE})", flush=True)
# Step down in 6 increments for very gentle placement
step_to(release_pos, grasp_quat, n_steps=6)

obs_pre_release = get_observation()
print(f"[place] pre-release ee_pos={np.array(obs_pre_release['robot_cartesian_pos'][:3]).round(3)}", flush=True)
open_gripper()
open_gripper()

# Settle BEFORE retreat
for _ in range(10):
    get_observation()

# ---------------- Retreat ----------------
retreat_pos = np.array([tgt_center[0], tgt_center[1], release_z + 0.18])
joints = solve_ik(retreat_pos.tolist(), grasp_quat.tolist())
if joints is not None:
    move_to_joints(joints)

for _ in range(15):
    get_observation()
