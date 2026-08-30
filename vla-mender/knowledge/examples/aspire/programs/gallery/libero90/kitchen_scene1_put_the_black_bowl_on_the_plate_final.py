import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


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
        return center, pts, mask
    return None, None, None


def select_bowl_mask(rgb, depth_img, K, E):
    candidates = []
    for prompt in ("small bowl", "bowl", "black bowl", "dark bowl"):
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:6]:
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 50:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c, e = obb["center"], obb["extent"]
            if c[0] < 0.3 or c[0] > 1.0:
                continue
            if c[2] > 0.15:
                continue
            ext_xy = max(e[0], e[1])
            if ext_xy > 0.18 or ext_xy < 0.04:
                continue
            if e[2] < 0.025 or e[2] > 0.10:
                continue
            candidates.append((m["score"], c, pts, mask))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, center, pts, mask = candidates[0]
    return center, pts, mask


def select_plate_mask(rgb, depth_img, K, E):
    candidates = []
    for prompt in ("round plate", "dinner plate", "plate", "white plate"):
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:6]:
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 80:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c, e = obb["center"], obb["extent"]
            if c[0] < 0.3 or c[0] > 1.0:
                continue
            if c[2] > 0.04:
                continue
            if e[2] > 0.03:
                continue
            ext_xy = max(e[0], e[1])
            if ext_xy < 0.10 or ext_xy > 0.22:
                continue
            candidates.append((m["score"], c, pts, mask))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, center, pts, mask = candidates[0]
    return center, pts, mask


def step_to(target_pos, quat, n_steps=4):
    obs = get_observation()
    current = np.array(obs['robot_cartesian_pos'][:3])
    for k in range(1, n_steps + 1):
        wp = current + (target_pos - current) * (k / n_steps)
        j = solve_ik(wp.tolist(), quat.tolist())
        if j is not None:
            move_to_joints(j)


def ik_multi_pass(target_pos, quat, n=4):
    for _ in range(n):
        j = solve_ik(target_pos.tolist(), quat.tolist())
        if j is not None:
            move_to_joints(j)


def attempt_grasp(grasp_xy, grasp_z, quat):
    """Approach + descend + close. Returns gripper width."""
    open_gripper()
    pre = np.array([grasp_xy[0], grasp_xy[1], grasp_z + 0.15])
    j = solve_ik(pre.tolist(), quat.tolist())
    if j is not None:
        move_to_joints(j)
    target = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
    ik_multi_pass(target, quat, n=4)
    close_gripper()
    obs_g = get_observation()
    cart = obs_g['robot_cartesian_pos']
    return cart[7] if len(cart) > 7 else 0.0, cart[:3]


# ---------------- Main ----------------
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]

bowl_center, bowl_pts, bowl_mask = select_bowl_mask(rgb, depth_img, K, E)
if bowl_center is None:
    bowl_center, bowl_pts, bowl_mask = localize_object(
        rgb, depth_img, K, E, ["small bowl", "bowl", "black bowl", "dark bowl"])
if bowl_center is None:
    raise RuntimeError("Black bowl not found")

bowl_obb = get_oriented_bounding_box_from_3d_points(bowl_pts)
bowl_top_z = float(bowl_pts[:, 2].max())
bowl_bottom_z = float(bowl_pts[:, 2].min())
bowl_height = bowl_top_z - bowl_bottom_z
bowl_radius = max(float(bowl_obb["extent"][0]), float(bowl_obb["extent"][1])) / 2.0

plate_center, plate_pts, plate_mask = select_plate_mask(rgb, depth_img, K, E)
if plate_center is None:
    plate_center, plate_pts, plate_mask = localize_object(
        rgb, depth_img, K, E,
        ["round plate", "dinner plate", "plate", "white plate"])
if plate_center is None:
    raise RuntimeError("Plate not found")

plate_surface_z = float(plate_pts[:, 2].max())
print(f"[task] bowl_center={bowl_center} bowl_top_z={bowl_top_z:.3f} bowl_h={bowl_height:.3f} bowl_r={bowl_radius:.3f} "
      f"plate_center={plate_center} plate_surface_z={plate_surface_z:.3f}", flush=True)

# ---------------- Grasp the bowl ----------------
grasp_quat = make_topdown_quat(0)

# Strategy: try center-grasp at top-005 first; if gw<0.05, try rim grasps.
# For wide bowl (>10cm), the fingers may go INSIDE and close on air. Rim grasp catches the rim.
bowl_xy = np.array([bowl_center[0], bowl_center[1]])

# Primary: center grasp at top - 0.005 (rim level)
target_z = bowl_top_z - 0.005
gw, end_cart = attempt_grasp(bowl_xy, target_z, grasp_quat)
print(f"[task] center grasp gw={gw:.3f} cart_z={end_cart[2]:.3f}", flush=True)
final_grasp_xy = bowl_xy.copy()

# If gw too narrow (closed on air inside bowl), try rim grasp
if gw < 0.05:
    print(f"[task] center grasp failed (gw={gw:.3f}); trying rim grasps", flush=True)
    open_gripper()
    # Move up first
    above = np.array([bowl_xy[0], bowl_xy[1], bowl_top_z + 0.15])
    j = solve_ik(above.tolist(), grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)

    # Try 4 rim positions (forced rim grasps from skill)
    r_rim = 0.95 * bowl_radius
    rim_positions = [
        (bowl_xy[0], bowl_xy[1] + r_rim),
        (bowl_xy[0], bowl_xy[1] - r_rim),
        (bowl_xy[0] + r_rim, bowl_xy[1]),
        (bowl_xy[0] - r_rim, bowl_xy[1]),
    ]
    best_gw = 0.0
    best_xy = bowl_xy.copy()
    for rx, ry in rim_positions:
        rim_xy = np.array([rx, ry])
        gw_r, _ = attempt_grasp(rim_xy, target_z, grasp_quat)
        print(f"[task] rim ({rx:.3f},{ry:.3f}) gw={gw_r:.3f}", flush=True)
        if gw_r > best_gw:
            best_gw = gw_r
            best_xy = rim_xy.copy()
        if gw_r >= 0.04:
            # Good grip — keep this one
            final_grasp_xy = rim_xy
            gw = gw_r
            break
        open_gripper()
        above = np.array([rim_xy[0], rim_xy[1], bowl_top_z + 0.15])
        j = solve_ik(above.tolist(), grasp_quat.tolist())
        if j is not None:
            move_to_joints(j)
    else:
        # No rim grasp succeeded — use best
        if best_gw > gw:
            final_grasp_xy = best_xy
            # Re-grasp at best position
            gw, _ = attempt_grasp(best_xy, target_z, grasp_quat)

print(f"[task] final grasp xy={final_grasp_xy} gw={gw:.3f}", flush=True)

# ---------------- Lift, transit, place ----------------
# Track grasp offset for placement compensation
grasp_offset_xy = final_grasp_xy - bowl_xy  # offset from bowl center

lift_z = max(target_z + 0.20, plate_surface_z + 0.22)

lift_pos = np.array([final_grasp_xy[0], final_grasp_xy[1], lift_z])
ik_multi_pass(lift_pos, grasp_quat, n=2)

# Aim gripper at plate_center + grasp_offset (so bowl center lands at plate center)
target_gripper_xy = np.array([plate_center[0], plate_center[1]]) + grasp_offset_xy
above = np.array([target_gripper_xy[0], target_gripper_xy[1], lift_z])
ik_multi_pass(above, grasp_quat, n=3)

# Verify transit XY accuracy
obs_t = get_observation()
cart_t = obs_t['robot_cartesian_pos']
drift = np.linalg.norm(np.array(cart_t[:2]) - target_gripper_xy)
print(f"[task] transit cart=({cart_t[0]:.3f},{cart_t[1]:.3f}) target=({target_gripper_xy[0]:.3f},{target_gripper_xy[1]:.3f}) drift={drift:.3f}", flush=True)
if drift > 0.015:
    dx = target_gripper_xy[0] - cart_t[0]
    dy = target_gripper_xy[1] - cart_t[1]
    corrected = np.array([target_gripper_xy[0] + dx, target_gripper_xy[1] + dy, lift_z])
    print(f"[task] XY correction -> {corrected[:2]}", flush=True)
    ik_multi_pass(corrected, grasp_quat, n=3)

# Lower to release height
release_z = plate_surface_z + bowl_height + 0.005
release_pos = np.array([target_gripper_xy[0], target_gripper_xy[1], release_z])
step_to(release_pos, grasp_quat, n_steps=4)

obs_r = get_observation()
cart_r = obs_r['robot_cartesian_pos']
print(f"[task] release cart=({cart_r[0]:.3f},{cart_r[1]:.3f},{cart_r[2]:.3f})", flush=True)

open_gripper()

for _ in range(3):
    get_observation()

# Retreat upward
retreat_pos = np.array([cart_r[0], cart_r[1], plate_surface_z + 0.25])
joints = solve_ik(retreat_pos.tolist(), grasp_quat.tolist())
if joints is not None:
    move_to_joints(joints)

for _ in range(3):
    get_observation()
