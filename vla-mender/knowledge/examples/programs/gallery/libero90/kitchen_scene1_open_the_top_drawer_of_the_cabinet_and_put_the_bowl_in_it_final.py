"""
KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it
Strategy:
1. Use quat_h (horizontal gripper, z=-y) to grasp drawer handle from front.
2. Approach from +y side, advance until IK saturates near handle.
3. Close gripper, pull in +y direction to open drawer.
4. Re-localize bowl, pick top-down.
5. Place bowl into open drawer interior.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def safe_ik(pos, quat):
    j = solve_ik(np.asarray(pos, dtype=float).tolist(), np.asarray(quat, dtype=float).tolist())
    if j is not None:
        move_to_joints(j)
        return True
    return False


def step_to(target_pos, quat, n_steps=4):
    obs = get_observation()
    current = np.array(obs['robot_cartesian_pos'][:3])
    for k in range(1, n_steps + 1):
        wp = current + (np.asarray(target_pos) - current) * (k / n_steps)
        j = solve_ik(wp.tolist(), np.asarray(quat).tolist())
        if j is not None:
            move_to_joints(j)


def localize_object(rgb, depth, K, E, prompts, x_min=None, x_max=None,
                   y_min=None, y_max=None, z_min=None, z_max=None,
                   ext_max=None, score_min=0.0):
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for mm in masks[:10]:
            if mm["score"] < score_min: continue
            m = mm["mask"].astype(np.uint8)
            pts = mask_to_world_points(m, depth, K, E)
            if pts is None or len(pts) < 10:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c = obb["center"]; ex = obb["extent"]
            if x_min is not None and c[0] < x_min: continue
            if x_max is not None and c[0] > x_max: continue
            if y_min is not None and c[1] < y_min: continue
            if y_max is not None and c[1] > y_max: continue
            if z_min is not None and c[2] < z_min: continue
            if z_max is not None and c[2] > z_max: continue
            if ext_max is not None and max(ex[:2]) > ext_max: continue
            return c, pts, m
    return None, None, None


def find_top_drawer_handle(rgb, depth, K, E):
    """Find the top drawer handle on the on-table cabinet."""
    candidates = []
    for prompt in ["drawer handle", "handle", "drawer pull", "metal handle", "silver handle", "knob"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for i, mm in enumerate(masks[:30]):
            m = mm["mask"].astype(np.uint8)
            pts = mask_to_world_points(m, depth, K, E)
            if pts is None or len(pts) < 5:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c = obb["center"]
            ex = obb["extent"]
            # Cabinet top handle: x in 0.5-0.85, y in -0.30 to 0.20, z in 0.13-0.22
            if (0.5 < c[0] < 0.85 and -0.30 < c[1] < 0.20
                    and 0.13 < c[2] < 0.22
                    and max(ex[:2]) < 0.10 and ex[2] < 0.05):
                candidates.append((c, pts, m, ex, mm["score"]))
    if not candidates:
        return None, None, None
    # Dedupe by 3D center
    deduped = []
    for cand in candidates:
        if not any(np.linalg.norm(cand[0] - prev[0]) < 0.03 for prev in deduped):
            deduped.append(cand)
    # Pick the topmost (highest z) compact one
    compact = [c for c in deduped if max(c[3][:2]) < 0.10]
    if compact:
        compact.sort(key=lambda c: -c[0][2])
        return compact[0][0], compact[0][1], compact[0][2]
    deduped.sort(key=lambda c: -c[0][2])
    return deduped[0][0], deduped[0][1], deduped[0][2]


# ============================================================
# Step 0: Settle physics
# ============================================================
goto_home_joint_position()
for _ in range(3):
    open_gripper()
    close_gripper()
open_gripper()
for _ in range(3):
    get_observation()


# ============================================================
# Step 1: Localize handle (initial position)
# ============================================================
obs = get_observation()
rgb = obs["agentview"]["images"]["rgb"]
depth_raw = obs["agentview"]["images"]["depth"]
depth = depth_raw[:, :, 0] if depth_raw.ndim == 3 else depth_raw
K = obs["agentview"]["intrinsics"]
E = obs["agentview"]["pose_mat"]

handle_center, handle_pts, _ = find_top_drawer_handle(rgb, depth, K, E)
if handle_center is None:
    raise RuntimeError("Could not find top drawer handle")
print(f"[HANDLE] initial center={handle_center.round(3)}")
hx, hy, hz = float(handle_center[0]), float(handle_center[1]), float(handle_center[2])

# Front-bar y position (more accurate target for grasping)
# The bar tip is at y_max of handle pts
bar_y_front = float(np.percentile(handle_pts[:, 1], 5))   # bar back (closest to face)
bar_y_back = float(np.percentile(handle_pts[:, 1], 95))   # bar front
print(f"[HANDLE] bar y range: {bar_y_front:.3f} to {bar_y_back:.3f}")

# Localize cabinet body NOW (before drawer is open) for x-center reference
cab_center_init, cab_pts_init, _ = localize_object(rgb, depth, K, E,
    ["dark wooden cabinet", "wooden box", "wooden block"],
    x_min=0.4, x_max=0.95, y_min=-0.6, y_max=0.05)
if cab_pts_init is not None:
    cab_pts_filt = cab_pts_init[(cab_pts_init[:, 0] > 0.4) & (cab_pts_init[:, 0] < 0.95)
                           & (cab_pts_init[:, 1] > -0.6) & (cab_pts_init[:, 1] < 0.05)]
    cab_x_center = float((np.percentile(cab_pts_filt[:, 0], 5) + np.percentile(cab_pts_filt[:, 0], 95)) / 2)
    cab_face_y_init = float(np.percentile(cab_pts_filt[:, 1], 95))
    print(f"[CABINET] x_center={cab_x_center:.3f} face_y={cab_face_y_init:.3f}")
else:
    cab_x_center = hx  # fallback
    cab_face_y_init = -0.214
    print(f"[CABINET] using fallback x_center={cab_x_center:.3f}")


# ============================================================
# Step 2: Open the top drawer with quat_h approach
# ============================================================
# quat_h: gripper z-axis = -world_y (points toward cabinet face)
# fingers spread along world_x
quat_h = np.array([0.707, 0.707, 0, 0])

# Pre-pose: ~10cm in front of handle (in +y direction since cabinet is at -y)
print("[OPEN] Approaching handle with quat_h...")
open_gripper()
pre_y = hy + 0.20  # 20cm in front
pre = [hx, pre_y, hz]
safe_ik(pre, quat_h)
obs = get_observation()
print(f"  pre achieved={np.array(obs['robot_cartesian_pos'][:3]).round(3)}")

# Step in toward handle (fewer waypoints to save steps)
for dy in [+0.10, +0.03, -0.04]:
    pos = [hx, hy + dy, hz]
    j = solve_ik(pos, quat_h.tolist())
    if j is not None:
        move_to_joints(j)

# Close gripper (grip handle bar)
close_gripper()
obs = get_observation()
grip_w0 = float(obs['robot_cartesian_pos'][7])
print(f"[OPEN] after close grip_w={grip_w0:.3f}")

# Pull in +y direction (away from cabinet) — drag drawer
print("[OPEN] Pulling drawer...")
pull_target_y = hy + 0.30  # 30cm pull
n_pull = 6
for k in range(1, n_pull + 1):
    target_y = -0.10 + (pull_target_y - (-0.10)) * (k / n_pull)
    pos = [hx, target_y, hz]
    j = solve_ik(pos, quat_h.tolist())
    if j is not None:
        move_to_joints(j)

# Release & retreat
open_gripper()
for _ in range(2): get_observation()

# Retreat: move arm up and away from drawer
retreat = [hx, 0.20, hz + 0.20]
safe_ik(retreat, quat_h)


# ============================================================
# Step 3: Verify drawer state and find bowl
# ============================================================
goto_home_joint_position()
for _ in range(3):
    get_observation()

obs = get_observation()
rgb = obs["agentview"]["images"]["rgb"]
depth_raw = obs["agentview"]["images"]["depth"]
depth = depth_raw[:, :, 0] if depth_raw.ndim == 3 else depth_raw
K = obs["agentview"]["intrinsics"]
E = obs["agentview"]["pose_mat"]

# Verify drawer opened
handle_after, handle_after_pts, _ = find_top_drawer_handle(rgb, depth, K, E)
if handle_after is not None:
    drawer_pull_y = float(handle_after[1] - hy)  # how far drawer moved in +y
    print(f"[VERIFY] handle now at {handle_after.round(3)}, drawer pulled by {drawer_pull_y:.3f}m")
else:
    drawer_pull_y = 0.0
    print(f"[VERIFY] no handle found after pull")


# ============================================================
# Step 4: Find bowl
# ============================================================
def find_bowl(rgb, depth, K, E):
    candidates = []
    for prompt in ["small bowl", "bowl", "metal bowl", "silver bowl", "round bowl"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks: continue
        for mm in masks[:8]:
            m = mm["mask"].astype(np.uint8)
            pts = mask_to_world_points(m, depth, K, E)
            if pts is None or len(pts) < 20: continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c = obb["center"]; ex = obb["extent"]
            if (c[2] < 0.10 and 0.4 < c[0] < 0.95
                    and -0.10 < c[1] < 0.30
                    and 0.06 < max(ex[:2]) < 0.20):
                candidates.append((c, pts, m, mm["score"]))
                break
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda c: -c[3])
    return candidates[0][0], candidates[0][1], candidates[0][2]

bowl_center, bowl_pts, bowl_mask = find_bowl(rgb, depth, K, E)
if bowl_center is None:
    raise RuntimeError("Could not find bowl after drawer open")
bowl_obb = get_oriented_bounding_box_from_3d_points(bowl_pts)
bowl_top_z = float(bowl_pts[:, 2].max())
bowl_height = bowl_top_z - float(bowl_pts[:, 2].min())
bowl_x = float(bowl_obb["center"][0])
bowl_y = float(bowl_obb["center"][1])
print(f"[BOWL] center=({bowl_x:.3f},{bowl_y:.3f}), top_z={bowl_top_z:.3f}, height={bowl_height:.3f}, ext={bowl_obb['extent'].round(3)}")


# ============================================================
# Step 5: Grasp bowl (yaw=0, side-pinch via IK fallback)
# ============================================================
# Discovered: yaw=0 with grasp_z = bowl_top - 0.020 produces a side-pinch
# fallback (IK can't reach top-down at low z, falls to tilted side approach)
# that grips the bowl rim. yaw=90 just closes on bowl interior contents.
bowl_top_p95 = float(np.percentile(bowl_pts[:, 2], 95))
bowl_top_max = float(bowl_pts[:, 2].max())

def try_plangrasp_attempt(z_offset_from_top, n_iters=6):
    """Use plan_grasp orientation, but pin z to bowl_top + z_offset, snap XY to OBB center.
    Returns (gw, grasp_pos, grasp_quat)."""
    gp_b, gs_b = plan_grasp(depth, K, bowl_mask)
    if gp_b is None or len(gp_b) == 0:
        return 0.0, None, None
    best_b, _ = select_top_down_grasp(gp_b, gs_b, E)
    if best_b is None:
        idx_b = np.argmax(gs_b)
        best_b = E @ gp_b[idx_b]
    pos_b, q_b = decompose_transform(best_b)
    pos_b = np.asarray(pos_b)
    q_b = np.asarray(q_b)
    pos_b[0] = bowl_x
    pos_b[1] = bowl_y
    pos_b[2] = bowl_top_max + z_offset_from_top

    open_gripper()
    goto_home_joint_position()
    for _ in range(3): get_observation()
    pre_b = pos_b.copy()
    pre_b[2] = max(pre_b[2] + 0.25, 0.30)
    j = solve_ik(pre_b.tolist(), q_b.tolist())
    if j is not None: move_to_joints(j)
    for _ in range(2): get_observation()
    for _ in range(n_iters):
        j = solve_ik(pos_b.tolist(), q_b.tolist())
        if j is not None: move_to_joints(j)
        get_observation()
    close_gripper()
    for _ in range(3): get_observation()
    obs = get_observation()
    gw_v = float(obs['robot_cartesian_pos'][7])
    return gw_v, pos_b, q_b

# Attempt 1: plan_grasp orientation + bowl_top - 0.020 (just below rim)
print(f"[BOWL] attempt1 plan_grasp + z=top-0.020")
gw_bowl, pos_b, quat_b = try_plangrasp_attempt(-0.020)
obs = get_observation()
print(f"[BOWL] attempt1 grip_w={gw_bowl:.3f}")

# Lift to high z (use top-down quat for transit - more stable)
quat_lift = make_topdown_quat(0)
lift_pos = np.array([bowl_x, bowl_y, 0.45])
j = solve_ik(lift_pos.tolist(), quat_lift.tolist())
if j is not None: move_to_joints(j)
obs = get_observation()
gw_lift = float(obs['robot_cartesian_pos'][7])
print(f"[BOWL] after lift grip_w={gw_lift:.3f}")

# Verify bowl lifted using SAM3 (reliable signal)
rgb_chk = obs["agentview"]["images"]["rgb"]
depth_chk = obs["agentview"]["images"]["depth"]
d_chk = depth_chk[:, :, 0] if depth_chk.ndim == 3 else depth_chk
masks_chk = segment_sam3_text_prompt(rgb_chk, "small bowl")
bowl_z_after = None
if masks_chk:
    m_chk = max(masks_chk, key=lambda d: d["score"])
    pts_chk = mask_to_world_points(m_chk["mask"].astype(np.uint8), d_chk, K, E)
    if pts_chk is not None and len(pts_chk) > 10:
        c_chk = get_oriented_bounding_box_from_3d_points(pts_chk)["center"]
        bowl_z_after = float(c_chk[2])
print(f"[BOWL] bowl_z after lift1: {bowl_z_after}")

# Retry if not lifted (only one retry to fit in step budget)
if bowl_z_after is None or bowl_z_after < 0.15:
    print(f"[BOWL] not lifted, retry attempt 2 z=top-0.025")
    open_gripper()
    goto_home_joint_position()
    for _ in range(3): get_observation()
    gw_bowl, pos_b, quat_b = try_plangrasp_attempt(-0.025, n_iters=6)
    print(f"[BOWL] attempt2 grip_w={gw_bowl:.3f}")
    j = solve_ik(lift_pos.tolist(), quat_lift.tolist())
    if j is not None: move_to_joints(j)
    obs = get_observation()
    gw_lift = float(obs['robot_cartesian_pos'][7])
    print(f"[BOWL] retry after lift grip_w={gw_lift:.3f}")

grasp_z = bowl_top_max - 0.020  # for downstream


# ============================================================
# Step 6: Place into open drawer
# ============================================================
# Drawer interior: between cabinet face (y=-0.214) and handle position (now further +y)
# Drawer interior y center: midpoint between handle_after_y and original cabinet face y_back
# (since drawer body slides out preserving handle on front)
# Drop x = drawer center x ≈ (cabinet_x_min + cabinet_x_max)/2 ≈ 0.50
# Or just use handle x

# Use INITIAL cabinet measurements (saved before drawer opened)
if handle_after is not None and drawer_pull_y > 0.05:
    # Drawer is open. Drop at drawer interior center.
    handle_y_open = float(handle_after[1])
    drop_x = cab_x_center  # cabinet x-center (saved before open)
    # Interior y center = midpoint between cabinet face and pulled-out handle position
    drop_y = (cab_face_y_init + handle_y_open) / 2.0
else:
    drop_x = cab_x_center
    drop_y = cab_face_y_init - 0.05
    print(f"[PLACE] WARNING drawer may not be open, trying anyway")
print(f"[PLACE] using cab_x_center={cab_x_center:.3f} cab_face_y_init={cab_face_y_init:.3f}")

# Drawer interior floor z: just below cabinet top (~0.215). Drop should clear interior wall
# Interior floor estimate: handle z - 0.05 (handle is near top of drawer face)
interior_floor_z = hz - 0.05  # approx 0.13
# But the drawer top is open, so we need release_z above floor + bowl_height
release_z = max(interior_floor_z + bowl_height + 0.03, 0.20)

print(f"[PLACE] drop=({drop_x:.3f}, {drop_y:.3f}, {release_z:.3f})")

# Transit at high z (use grasp quat to maintain grip stability)
transit_z = 0.45
above_drop = np.array([drop_x, drop_y, transit_z])
safe_ik(above_drop, quat_b)

# Lower to release height
release_pos = np.array([drop_x, drop_y, release_z])
step_to(release_pos, quat_b, n_steps=4)

# Release
open_gripper()
for _ in range(3):
    get_observation()

# Retreat up
retreat2 = np.array([drop_x, drop_y, release_z + 0.20])
safe_ik(retreat2, quat_b)

goto_home_joint_position()
for _ in range(3):
    get_observation()
