"""
KITCHEN_SCENE3_put_the_frying_pan_on_the_stove — pick-and-place via HANDLE grasp.

Scene (seed 51):
- Frying pan body at world XY ≈ (0.587, -0.252), 20cm dia, ~4cm thick. Body Z: -0.011 to 0.037 (rim).
- Pan handle extends in +Y, ~13.6cm long, ~4cm wide, ~3.7cm thick. Handle center ≈ (0.606, -0.095, 0.033). Handle top z ≈ 0.043.
- Stove top is ESSENTIALLY FLUSH WITH TABLE: stove top z ≈ 0.020. SAM3 "stove top": center (0.632, 0.203, 0.014).
  Stove X=[0.52, 0.71], Y=[0.11, 0.30].
- Burner target zone: SAM3 "burner" = (0.618, 0.204, 0.018), X=[0.55, 0.68], Y=[0.14, 0.27].

Strategy:
1. Localize pan + handle (low-Z filter to remove pan body wall from handle SAM3 mask).
2. Localize stove (filter to foreground, X∈[0.4, 0.8], Y>0).
3. Grasp handle yaw=90° at handle_y_min+0.04 (near body for stability), grasp_z = handle_top - 0.020.
4. Lift HIGH (z=0.50+) — stove is flat, not elevated, so plenty of clearance.
5. Lateral transit to above burner with Y compensation:
   body_offset_y = grasp_y - pan_body_center_y; arm_target_y = burner_cy + body_offset_y.
   But arm Y workspace limit — clip arm_target_y to ≤ 0.32.
6. Release at wrist_z = stove_top_z + 0.135 (pan body settles on stove top, handle still gripped).

Differences from KITCHEN_SCENE9 (cabinet):
- Stove is FLAT (~2cm tall) vs cabinet (22cm). Different release_z formula.
- Pan body is in -Y, stove in +Y → arm transit travels Y from -0.10 to +0.30 (40cm).
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ np.array(
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    )
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


# yaw=90° so gripper closes along world X (handle is along Y axis).
HANDLE_QUAT = make_topdown_quat(yaw_deg=90)


def localize_pan_and_handle(rgb, depth_img, K, E):
    """Returns (pan_pts, body_pts, body_center, handle_pts_low, handle_center, handle_top_z)."""
    # Try both 'frying pan' and 'pan' for the body
    pan_pts = None
    for prompt in ["frying pan", "pan", "skillet"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:3]:
            if m["score"] < 0.3:
                continue
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 500:
                continue
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            # Foreground filter: pan is at robot front, X>0.4
            if cx < 0.4 or cx > 0.85:
                continue
            pan_pts = pts
            print(f"  Pan via '{prompt}' score={m['score']:.3f}: pts={len(pts)} center=({cx:.3f},{cy:.3f})", flush=True)
            break
        if pan_pts is not None:
            break

    if pan_pts is None:
        return None, None, None, None, None, None

    # Pan body Y typically more negative (closer to robot) than handle Y.
    # Use median Y as cutoff: body has Y < median + small margin, handle Y > median - small.
    med_y = float(np.median(pan_pts[:, 1]))
    # Handle is the part with Y > body center area — try Y > med_y + ~0.05 (handle is in tail)
    # Actually pan body is wider than handle, so most points are body. Take points with Y > med_y as candidate handle area.
    # But we also need to filter Z to avoid pan walls (which have high Z).

    # Geometric fallback handle: take highest 30% Y region (the tip end), Z < 0.05
    pan_y_max = float(pan_pts[:, 1].max())
    pan_y_min = float(pan_pts[:, 1].min())
    pan_y_span = pan_y_max - pan_y_min
    handle_mask_g = (pan_pts[:, 1] > pan_y_max - 0.30 * pan_y_span) & (pan_pts[:, 2] < 0.05)
    body_mask = pan_pts[:, 1] < med_y
    body_pts = pan_pts[body_mask]
    handle_pts_low = pan_pts[handle_mask_g] if handle_mask_g.sum() > 50 else None

    # Now refine handle via SAM3, scoring candidates by geometric quality.
    # Pan body center XY (used to constrain candidate location near pan)
    pan_cx = float(np.mean(pan_pts[:, 0]))
    pan_cy = float(np.mean(pan_pts[:, 1]))
    pan_y_max = float(pan_pts[:, 1].max())
    pan_y_min = float(pan_pts[:, 1].min())
    best_handle = None
    best_score = -1
    best_meta = None
    for hprompt in ["frying pan handle", "pan handle"]:  # drop "black handle" — matches kitchen drawer handles
        masks = segment_sam3_text_prompt(rgb, hprompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:6]:
            if m["score"] < 0.20:
                continue
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 30:
                continue
            # Filter Z<0.05 (pan handle is below body wall top)
            pts_low = pts[pts[:, 2] < 0.05]
            if len(pts_low) < 100:
                continue
            cx = float(np.mean(pts_low[:, 0]))
            cy = float(np.mean(pts_low[:, 1]))
            # Spatial proximity to pan: handle must be within pan's bounding box (extended by ~5cm)
            if not (pan_pts[:, 0].min() - 0.02 < cx < pan_pts[:, 0].max() + 0.02):
                continue
            if not (pan_y_min - 0.02 < cy < pan_y_max + 0.05):
                continue
            x_extent = float(pts_low[:, 0].max() - pts_low[:, 0].min())
            y_extent = float(pts_low[:, 1].max() - pts_low[:, 1].min())
            y_min_cand = float(pts_low[:, 1].min())
            top_z_cand = float(pts_low[:, 2].max())
            # Reject if X extent too wide (handle is narrow ~5-7cm; body wall fragments are wider)
            if x_extent > 0.10:
                continue
            # Reject if top_z too high (handle is at most ~0.045; >0.05 means we caught body wall)
            if top_z_cand > 0.052:
                continue
            # Score: prefer many points, narrow X, longer Y, lower top_z
            quality = len(pts_low) - 1000 * x_extent + 100 * y_extent - 500 * max(0, top_z_cand - 0.045)
            print(f"  cand '{hprompt}' s3={m['score']:.3f} pts={len(pts_low)} x_ext={x_extent:.3f} y_ext={y_extent:.3f} top_z={top_z_cand:.3f} c=({cx:.3f},{cy:.3f}) q={quality:.1f}", flush=True)
            if quality > best_score:
                best_score = quality
                best_handle = pts_low
                best_meta = (hprompt, m["score"])
    if best_handle is not None:
        handle_pts_low = best_handle
        print(f"  Handle picked via '{best_meta[0]}' s3={best_meta[1]:.3f}: pts={len(handle_pts_low)}", flush=True)

    if handle_pts_low is None or len(handle_pts_low) < 30:
        return pan_pts, body_pts, None, None, None, None

    handle_center = handle_pts_low.mean(axis=0)
    handle_top_z = float(handle_pts_low[:, 2].max())
    body_center = body_pts.mean(axis=0) if len(body_pts) > 0 else None
    return pan_pts, body_pts, body_center, handle_pts_low, handle_center, handle_top_z


def localize_stove(rgb, depth_img, K, E):
    """Localize stove top surface; returns (pts, center, top_z)."""
    for prompt in ["stove top", "stove", "gas stove", "stovetop", "burner", "stove burner"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:3]:
            if m["score"] < 0.4:
                continue
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 200:
                continue
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            cz = float(np.mean(pts[:, 2]))
            # Stove is at +Y, foreground X
            if not (0.4 < cx < 0.85 and 0.05 < cy < 0.40 and -0.05 < cz < 0.10):
                continue
            return pts, np.array([cx, cy, cz]), float(pts[:, 2].max()), prompt
    return None, None, None, None


# ===== START =====
print(f"Task: {env.handle.task_language}", flush=True)
goto_home_joint_position()
open_gripper()

obs0 = get_observation()
cam0 = obs0["agentview"]
rgb0 = cam0["images"]["rgb"]
depth0 = cam0["images"]["depth"]
depth_img0 = depth0[:, :, 0] if len(depth0.shape) == 3 else depth0
K0 = cam0["intrinsics"]
E0 = cam0["pose_mat"]

# 1. Localize pan + handle
pan_pts, body_pts, body_center, handle_pts, handle_center, handle_top_z = localize_pan_and_handle(rgb0, depth_img0, K0, E0)
if handle_center is None:
    raise RuntimeError("Could not find pan handle")
hx, hy, hz = float(handle_center[0]), float(handle_center[1]), float(handle_center[2])
handle_y_min = float(handle_pts[:, 1].min())
handle_y_max = float(handle_pts[:, 1].max())
pbx = float(body_center[0]) if body_center is not None else hx
pby = float(body_center[1]) if body_center is not None else hy - 0.18
pbz = float(body_center[2]) if body_center is not None else 0.0
print(f"Handle: center=({hx:.3f},{hy:.3f},{hz:.3f}) top_z={handle_top_z:.3f} Y∈({handle_y_min:.3f},{handle_y_max:.3f})", flush=True)
print(f"Pan body: center=({pbx:.3f},{pby:.3f},{pbz:.3f})", flush=True)

# 2. Localize stove
stove_pts, stove_center, stove_top_z, stove_prompt = localize_stove(rgb0, depth_img0, K0, E0)
if stove_center is None:
    raise RuntimeError("Could not find stove")
sx, sy, sz = float(stove_center[0]), float(stove_center[1]), float(stove_center[2])
stove_y_min = float(stove_pts[:, 1].min())
stove_y_max = float(stove_pts[:, 1].max())
print(f"Stove ({stove_prompt}): center=({sx:.3f},{sy:.3f},{sz:.3f}) top_z={stove_top_z:.3f} Y∈({stove_y_min:.3f},{stove_y_max:.3f})", flush=True)

# 3. Grasp plan: ADAPTIVE yaw based on handle orientation
# Compute handle's principal XY axis. If handle extends more in Y, yaw=90 (gripper along X);
# if more in X, yaw=0 (gripper along Y).
hx_ext = float(handle_pts[:, 0].max() - handle_pts[:, 0].min())
hy_ext = float(handle_pts[:, 1].max() - handle_pts[:, 1].min())
handle_x_min = float(handle_pts[:, 0].min())
handle_x_max = float(handle_pts[:, 0].max())

# Use PCA on XY-projected handle points to find principal axis.
# (OBB axes can include the vertical Z direction which is not useful for gripper yaw.)
long_axis_xy = None
long_axis_xy_norm = 0.0
long_extent_xy = 0.05
try:
    xy = handle_pts[:, :2] - handle_pts[:, :2].mean(axis=0)
    cov = np.cov(xy.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Largest eigenvalue is principal axis
    long_axis_xy = eigvecs[:, -1]  # 2D unit vector
    long_axis_xy = long_axis_xy / np.linalg.norm(long_axis_xy)
    long_axis_xy_norm = 1.0
    # Effective extent along principal axis: 2 * std-dev (or 4*sqrt(eigval))
    long_extent_xy = 4.0 * float(np.sqrt(max(eigvals[-1], 1e-9)))
    angle_long = np.degrees(np.arctan2(long_axis_xy[1], long_axis_xy[0]))
    # Empirically validated: yaw=-angle_long (after wrap to (-90,90]) gives stable grasps on
    # seeds 51-55 (with handle PCA angles 67-110). The math says yaw=angle_long is theoretically
    # perpendicular to long axis, but yaw=-angle_long performs better — IK picks a more
    # natural arm configuration that doesn't twist during transit.
    yaw_deg = -angle_long
    while yaw_deg > 90:
        yaw_deg -= 180
    while yaw_deg <= -90:
        yaw_deg += 180
    print(f"Handle PCA: long_axis_xy=({long_axis_xy[0]:.3f},{long_axis_xy[1]:.3f}) angle={angle_long:.1f} long_ext={long_extent_xy:.3f} → yaw={yaw_deg:.1f}", flush=True)
except Exception as ex:
    print(f"PCA failed: {ex}, fallback to extent-based", flush=True)
    yaw_deg = 90.0 if hy_ext > hx_ext else 0.0

# Confidence check: if handle has nearly square XY extent (rotation ambiguous), prefer yaw=90 (default).
if abs(hx_ext - hy_ext) < 0.015:
    yaw_deg = 90.0
    print(f"  Square extent — defaulting to yaw=90", flush=True)

grasp_quat = make_topdown_quat(yaw_deg=yaw_deg)

# Grasp position: at center of handle, biased toward the body side.
# Body is at lower Y typically. Bias along the long axis toward body.
# Long axis direction toward body: from handle center to body center.
toward_body = np.array([pbx - hx, pby - hy])
toward_body_norm = np.linalg.norm(toward_body)
if toward_body_norm > 1e-3:
    toward_body = toward_body / toward_body_norm
else:
    toward_body = np.array([0.0, -1.0])

# Move grasp center 0.03m along the long axis toward body
if long_axis_xy is not None and long_axis_xy_norm > 1e-3:
    proj = float(np.dot(toward_body, long_axis_xy))  # scalar
    bias_dir = proj * long_axis_xy  # along long axis, toward body
else:
    bias_dir = toward_body * 0.5

# Magnitude: bias toward body but stay inside handle. Use min(half long extent, 3.5cm).
bias_mag = min(0.035, long_extent_xy * 0.30)
bias_dir_norm_arr = bias_dir / max(np.linalg.norm(bias_dir), 1e-6)
grasp_x = float(hx + bias_dir_norm_arr[0] * bias_mag)
grasp_y = float(hy + bias_dir_norm_arr[1] * bias_mag)
grasp_z = max(handle_top_z - 0.020, 0.005)

print(f"Grasp plan: pos=({grasp_x:.3f},{grasp_y:.3f},{grasp_z:.3f}) yaw={yaw_deg:.1f} (h_xext={hx_ext:.3f}, h_yext={hy_ext:.3f})", flush=True)

# 4. Approach + grasp
hover_z = 0.20
j = solve_ik([grasp_x, grasp_y, hover_z], grasp_quat.tolist())
if j is not None:
    move_to_joints(j)
for z_step in [0.10, grasp_z]:
    j = solve_ik([grasp_x, grasp_y, z_step], grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)
close_gripper()

obs_g = get_observation()
gw_after = float(obs_g["robot_cartesian_pos"][-1])
print(f"After close: gw={gw_after:.3f}", flush=True)

# Retry with adjusted z + alt grasp positions if grasp failed
if gw_after < 0.020:
    print("Air grasp — retrying with z + position variants", flush=True)
    # Alternate grasp positions: nudge toward and away from handle center, plus alt yaw
    alt_grasps = [
        (grasp_x, grasp_y, grasp_z + 0.005, yaw_deg),
        (grasp_x, grasp_y, grasp_z - 0.005, yaw_deg),
        (grasp_x, grasp_y, grasp_z + 0.010, yaw_deg),
        # Fall back to handle center (no body bias)
        (hx, hy, grasp_z, yaw_deg),
        # Try yaw=90 (cabinet recipe)
        (hx, float(np.clip(handle_y_min + 0.04, hy - 0.04, hy + 0.02)), max(handle_top_z - 0.020, 0.005), 90.0),
        # Try yaw=0
        (hx, hy, grasp_z, 0.0),
    ]
    for (rx, ry, rz, ryaw) in alt_grasps:
        rz = max(rz, 0.003)
        rquat = make_topdown_quat(yaw_deg=ryaw)
        open_gripper()
        j = solve_ik([rx, ry, hover_z], rquat.tolist())
        if j is not None:
            move_to_joints(j)
        j = solve_ik([rx, ry, rz], rquat.tolist())
        if j is not None:
            move_to_joints(j)
        close_gripper()
        obs_r = get_observation()
        gw_after = float(obs_r["robot_cartesian_pos"][-1])
        print(f"  retry pos=({rx:.3f},{ry:.3f},{rz:.3f}) yaw={ryaw:.1f}: gw={gw_after:.3f}", flush=True)
        if gw_after >= 0.020:
            grasp_x, grasp_y, grasp_z = rx, ry, rz
            grasp_quat = rquat
            yaw_deg = ryaw
            break

# 5. Lift HIGH for transit (stove is short, but transit goes 40cm in Y)
lift_z = max(grasp_z + 0.30, 0.50)
for step_z in [grasp_z + 0.05, grasp_z + 0.15, lift_z]:
    j = solve_ik([grasp_x, grasp_y, step_z], grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)
obs_l = get_observation()
gw_lift = float(obs_l["robot_cartesian_pos"][-1])
print(f"After lift to z={lift_z:.3f}: gw={gw_lift:.3f}", flush=True)

# 6. Lateral transit. Pan body hangs from grasp by offset (pbx-grasp_x, pby-grasp_y).
# Place pan body at stove center: arm = stove_xy - (body_pos - grasp_pos) = stove_xy + (grasp_pos - body_pos)
body_offset_x = grasp_x - pbx
body_offset_y = grasp_y - pby
target_x_stove = sx
target_y_stove = sy
raw_arm_target_x = target_x_stove + body_offset_x
raw_arm_target_y = target_y_stove + body_offset_y
# Workspace clamp: Y should not exceed 0.30 (IK becomes unreliable beyond)
arm_target_x = float(np.clip(raw_arm_target_x, 0.40, 0.75))
arm_target_y = min(raw_arm_target_y, 0.30)
y_clipped = raw_arm_target_y > 0.30
print(f"Body offset=({body_offset_x:.3f},{body_offset_y:.3f}), raw_arm=({raw_arm_target_x:.3f},{raw_arm_target_y:.3f}), clipped arm_target=({arm_target_x:.3f},{arm_target_y:.3f}) y_clipped={y_clipped}", flush=True)

# Interpolated transit at lift height (more steps for smoother trajectory)
n_steps = 8
for i in range(1, n_steps + 1):
    t = i / n_steps
    wx = grasp_x + t * (arm_target_x - grasp_x)
    wy = grasp_y + t * (arm_target_y - grasp_y)
    j = solve_ik([wx, wy, lift_z], grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)
obs_t = get_observation()
gw_t = float(obs_t["robot_cartesian_pos"][-1])
print(f"At above-target: gw={gw_t:.3f}", flush=True)

# 7. Descend. Need wrist_z such that pan_bottom is just above stove top.
# Empirical model: when handle is gripped at handle_top - 0.02, pan_bottom_z ≈ wrist_z - 0.135 (TCP 10cm + handle/body offset).
# So wrist_z = stove_top_z + small_margin + 0.135.
# Use small margin 0.005-0.020 to ensure clean release.
release_z_wrist = stove_top_z + 0.135 + 0.020  # = stove_top_z + 0.155
print(f"Release wrist z = {release_z_wrist:.3f} (stove_top_z={stove_top_z:.3f})", flush=True)

n_d = 6
for i in range(1, n_d + 1):
    t = i / n_d
    wz = lift_z + t * (release_z_wrist - lift_z)
    j = solve_ik([arm_target_x, arm_target_y, wz], grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)
    # Mid-descent gw check — if grip dropped, abort and place at higher Z
    if i == 3:
        obs_mid = get_observation()
        gw_mid = float(obs_mid["robot_cartesian_pos"][-1])
        if gw_mid < 0.020:
            print(f"  WARN: gw={gw_mid:.3f} mid-descent — pan slipped, releasing here", flush=True)
            break
obs_p = get_observation()
gw_p = float(obs_p["robot_cartesian_pos"][-1])
ee_p = obs_p["robot_cartesian_pos"][:3]
print(f"At release pos=({ee_p[0]:.3f},{ee_p[1]:.3f},{ee_p[2]:.3f}): gw={gw_p:.3f}", flush=True)

# 8. Release & settle
open_gripper()
for _ in range(10):
    get_observation()

# 9. Retreat up
j = solve_ik([arm_target_x, arm_target_y, lift_z], grasp_quat.tolist())
if j is not None:
    move_to_joints(j)
goto_home_joint_position()
print("Done", flush=True)
