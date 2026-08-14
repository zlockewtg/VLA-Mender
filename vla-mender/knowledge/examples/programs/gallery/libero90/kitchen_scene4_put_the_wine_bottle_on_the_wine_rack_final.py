"""
KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack

Wine bottle is upright on the table. Wine rack is a tilted V-shape wooden structure
on the LEFT side. Bottle must end up on the rack.

Strategy:
1. Settle physics, localize wine bottle (body_xy_centroid).
2. Localize wine rack via "wooden tilted board" — top mask is the upper board (peak z~0.30).
3. Grasp bottle at top - 0.04 (yaw=0, top-down), like wine_drawer task.
4. Lift to safe z.
5. Move above rack center XY.
6. Lower until bottle bottom hovers ~5cm above rack peak.
7. Release; let physics settle.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def body_xy_centroid(pts, radius=0.04, iters=3):
    xy = pts[:, :2]
    cx, cy = float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))
    for _ in range(iters):
        dist = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
        inliers = pts[dist < radius]
        if len(inliers) < 5:
            break
        cx, cy = float(inliers[:, 0].mean()), float(inliers[:, 1].mean())
    return cx, cy


# Step 1: settle physics
for _ in range(3):
    open_gripper()
    close_gripper()
open_gripper()

obs = get_observation()
rgb = obs["agentview"]["images"]["rgb"]
depth = obs["agentview"]["images"]["depth"]
K = obs["agentview"]["intrinsics"]
E = obs["agentview"]["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

# Step 2: localize wine bottle
bottle_pts = None
bottle_mask = None
for prompt in ["wine bottle", "bottle", "dark bottle"]:
    masks = segment_sam3_text_prompt(rgb, prompt)
    if not masks:
        continue
    best = max(masks, key=lambda d: d["score"])
    if best["score"] < 0.50:
        continue
    bottle_mask = best["mask"].astype(np.uint8)
    pts = mask_to_world_points(bottle_mask, depth_img, K, E)
    if pts is None or len(pts) < 50:
        continue
    bottle_pts = pts
    break
if bottle_pts is None:
    raise RuntimeError("No wine bottle found")

bx, by = body_xy_centroid(bottle_pts, radius=0.04, iters=3)
# z_top from points within bottle XY
z_keep = bottle_pts[(np.abs(bottle_pts[:, 0] - bx) < 0.04) & (np.abs(bottle_pts[:, 1] - by) < 0.04)]
bottle_z_top = float(z_keep[:, 2].max() if len(z_keep) > 5 else bottle_pts[:, 2].max())
bottle_z_bot = float(z_keep[:, 2].min() if len(z_keep) > 5 else bottle_pts[:, 2].min())
bottle_height = max(0.10, bottle_z_top - bottle_z_bot)
print(f"BOTTLE: body=({bx:.3f},{by:.3f}), top={bottle_z_top:.3f}, h={bottle_height:.3f}", flush=True)

# Step 3: localize wine rack — find both boards (upper tilted + lower flat)
upper_pts = None  # tilted upper board (peak)
lower_pts = None  # lower flat board
for prompt in ["wooden tilted board", "wooden boards", "wood platform"]:
    masks = segment_sam3_text_prompt(rgb, prompt)
    if not masks:
        continue
    sorted_masks = sorted(masks, key=lambda d: d["score"], reverse=True)
    for m in sorted_masks[:5]:
        if m["score"] < 0.50:
            continue
        pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
        if pts is None or len(pts) < 100:
            continue
        # filter to workspace + rack region (Y < 0 = left side)
        pts_ws = pts[(pts[:, 2] > 0.05) & (pts[:, 2] < 0.40) & (pts[:, 0] > 0.30) & (pts[:, 0] < 0.85) & (pts[:, 1] < 0.0)]
        if len(pts_ws) < 100:
            continue
        z_p50 = float(np.percentile(pts_ws[:, 2], 50))
        if z_p50 > 0.20 and upper_pts is None:
            upper_pts = pts_ws
        elif 0.08 < z_p50 < 0.18 and lower_pts is None:
            lower_pts = pts_ws
    if upper_pts is not None and lower_pts is not None:
        break

if upper_pts is None:
    raise RuntimeError("Wine rack upper board not found")

# Use the V-trough point: where bottle naturally settles (between upper back board and lower front board)
upper_x = float(np.percentile(upper_pts[:, 0], 50))
upper_y = float(np.percentile(upper_pts[:, 1], 50))
upper_z = float(np.percentile(upper_pts[:, 2], 95))

if lower_pts is not None:
    lower_x = float(np.percentile(lower_pts[:, 0], 50))
    lower_y = float(np.percentile(lower_pts[:, 1], 50))
    lower_z = float(np.percentile(lower_pts[:, 2], 95))
else:
    lower_x = upper_x
    lower_y = upper_y + 0.05  # estimate front offset
    lower_z = upper_z - 0.15

# Drop point: place bottle on lower-board flat surface, biased BACK (toward upper board base)
# so the bottle doesn't slide off the front. Lower_y is less negative than upper_y.
# Pick a Y between upper_y and lower_y, weighted toward lower (so we're on lower-board surface).
# Place bottle at midpoint between boards (V-trough)
# Force bottle to tip INTO upper board: place SLIGHTLY in front of trough
rack_x = (upper_x + lower_x) / 2
rack_y = (upper_y + lower_y) / 2 - 0.005  # tiny bias toward upper (back)
rack_z_peak = upper_z
rack_z_low = lower_z
print(f"RACK upper=({upper_x:.3f},{upper_y:.3f},{upper_z:.3f})", flush=True)
print(f"RACK lower=({lower_x:.3f},{lower_y:.3f},{lower_z:.3f})", flush=True)
print(f"RACK drop=({rack_x:.3f},{rack_y:.3f}) z_low={rack_z_low:.3f} z_peak={rack_z_peak:.3f}", flush=True)

# Step 4: grasp bottle (top-down, yaw=0)
# Grasp on the upper-body shoulder (below the neck, on the wider body) for a firm grip.
# bottle is ~16cm tall, top is cap+neck (~3cm), shoulder starts at top-0.05.
# Grasp at top - 0.06 (deeper than top-0.04) to get body grip.
grasp_z = bottle_z_top - 0.06
quat = make_topdown_quat(0)

# Pre-grasp hover
pre_z = grasp_z + 0.20
j = solve_ik([bx, by, pre_z], quat.tolist())
if j is not None:
    move_to_joints(j)

# Lower (4 passes is enough for IK convergence)
for _ in range(4):
    j = solve_ik([bx, by, grasp_z], quat.tolist())
    if j is not None:
        move_to_joints(j)
close_gripper()
obs_c = get_observation()
gw = obs_c["robot_cartesian_pos"][7]
print(f"After close: gw={gw:.3f}", flush=True)

# Retry deeper if grip too weak (need at least 0.10 for stable cylinder lift)
if gw < 0.10:
    for z_extra in [-0.020, -0.040]:
        open_gripper()
        for _ in range(4):
            j = solve_ik([bx, by, grasp_z + z_extra], quat.tolist())
            if j is not None:
                move_to_joints(j)
        close_gripper()
        obs_c = get_observation()
        gw = obs_c["robot_cartesian_pos"][7]
        print(f"Retry z+{z_extra}: gw={gw:.3f}", flush=True)
        if gw > 0.10:
            grasp_z = grasp_z + z_extra
            break

# Step 5: lift straight up
lift_z = grasp_z + 0.35  # ~0.46
for _ in range(3):
    j = solve_ik([bx, by, lift_z], quat.tolist())
    if j is not None:
        move_to_joints(j)
obs_l = get_observation()
print(f"After lift: pos={obs_l['robot_cartesian_pos'][:3]}, gw={obs_l['robot_cartesian_pos'][7]:.3f}", flush=True)

# Step 6: move laterally above rack at high z
transit_z = max(0.50, lift_z)
for _ in range(3):
    j = solve_ik([rack_x, rack_y, transit_z], quat.tolist())
    if j is not None:
        move_to_joints(j)
obs_t = get_observation()
print(f"After transit: pos={obs_t['robot_cartesian_pos'][:3]}, gw={obs_t['robot_cartesian_pos'][7]:.3f}", flush=True)

# Step 7: place bottle UPRIGHT on the lower flat board surface.
tips_offset_to_bottom = bottle_height - 0.06
# Bottle BOTTOM at lower-board surface (z_low); add 1cm clearance.
release_z = rack_z_low + tips_offset_to_bottom + 0.01
print(f"release_z target={release_z:.3f} (rack_z_low={rack_z_low:.3f}, off={tips_offset_to_bottom:.3f})", flush=True)

# Multi-pass descent with intermediate z stops
descent_steps = list(np.linspace(transit_z, release_z, 6))
for z_step in descent_steps:
    for _ in range(4):
        j = solve_ik([rack_x, rack_y, float(z_step)], quat.tolist())
        if j is not None:
            move_to_joints(j)
obs_d = get_observation()
print(f"After descent: pos={obs_d['robot_cartesian_pos'][:3]}, gw={obs_d['robot_cartesian_pos'][7]:.3f}", flush=True)

# Step 8: release
open_gripper()
for _ in range(8):
    get_observation()

# Step 9: gentle retreat — move STRAIGHT UP slowly
obs_post = get_observation()
cur_x, cur_y, cur_z = obs_post["robot_cartesian_pos"][:3]
for dz in [0.05, 0.15, 0.25]:
    target_tips_z = (cur_z - 0.10) + dz
    j = solve_ik([float(cur_x), float(cur_y), target_tips_z], quat.tolist())
    if j is not None:
        move_to_joints(j)

# Settle
for _ in range(30):
    get_observation()

# Check bottle position for diagnostics
obs_final = get_observation()
rgb_final = obs_final["agentview"]["images"]["rgb"]
masks_final = segment_sam3_text_prompt(rgb_final, "wine bottle")
if masks_final:
    best_f = max(masks_final, key=lambda d: d["score"])
    pts_f = mask_to_world_points(best_f["mask"].astype(np.uint8), depth_img, K, E)
    if pts_f is not None and len(pts_f) > 5:
        ctr_f = pts_f.mean(axis=0)
        print(f"FINAL bottle: ctr=({ctr_f[0]:.3f},{ctr_f[1]:.3f},{ctr_f[2]:.3f}) z_min={pts_f[:,2].min():.3f} z_max={pts_f[:,2].max():.3f}", flush=True)

# Move arm home
goto_home_joint_position()
for _ in range(15):
    get_observation()
