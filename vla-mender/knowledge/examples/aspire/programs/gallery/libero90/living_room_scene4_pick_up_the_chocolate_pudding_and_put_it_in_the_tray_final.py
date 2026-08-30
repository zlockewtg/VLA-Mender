"""
LIVING_ROOM_SCENE4: pick up the chocolate pudding and put it in the tray.

Scene from seed 51:
- Pudding (flat brown box ~10×5×3cm) at ~[0.616, -0.206, 0.036], top_z~0.046
  Best prompt: "chocolate pudding box" (score 0.918)
- Tray (LARGE wicker tray ~44×19×9.5cm) at ~[0.426, 0.270, 0.056], top_z~0.10
  Best prompt: "wooden tray" (score 0.93)
- Transport mostly in Y direction (~47cm)
- Pudding x=0.616 is comfortable (no workspace edge issue)

Strategy:
1. Detect both objects pre-grasp (clean view)
2. GraspNet for pudding XY+Z, override quat to TOP_DOWN
3. Lift, transport diagonally, drop into tray
4. Tray center via p10/p90 midpoint (avoids SAM3 wall bias)
5. drop_z = tray_floor_z + obj_height + 0.05 OR tray_top_z + 0.04 (above rim)
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


TOP_DOWN_QUAT = make_topdown_quat(0)


def localize_object_filtered(rgb, depth, K, E, prompts,
                             z_min=0.005, z_max=0.20, min_pts=10,
                             extent_max=None, min_score=0.20):
    """Try prompts in order. Return (center, pts, mask, obb) for first valid."""
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:5]:
            if m["score"] < min_score:
                continue
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < min_pts:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c = obb["center"]
            if c[2] < z_min or c[2] > z_max:
                continue
            if extent_max is not None and max(obb["extent"]) > extent_max:
                continue
            return c, pts, mask, obb
    return None, None, None, None


# ── Observe ──────────────────────────────────────────────────────────────────
goto_home_joint_position()
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]
print(f"Task: {env.handle.task_language}", flush=True)


# ── Localize chocolate pudding ──────────────────────────────────────────────
# Flat brown box ~10×5×3cm. Filter by extent < 0.15 to reject the tray.
pudding_center, pudding_pts, pudding_mask, pudding_obb = localize_object_filtered(
    rgb, depth, K, E,
    ["chocolate pudding box", "chocolate pudding", "dark brown box", "brown box", "pudding"],
    z_min=0.005, z_max=0.080,
    extent_max=0.15,  # tray extent ~0.44, pudding extent ~0.08
    min_score=0.30,
)
if pudding_center is None:
    raise RuntimeError("Chocolate pudding not found")

pudding_extent = pudding_obb["extent"]
pudding_z_min = pudding_pts[:, 2].min()
pudding_z_max = pudding_pts[:, 2].max()
pudding_height = pudding_z_max - pudding_z_min
print(f"Pudding center={pudding_center.round(3)}, ext={pudding_extent.round(3)}, "
      f"z=[{pudding_z_min:.3f},{pudding_z_max:.3f}], h={pudding_height:.3f}", flush=True)


# ── Localize wooden tray ─────────────────────────────────────────────────────
# Large wicker tray ~44×19×9.5cm. Use full pts to compute true center via p10/p90.
tray_center_obb, tray_pts, tray_mask, tray_obb = localize_object_filtered(
    rgb, depth, K, E,
    ["wooden tray", "wicker tray", "tray", "basket"],
    z_min=0.020, z_max=0.20,
    min_score=0.40,
)
if tray_center_obb is None:
    raise RuntimeError("Tray not found")

# True tray center via p10/p90 midpoints (avoid SAM3 wall bias)
tray_x_mid = (np.percentile(tray_pts[:, 0], 10) + np.percentile(tray_pts[:, 0], 90)) / 2
tray_y_mid = (np.percentile(tray_pts[:, 1], 10) + np.percentile(tray_pts[:, 1], 90)) / 2
tray_floor_z = float(np.percentile(tray_pts[:, 2], 10))   # interior floor
tray_top_z   = float(np.percentile(tray_pts[:, 2], 90))   # rim
tray_center = np.array([tray_x_mid, tray_y_mid, tray_floor_z])
print(f"Tray center={tray_center.round(3)}, floor_z={tray_floor_z:.3f}, top_z={tray_top_z:.3f}", flush=True)
print(f"Tray X range: [{np.percentile(tray_pts[:, 0], 10):.3f}, {np.percentile(tray_pts[:, 0], 90):.3f}]", flush=True)
print(f"Tray Y range: [{np.percentile(tray_pts[:, 1], 10):.3f}, {np.percentile(tray_pts[:, 1], 90):.3f}]", flush=True)


# ── Plan grasp via GraspNet ──────────────────────────────────────────────────
grasp_poses, grasp_scores = plan_grasp(depth, K, pudding_mask)
if len(grasp_poses) == 0:
    raise RuntimeError("No grasp candidates from GraspNet")

best_grasp_world, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
if best_grasp_world is None:
    best_grasp_world = E @ grasp_poses[grasp_scores.argmax()]

gn_pos, _ = decompose_transform(best_grasp_world)

# Sanity check: GraspNet XY must be near pudding center
xy_dist = np.linalg.norm(gn_pos[:2] - pudding_center[:2])
if xy_dist > 0.08:
    print(f"GraspNet XY off by {xy_dist:.3f}, falling back to centroid", flush=True)
    grasp_x, grasp_y = pudding_center[0], pudding_center[1]
else:
    grasp_x, grasp_y = gn_pos[0], gn_pos[1]

# Flat-box grasp Z: just below top surface
grasp_z = pudding_z_max - 0.010
# Clamp to safe range
grasp_z = float(np.clip(grasp_z, pudding_center[2] - 0.005, pudding_z_max + 0.005))

print(f"Grasp pos: x={grasp_x:.3f}, y={grasp_y:.3f}, z={grasp_z:.3f}", flush=True)


# ── Pick the pudding ─────────────────────────────────────────────────────────
open_gripper()

approach_z = max(grasp_z + 0.18, 0.25)
goto_pose([grasp_x, grasp_y, approach_z], TOP_DOWN_QUAT.tolist())
goto_pose([grasp_x, grasp_y, grasp_z + 0.06], TOP_DOWN_QUAT.tolist())
goto_pose([grasp_x, grasp_y, grasp_z], TOP_DOWN_QUAT.tolist())
close_gripper()

# Settle and re-close
for _ in range(5):
    get_observation()
close_gripper()

obs_g = get_observation()
rcp = obs_g.get("robot_cartesian_pos", [])
gw = float(rcp[-1]) if len(rcp) >= 4 else 0.0
print(f"After grasp: gw={gw:.4f}", flush=True)

# Retry with center grasp if air grasp
if gw < 0.04:
    print("Air grasp — retrying with center top-down grasp at multiple Zs", flush=True)
    for retry_dz in [0.005, 0.000, -0.005, 0.010]:
        open_gripper()
        retry_z = pudding_center[2] + retry_dz
        retry_z = float(np.clip(retry_z, pudding_z_min + 0.005, pudding_z_max + 0.005))
        goto_pose([pudding_center[0], pudding_center[1], approach_z], TOP_DOWN_QUAT.tolist())
        goto_pose([pudding_center[0], pudding_center[1], retry_z + 0.04], TOP_DOWN_QUAT.tolist())
        goto_pose([pudding_center[0], pudding_center[1], retry_z], TOP_DOWN_QUAT.tolist())
        close_gripper()
        for _ in range(5):
            get_observation()
        close_gripper()
        obs_r = get_observation()
        rcp_r = obs_r.get("robot_cartesian_pos", [])
        gw = float(rcp_r[-1]) if len(rcp_r) >= 4 else 0.0
        print(f"Retry dz={retry_dz:+.3f}: gw={gw:.4f}", flush=True)
        if gw >= 0.04:
            grasp_x, grasp_y = pudding_center[0], pudding_center[1]
            grasp_z = retry_z
            break


# ── Lift ─────────────────────────────────────────────────────────────────────
lift_z = max(grasp_z + 0.25, 0.32)
goto_pose([grasp_x, grasp_y, lift_z], TOP_DOWN_QUAT.tolist())

obs_l = get_observation()
rcp_l = obs_l.get("robot_cartesian_pos", [])
gw_l = float(rcp_l[-1]) if len(rcp_l) >= 4 else 0.0
print(f"After lift: gw={gw_l:.4f}", flush=True)


# ── Transport via 3-step lateral (constant lift_z) ───────────────────────────
# XY offset from grasp to actual pudding centroid (preserves true object position)
grasp_offset_xy = np.array([grasp_x - pudding_center[0], grasp_y - pudding_center[1]])
# Place at tray center + grasp_offset (so the pudding lands at tray_center)
release_x = tray_center[0] + grasp_offset_xy[0]
release_y = tray_center[1] + grasp_offset_xy[1]

for frac in [0.33, 0.67, 1.0]:
    wpt_x = grasp_x + frac * (release_x - grasp_x)
    wpt_y = grasp_y + frac * (release_y - grasp_y)
    goto_pose([wpt_x, wpt_y, lift_z], TOP_DOWN_QUAT.tolist())
    close_gripper()


# ── Lower into tray ──────────────────────────────────────────────────────────
# drop_z = tray_floor_z + pudding_height + safety margin
# But also keep above rim during the descent to avoid hitting the wall
drop_z = max(tray_floor_z + pudding_height + 0.05, tray_top_z + 0.04)

print(f"Drop XY=({release_x:.3f},{release_y:.3f}), drop_z={drop_z:.3f}", flush=True)

# Multi-step descent
for step_z in [lift_z, tray_top_z + 0.18, tray_top_z + 0.10, drop_z]:
    goto_pose([release_x, release_y, step_z], TOP_DOWN_QUAT.tolist())

# Open gripper, settle
open_gripper()
for _ in range(8):
    get_observation()

# Retract
goto_pose([release_x, release_y, lift_z], TOP_DOWN_QUAT.tolist())
goto_home_joint_position()

for _ in range(10):
    get_observation()
print("Done", flush=True)
