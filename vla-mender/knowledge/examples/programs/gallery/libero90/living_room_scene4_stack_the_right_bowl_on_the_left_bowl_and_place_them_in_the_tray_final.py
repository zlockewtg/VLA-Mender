"""
LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray

Task: Stack the RIGHT bowl on the LEFT bowl, and place them in the tray.
Convention: left = min Y, right = max Y.

Strategy:
  Phase 1: Pick LEFT bowl, place in tray (aim at one compartment).
  Phase 2: Pick RIGHT bowl, drop directly above left bowl in tray.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def localize_first(rgb, depth_img, K, E, prompts, min_score=0.0):
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        best = max(masks, key=lambda d: d["score"])
        if best["score"] < min_score:
            continue
        mask = best["mask"].astype(np.uint8)
        pts = mask_to_world_points(mask, depth_img, K, E)
        if pts is None or len(pts) < 10:
            continue
        center = get_oriented_bounding_box_from_3d_points(pts)["center"]
        return center, pts, mask, best["score"]
    return None, None, None, 0.0


def find_bowls(rgb, depth_img, K, E):
    """Return up to 2 bowls sorted by world-Y ascending."""
    candidates = []
    for prompt in ("small bowl", "bowl", "black bowl"):
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in masks[:10]:
            if m["score"] < 0.5:
                continue
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 100:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c = obb["center"]
            ext = obb["extent"]
            if c[2] < 0.0 or c[2] > 0.20:
                continue
            if c[0] < 0.30 or c[0] > 0.70:
                continue
            max_xy = max(ext[0], ext[1])
            if max_xy < 0.06 or max_xy > 0.18:
                continue
            if ext[2] > 0.15:
                continue
            if any(np.linalg.norm(c[:2] - prev[0][:2]) < 0.06 for prev in candidates):
                continue
            candidates.append((c, pts, mask, m["score"]))
    candidates.sort(key=lambda t: t[0][1])
    return candidates


def grasp_bowl(bowl_center, bowl_pts, yaw_deg=0):
    """OBB-snapped grasp with 2-attempt retry. Returns (final_pos, quat, grip_w)."""
    bowl_obb = get_oriented_bounding_box_from_3d_points(bowl_pts)
    bowl_top_z = float(bowl_pts[:, 2].max())
    bowl_xy = np.array([bowl_obb["center"][0], bowl_obb["center"][1]])
    target_grasp_z = bowl_top_z - 0.040
    quat = make_topdown_quat(yaw_deg)

    open_gripper()
    pre = np.array([bowl_xy[0], bowl_xy[1], target_grasp_z + 0.15])
    j = solve_ik(pre.tolist(), quat.tolist())
    if j is not None: move_to_joints(j)

    grip_w = 0.0
    final_pos = np.array([bowl_xy[0], bowl_xy[1], target_grasp_z])
    for attempt, depth_extra in enumerate([0.0, -0.020]):
        if attempt > 0:
            open_gripper()
        gz = target_grasp_z + depth_extra
        for _ in range(4):
            j = solve_ik([bowl_xy[0], bowl_xy[1], gz], quat.tolist())
            if j is not None: move_to_joints(j)
        obs = get_observation()
        cur = np.array(obs['robot_cartesian_pos'][:3])
        print(f"  [grasp.{attempt}] target_z={gz:.3f} hand=({cur[0]:.3f},{cur[1]:.3f},{cur[2]:.3f})", flush=True)
        close_gripper()
        obs = get_observation()
        grip_w = float(obs["robot_cartesian_pos"][7]) if len(obs["robot_cartesian_pos"]) > 7 else -1.0
        print(f"  [grasp.{attempt}] grip_w={grip_w:.3f}", flush=True)
        if grip_w > 0.05:
            final_pos[2] = gz
            break
    return final_pos, quat, grip_w


def stepped_lift_and_transit(grasp_pos, target_xy, quat, lift_z=0.40):
    """Compact: stepped lift + lateral transit."""
    obs = get_observation()
    hand_z = float(obs["robot_cartesian_pos"][2])
    for z in [hand_z + 0.07, hand_z + 0.18, lift_z]:
        j = solve_ik([grasp_pos[0], grasp_pos[1], z], quat.tolist())
        if j is not None: move_to_joints(j)
    j = solve_ik([target_xy[0], target_xy[1], lift_z], quat.tolist())
    if j is not None: move_to_joints(j)


def descend_release_retreat(target_xy, target_z, quat, retreat_z=0.35):
    """Compact descent + release + retreat."""
    for z in [0.30, 0.20, target_z]:
        j = solve_ik([target_xy[0], target_xy[1], z], quat.tolist())
        if j is not None: move_to_joints(j)
    j = solve_ik([target_xy[0], target_xy[1], target_z], quat.tolist())
    if j is not None: move_to_joints(j)
    obs = get_observation()
    cur = np.array(obs['robot_cartesian_pos'][:3])
    print(f"  [release] target=({target_xy[0]:.3f},{target_xy[1]:.3f},{target_z:.3f}) hand={cur.tolist()}", flush=True)
    open_gripper()
    for _ in range(3):
        get_observation()
    j = solve_ik([target_xy[0], target_xy[1], retreat_z], quat.tolist())
    if j is not None: move_to_joints(j)


# ============================================================
# MAIN
# ============================================================

# Settle
for _ in range(3):
    open_gripper()
    close_gripper()
open_gripper()

obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

# Find bowls
bowls = find_bowls(rgb, depth_img, K, E)
print(f"[main] found {len(bowls)} bowls", flush=True)
for i, (c, _, _, s) in enumerate(bowls):
    print(f"  bowl[{i}] center=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) score={s:.3f}", flush=True)
if len(bowls) < 2:
    raise RuntimeError(f"Need 2 bowls, found {len(bowls)}")

left_center, left_pts, _, _ = bowls[0]
right_center, right_pts, _, _ = bowls[-1]
bowl_height = float(get_oriented_bounding_box_from_3d_points(right_pts)["extent"][2])

# Find tray
tray_center, tray_pts, _, _ = localize_first(
    rgb, depth_img, K, E, ["wooden tray", "tray"], min_score=0.3
)
if tray_center is None:
    raise RuntimeError("Tray not found")
tray_rim_z = float(np.percentile(tray_pts[:, 2], 95))
print(f"[main] tray=({tray_center[0]:.3f},{tray_center[1]:.3f}) rim_z={tray_rim_z:.3f}", flush=True)

# Tray X range. The wooden tray has 3 compartments along X; aim at left compartment
# (smaller X) which is large enough to hold a bowl (~13cm wide).
# Tray ext_x ≈ 0.44, half = 0.22. Tray X center ~ 0.43. Left compartment at X ~ 0.43 - 0.10 = 0.33.
# Use back-end of tray (largest Y = farther from camera) since tray Y center ~ 0.27.
# Aim at (tray_x - 0.10, tray_y) — back-left compartment, OR just tray center.
target_in_tray_xy = np.array([tray_center[0], tray_center[1]])
print(f"[main] target_in_tray={target_in_tray_xy}", flush=True)

quat = make_topdown_quat(0)


# ============================================================
# SUBTASK 1: Pick LEFT bowl, place in tray
# ============================================================
print("\n[SUBTASK 1] Pick LEFT bowl, place in tray", flush=True)
grasp_pos1, quat, gw1 = grasp_bowl(left_center, left_pts, yaw_deg=0)
print(f"[SUBTASK 1] grip_w={gw1:.3f}", flush=True)
stepped_lift_and_transit(grasp_pos1, target_in_tray_xy, quat, lift_z=0.40)
# Release deep in tray
descend_release_retreat(target_in_tray_xy, 0.05, quat, retreat_z=0.35)

for _ in range(5):
    get_observation()


# ============================================================
# SUBTASK 2: Pick RIGHT bowl, stack on LEFT bowl in tray
# ============================================================
print("\n[SUBTASK 2] Pick RIGHT bowl, stack", flush=True)
goto_home_joint_position()
for _ in range(2):
    get_observation()
obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

bowls2 = find_bowls(rgb, depth_img, K, E)
print(f"[main] found {len(bowls2)} bowls after subtask 1", flush=True)
for i, (c, _, _, s) in enumerate(bowls2):
    print(f"  bowl[{i}] center=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) score={s:.3f}", flush=True)

if len(bowls2) >= 2:
    bowls2_by_y = sorted(bowls2, key=lambda t: t[0][1])
    right_now = bowls2_by_y[0]
    left_in_tray = bowls2_by_y[-1]
    right_center2, right_pts2, _, _ = right_now
    left_in_tray_center, left_in_tray_pts, _, _ = left_in_tray
    left_in_tray_top_z = float(left_in_tray_pts[:, 2].max())
    stack_target_xy = np.array([left_in_tray_center[0], left_in_tray_center[1]])
elif len(bowls2) == 1:
    right_center2, right_pts2, _, _ = bowls2[0]
    left_in_tray_top_z = tray_rim_z + bowl_height
    stack_target_xy = target_in_tray_xy.copy()
else:
    raise RuntimeError("No bowls after subtask 1")

print(f"  [stack] left_in_tray_top_z={left_in_tray_top_z:.3f} target_xy={stack_target_xy}", flush=True)

grasp_pos2, quat, gw2 = grasp_bowl(right_center2, right_pts2, yaw_deg=0)
print(f"[SUBTASK 2] grip_w={gw2:.3f}", flush=True)
stepped_lift_and_transit(grasp_pos2, stack_target_xy, quat, lift_z=0.40)

# Stack release: target deep, IK plateaus higher.
# Want bowl_bottom = left_top + 0.005 → fingers = bowl_bottom + bowl_h = 0.085 + 0.064 = 0.149 → hand = 0.249
# Aim deep so IK gives ~0.22-0.25
descend_release_retreat(stack_target_xy, left_in_tray_top_z - 0.02, quat, retreat_z=0.35)

for _ in range(5):
    get_observation()
print("\n[DONE]", flush=True)
