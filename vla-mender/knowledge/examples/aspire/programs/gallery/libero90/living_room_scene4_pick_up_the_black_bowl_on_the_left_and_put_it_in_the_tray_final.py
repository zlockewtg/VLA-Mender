"""
LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray
Task language: 'pick up the black bowl on the left and put it in the tray'
Task type: pick-and-place

Pick: the LEFT black bowl (small bowl, ~11cm dia, ~6.6cm tall).
  Two visually-identical bowls in scene at world Y={~0.05, ~-0.13}.
  "left" in LIBERO convention = min world-Y.
Place: into wooden tray (center ~ (0.50, 0.27), rim_z ~ 0.10).

v2: Stepped lift after grasp (3 waypoints) to avoid sudden IK joint
snap that drops the bowl. v1 lifted directly to z=0.40 → ~24% drop rate.
"""
import numpy as np
from scipy.spatial.transform import Rotation


# ============================================================
# Helpers
# ============================================================

def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


def localize_object(rgb, depth_img, K, E, prompts, min_score=0.0):
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
    """Return list of (center, pts, mask, score) for table-top bowls,
    deduped by 3D distance, sorted by world-Y ascending.
    LIBERO: left = min Y (index 0), right = max Y (index -1)."""
    candidates = []
    for prompt in ("small bowl", "black bowl", "bowl", "ceramic bowl"):
        masks = segment_sam3_text_prompt(rgb, prompt)
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
            if c[2] < 0.0 or c[2] > 0.15:
                continue
            if c[0] < 0.30 or c[0] > 0.70:
                continue
            max_xy = max(ext[0], ext[1])
            if max_xy < 0.06 or max_xy > 0.18:
                continue
            if ext[2] > 0.12:
                continue
            if any(np.linalg.norm(c[:2] - prev[0][:2]) < 0.06 for prev in candidates):
                continue
            candidates.append((c, pts, mask, m["score"]))
    candidates.sort(key=lambda t: t[0][1])  # left = min Y first
    return candidates


# ============================================================
# Step 1: Settle physics + deocclude
# ============================================================
for _ in range(3):
    open_gripper()
    close_gripper()
open_gripper()

quat_topdown = make_topdown_quat(0)
side_pos = np.array([0.4, -0.4, 0.3])
j = solve_ik(side_pos.tolist(), quat_topdown.tolist())
if j is not None:
    move_to_joints(j)
for _ in range(3):
    obs = get_observation()


# ============================================================
# Step 2: Localize bowls and tray
# ============================================================
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

bowls = find_bowls(rgb, depth_img, K, E)
print(f"[main] found {len(bowls)} bowls", flush=True)
for i, (c, _, _, s) in enumerate(bowls):
    print(f"  bowl[{i}] center=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) score={s:.3f}", flush=True)
if len(bowls) < 2:
    raise RuntimeError(f"Need 2 bowls (to disambiguate left), found {len(bowls)}")

left_center, left_pts, left_mask, left_score = bowls[0]
print(f"[main] LEFT bowl chosen at ({left_center[0]:.3f},{left_center[1]:.3f},{left_center[2]:.3f})", flush=True)

tray_center, tray_pts, _, tray_score = localize_object(
    rgb, depth_img, K, E,
    ["wooden tray", "rectangular tray", "tray", "serving tray"],
    min_score=0.30,
)
if tray_center is None:
    raise RuntimeError("Tray not found")
tray_rim_z = float(np.percentile(tray_pts[:, 2], 95))
print(f"[main] tray center=({tray_center[0]:.3f},{tray_center[1]:.3f}) rim_z={tray_rim_z:.3f} score={tray_score:.3f}", flush=True)


# ============================================================
# Step 3: Grasp left bowl (top-down)
# ============================================================
left_obb = get_oriented_bounding_box_from_3d_points(left_pts)
left_top_z = float(left_pts[:, 2].max())
grasp_xy = np.array([left_obb["center"][0], left_obb["center"][1]])
target_grasp_z = left_top_z - 0.040
quat = make_topdown_quat(0)

print(f"[grasp] target=({grasp_xy[0]:.3f},{grasp_xy[1]:.3f},{target_grasp_z:.3f}) yaw=0", flush=True)

open_gripper()
pre_pos = np.array([grasp_xy[0], grasp_xy[1], target_grasp_z + 0.15])
j = solve_ik(pre_pos.tolist(), quat.tolist())
if j is not None:
    move_to_joints(j)

grip_w = 0.0
final_grasp_z = target_grasp_z
for attempt, depth_extra in enumerate([0.0, -0.020, -0.040]):
    if attempt > 0:
        print(f"  [retry] attempt={attempt} extra_depth={depth_extra}", flush=True)
        open_gripper()
    gz = target_grasp_z + depth_extra
    mid = np.array([grasp_xy[0], grasp_xy[1], gz + 0.05])
    j = solve_ik(mid.tolist(), quat.tolist())
    if j is not None:
        move_to_joints(j)
    for _ in range(5):
        j = solve_ik([grasp_xy[0], grasp_xy[1], gz], quat.tolist())
        if j is not None:
            move_to_joints(j)
    obs_d = get_observation()
    cur = np.array(obs_d['robot_cartesian_pos'][:3])
    print(f"  [descent.{attempt}] target_z={gz:.3f} achieved hand=({cur[0]:.3f},{cur[1]:.3f},{cur[2]:.3f})", flush=True)
    close_gripper()
    obs_g = get_observation()
    grip_w = float(obs_g["robot_cartesian_pos"][7]) if len(obs_g["robot_cartesian_pos"]) > 7 else -1.0
    print(f"  [grasp.{attempt}] grip_w={grip_w:.3f}", flush=True)
    if grip_w > 0.05:
        final_grasp_z = gz
        break


# ============================================================
# Step 4: STEPPED Lift, transit, place into tray
# v2 fix: small incremental lift waypoints prevent IK joint snap
# that violently jerks the bowl out of the gripper.
# ============================================================
quat_lift = quat  # keep the same quat throughout for IK warm-state continuity

# Stepped lift in 5 waypoints (~5cm increments)
hand_after_grasp_z = float(get_observation()["robot_cartesian_pos"][2])
lift_zs = [hand_after_grasp_z + 0.05,
           hand_after_grasp_z + 0.10,
           hand_after_grasp_z + 0.18,
           0.30,
           0.40]
for z in lift_zs:
    j = solve_ik([grasp_xy[0], grasp_xy[1], z], quat_lift.tolist())
    if j is not None:
        move_to_joints(j)

obs_l = get_observation()
print(f"[lift] hand_z={obs_l['robot_cartesian_pos'][2]:.3f} grip_w={obs_l['robot_cartesian_pos'][7]:.3f}", flush=True)

# Lateral transit at lift_z=0.40 in 3 waypoints
lift_z = 0.40
cur_xy = np.array(obs_l['robot_cartesian_pos'][:2])
for k in range(1, 4):
    t = k / 3.0
    wpx = cur_xy[0] + (tray_center[0] - cur_xy[0]) * t
    wpy = cur_xy[1] + (tray_center[1] - cur_xy[1]) * t
    j = solve_ik([wpx, wpy, lift_z], quat_lift.tolist())
    if j is not None:
        move_to_joints(j)

# Lower to release height (stepped)
release_z = tray_rim_z + 0.05
for z in [0.30, 0.20, release_z]:
    j = solve_ik([tray_center[0], tray_center[1], z], quat_lift.tolist())
    if j is not None:
        move_to_joints(j)

obs_r = get_observation()
print(f"[release] hand=({obs_r['robot_cartesian_pos'][0]:.3f},{obs_r['robot_cartesian_pos'][1]:.3f},{obs_r['robot_cartesian_pos'][2]:.3f}) grip_w={obs_r['robot_cartesian_pos'][7]:.3f}", flush=True)

open_gripper()
for _ in range(3):
    get_observation()

# Retreat upward
retreat = np.array([tray_center[0], tray_center[1], release_z + 0.20])
j = solve_ik(retreat.tolist(), quat_lift.tolist())
if j is not None:
    move_to_joints(j)

goto_home_joint_position()
for _ in range(15):
    get_observation()
print("[DONE]", flush=True)
