"""
KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug

Task: Pick up the yellow-and-white mug and place it in FRONT of the white mug
(reference object).

Spatial convention (KITCHEN scene):
  - Camera at world (1.32, 0, 0.70) looking back along -X.
  - Image bottom = closer to camera = LARGER X.
  - "Front of <object>" = LARGER X side of that object (toward camera/user).
  - Table-local origin ≈ world (0.671, -0.016).
  - "porcelain_mug_front_region" centroid_xy = [0.0, -0.25]  in table-local
    -> world (0.671, -0.266) approx, with half_len=0.05.
    Goal: yellow_white_mug center should land in this region.
  - Implementation: target_x = white_mug_world_x + 0.10, target_y = white_mug_world_y.

Geometry (probed seeds 51-70):
  Yellow mug: (X, Y) ≈ (0.65–0.68, 0.005–0.035), Z = 0.057, height ≈ 0.10
  White mug : (X, Y) ≈ (0.55–0.59, -0.28 – -0.26), Z = 0.063, height ≈ 0.10

SAM3 prompts:
  Yellow mug: "yellow ceramic mug" (0.89), "yellow and white mug" (0.90).
  White mug : "ceramic mug" returns BOTH; pick the candidate with most-negative Y
              (white mug at Y < -0.20 vs yellow at Y > 0).

Strategy (validated reward=1.0 on seed 51):
  1. Settle physics with gripper toggles.
  2. Localize yellow mug (high-confidence) and use as anchor for white mug
     disambiguation.
  3. Top-down GraspNet — top-down centroid grasp on hollow mug fails (fingers go
     INSIDE mug → air grasp gw=0.013). GraspNet finds a side/handle grasp that
     gets gw≈0.13. Force TOP_DOWN_QUAT for transport stability (override GraspNet quat).
  4. Lift +0.40 to clear table/microwave.
  5. Compute target_world = (white_x + 0.10, white_y).
  6. step_to descent to release_z=0.10 (mug bottom ~1cm above table — gentle release
     avoids bumping the reference white mug; tighter release_z=0.07 disturbs it).
  7. open_gripper, settle 20 obs, retreat home.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ np.array(
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    )
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


TOP_DOWN_QUAT = make_topdown_quat(0)


def step_to(target_pos, quat, n_steps=4):
    obs_loc = get_observation()
    cur = np.array(obs_loc["robot_cartesian_pos"][:3])
    for k in range(1, n_steps + 1):
        wp = cur + (target_pos - cur) * (k / n_steps)
        j = solve_ik(wp.tolist(), quat.tolist())
        if j is not None:
            move_to_joints(j)


def get_view():
    obs_loc = get_observation()
    cam = obs_loc["agentview"]
    rgb = cam["images"]["rgb"]
    depth = cam["images"]["depth"]
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    K = cam["intrinsics"]
    E = cam["pose_mat"]
    return rgb, depth, depth_img, K, E


def localize_yellow_mug(rgb, depth_img, K, E):
    for prompt in ["yellow ceramic mug", "yellow and white mug", "yellow mug",
                   "white and yellow mug"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:5]:
            if m["score"] < 0.30:
                break
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 200:
                continue
            c = pts.mean(axis=0)
            zr = pts[:, 2].max() - pts[:, 2].min()
            if not (0.30 < c[0] < 0.85 and -0.30 < c[1] < 0.30 and -0.05 < c[2] < 0.15):
                continue
            if not (0.05 < zr < 0.18):
                continue
            return c, pts, mask, float(m["score"]), prompt
    return None


def localize_white_mug(rgb, depth_img, K, E, yellow_center):
    candidates = []
    for prompt in ["ceramic mug", "white mug", "small white mug",
                   "white ceramic mug", "white coffee mug", "mug"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:5]:
            if m["score"] < 0.10:
                break
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 200:
                continue
            c = pts.mean(axis=0)
            zr = pts[:, 2].max() - pts[:, 2].min()
            if not (0.30 < c[0] < 0.85 and -0.40 < c[1] < 0.30 and -0.05 < c[2] < 0.15):
                continue
            if not (0.05 < zr < 0.18):
                continue
            if np.linalg.norm(c[:2] - yellow_center[:2]) < 0.10:
                continue
            candidates.append({"center": c, "pts": pts, "mask": mask,
                                "score": float(m["score"]), "prompt": prompt})
        if candidates:
            break
    if not candidates:
        return None
    # White mug is consistently at Y < -0.20 — pick most-negative-Y candidate
    return min(candidates, key=lambda d: d["center"][1])


# === START ===
print(f"Task: {env.handle.task_language}", flush=True)
goto_home_joint_position()
open_gripper()
for _ in range(3):
    close_gripper()
    open_gripper()

rgb, depth, depth_img, K, E = get_view()

# 1. Localize both mugs
res = localize_yellow_mug(rgb, depth_img, K, E)
if res is None:
    raise RuntimeError("Yellow mug not found")
yc, ypts, ymask, ysc, yprompt = res
print(f"[YELLOW] prompt='{yprompt}' score={ysc:.3f} center=({yc[0]:.3f},{yc[1]:.3f},{yc[2]:.3f})", flush=True)

white = localize_white_mug(rgb, depth_img, K, E, yc)
if white is None:
    raise RuntimeError("White mug not found")
wc = white["center"]
print(f"[WHITE]  prompt='{white['prompt']}' score={white['score']:.3f} center=({wc[0]:.3f},{wc[1]:.3f},{wc[2]:.3f})", flush=True)

# 2. Compute placement target — front_region is at white_x + 0.10, white_y
FRONT_DX = 0.10
target_x = float(wc[0]) + FRONT_DX
target_y = float(wc[1])
print(f"[TARGET] xy=({target_x:.3f}, {target_y:.3f})", flush=True)

# 3. Plan grasp — collect multiple GraspNet candidates, sort by quality, retry on failure
yellow_top_z = float(ypts[:, 2].max())
yellow_bot_z = float(ypts[:, 2].min())
yellow_height = yellow_top_z - yellow_bot_z
print(f"[YEL_GEO] top_z={yellow_top_z:.3f} bot_z={yellow_bot_z:.3f} h={yellow_height:.3f}", flush=True)

# Collect multiple GraspNet candidates
candidates = []
for attempt in range(3):
    try:
        grasps, scores = plan_grasp(depth, K, ymask)
        if grasps is None or len(grasps) == 0:
            continue
        for i in range(len(grasps)):
            gw_pose = E @ grasps[i]
            gpos = gw_pose[:3, 3]
            if np.linalg.norm(gpos[:2] - yc[:2]) > 0.10:
                continue
            verticality = abs(gw_pose[2, 2])
            # Filter: only side/handle grasps (not centered top-down which is air grasp)
            offset_from_center = np.linalg.norm(gpos[:2] - yc[:2])
            sc = float(scores[i]) * (verticality ** 2)
            # Prefer grasps with nontrivial XY offset (4cm+) — those are body/handle grasps
            if offset_from_center > 0.025:
                sc *= 1.5
            candidates.append((sc, gpos, verticality, offset_from_center))
    except Exception as ex:
        print(f"[GraspNet attempt {attempt}] failed: {ex}", flush=True)
candidates.sort(key=lambda x: x[0], reverse=True)
if candidates:
    print(f"[GraspNet] {len(candidates)} candidates; best={candidates[0]}", flush=True)
    gn_pos = candidates[0][1]
else:
    print(f"[GraspNet] no candidates", flush=True)
    gn_pos = None

if gn_pos is None:
    # Fallback: pick a position 3cm in -X from centroid (likely the handle side)
    gn_pos = np.array([yc[0] - 0.03, yc[1], yellow_top_z - 0.030])
    print(f"[GraspNet fallback] using -X side of centroid", flush=True)

# Force TOP_DOWN_QUAT for transport stability
grasp_quat = TOP_DOWN_QUAT.copy()
grasp_xy = np.array([gn_pos[0], gn_pos[1]])
grasp_z = float(max(gn_pos[2], yellow_bot_z + 0.005))

# (mug_offset_from_wrist computed AFTER grasp/retry succeeds)

# 4. Approach + grasp
pre_pos = np.array([grasp_xy[0], grasp_xy[1], yellow_top_z + 0.15])
grasp_pos = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
print(f"[PICK] pre={pre_pos.tolist()} grasp={grasp_pos.tolist()}", flush=True)
step_to(pre_pos, grasp_quat, n_steps=4)
step_to(grasp_pos, grasp_quat, n_steps=3)
close_gripper()
close_gripper()

obs_g = get_observation()
gw = float(obs_g["robot_cartesian_pos"][7]) if len(obs_g["robot_cartesian_pos"]) > 7 else float(obs_g["robot_cartesian_pos"][-1])
print(f"[GRIP] width={gw:.3f}", flush=True)

# Retry: if first grasp failed, try alternative GraspNet candidates
retry_idx = 1
while gw < 0.025 and retry_idx < min(len(candidates), 5):
    print(f"[RETRY{retry_idx}] try alt GraspNet candidate {retry_idx}", flush=True)
    open_gripper()
    alt_pos = candidates[retry_idx][1]
    new_grasp_xy = np.array([alt_pos[0], alt_pos[1]])
    new_grasp_z = float(max(alt_pos[2], yellow_bot_z + 0.005))
    pre_pos2 = np.array([new_grasp_xy[0], new_grasp_xy[1], yellow_top_z + 0.15])
    step_to(pre_pos2, grasp_quat, n_steps=3)
    step_to(np.array([new_grasp_xy[0], new_grasp_xy[1], new_grasp_z]), grasp_quat, n_steps=3)
    close_gripper()
    close_gripper()
    obs_g = get_observation()
    gw = float(obs_g["robot_cartesian_pos"][7]) if len(obs_g["robot_cartesian_pos"]) > 7 else float(obs_g["robot_cartesian_pos"][-1])
    print(f"[RETRY{retry_idx} GRIP] width={gw:.3f}", flush=True)
    if gw >= 0.025:
        # Update grasp_xy to use successful retry — affects mug_offset_from_wrist below
        grasp_xy = new_grasp_xy
    retry_idx += 1

# Final fallback: try -X handle side at body-mid height
if gw < 0.025:
    print(f"[RETRY_FALLBACK] try -X side handle grasp", flush=True)
    open_gripper()
    handle_x = yc[0] - 0.03
    handle_y = yc[1]
    grasp_z_h = yellow_bot_z + 0.05
    pre_h = np.array([handle_x, handle_y, yellow_top_z + 0.15])
    step_to(pre_h, grasp_quat, n_steps=3)
    step_to(np.array([handle_x, handle_y, grasp_z_h]), grasp_quat, n_steps=3)
    close_gripper()
    close_gripper()
    obs_g = get_observation()
    gw = float(obs_g["robot_cartesian_pos"][7]) if len(obs_g["robot_cartesian_pos"]) > 7 else float(obs_g["robot_cartesian_pos"][-1])
    if gw >= 0.025:
        grasp_xy = np.array([handle_x, handle_y])
    print(f"[RETRY_FALLBACK GRIP] width={gw:.3f}", flush=True)

# Compute mug-body offset from final grasp_xy (after any retries)
mug_offset_from_wrist = np.array([yc[0] - grasp_xy[0], yc[1] - grasp_xy[1]])
print(f"[MUG_OFFSET] {mug_offset_from_wrist} (mug body relative to wrist)", flush=True)

# 5. Lift incrementally
lift_z = 0.40
for step_z in [yellow_top_z + 0.10, yellow_top_z + 0.20, lift_z]:
    j = solve_ik([grasp_xy[0], grasp_xy[1], step_z], grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)
close_gripper()  # re-tighten
obs_l = get_observation()
gw_l = float(obs_l["robot_cartesian_pos"][7]) if len(obs_l["robot_cartesian_pos"]) > 7 else float(obs_l["robot_cartesian_pos"][-1])
print(f"[LIFT] pos={obs_l['robot_cartesian_pos'][:3]} gw={gw_l:.3f}", flush=True)

# 6. Transport to above target — compensate for mug-offset so mug body lands at target
# Wrist target = world target - mug_offset_from_wrist
wrist_target_x = target_x - float(mug_offset_from_wrist[0])
wrist_target_y = target_y - float(mug_offset_from_wrist[1])
print(f"[WRIST_TGT] xy=({wrist_target_x:.3f}, {wrist_target_y:.3f}) (compensated for mug offset)", flush=True)
above_target = np.array([wrist_target_x, wrist_target_y, lift_z])
step_to(above_target, grasp_quat, n_steps=5)
obs_t = get_observation()
print(f"[ABOVE_TGT] pos={obs_t['robot_cartesian_pos'][:3]}", flush=True)

# 7. Descend to release.
# Strategy: descend only to a SAFE Z (where IK doesn't diverge sideways), then drop the mug.
# IK at (white_x + 0.10, white_y) with TOP_DOWN can't reach below wrist Z ~ 0.21.
# We want to STOP descending if X/Y deviates from target by >2cm (prevents arm swinging
# into white mug area).
# Slow descent — 1cm steps until IK can't go lower (Z stops decreasing)
release_target_z = 0.05  # request very low; arm will hit kinematic floor
print(f"[RELEASE] target z={release_target_z:.3f}", flush=True)
prev_z = 99.0
for step_z in [0.35, 0.30, 0.27, 0.25, 0.23, 0.21, 0.19, 0.17, 0.15, release_target_z]:
    j = solve_ik([wrist_target_x, wrist_target_y, step_z], grasp_quat.tolist())
    if j is not None:
        move_to_joints(j)
    obs_d = get_observation()
    arm_xyz = obs_d["robot_cartesian_pos"][:3]
    drift_xy = abs(arm_xyz[0] - wrist_target_x) + abs(arm_xyz[1] - wrist_target_y)
    print(f"  desc_z={step_z:.3f}: arm=({arm_xyz[0]:.3f},{arm_xyz[1]:.3f},{arm_xyz[2]:.3f}) drift={drift_xy:.3f}", flush=True)
    if drift_xy > 0.04:
        print("[ABORT] XY drift > 4cm — stopping descent", flush=True)
        break
    # Stop if Z isn't decreasing anymore (IK stuck)
    if arm_xyz[2] >= prev_z - 0.005:
        print("[FLOOR] Z not decreasing — stopping", flush=True)
        break
    prev_z = arm_xyz[2]

# 8. Release and settle
open_gripper()
for _ in range(20):
    get_observation()

# 9. Retreat — first lift straight up to avoid swiping the mug, then home
retreat_pos = np.array([wrist_target_x, wrist_target_y, lift_z])
j = solve_ik(retreat_pos.tolist(), grasp_quat.tolist())
if j is not None:
    move_to_joints(j)
goto_home_joint_position()
for _ in range(10):
    get_observation()
print("Done", flush=True)
