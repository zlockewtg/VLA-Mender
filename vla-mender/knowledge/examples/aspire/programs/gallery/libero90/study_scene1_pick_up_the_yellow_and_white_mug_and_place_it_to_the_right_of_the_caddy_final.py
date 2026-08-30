"""STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy

V5: Combined approach.
- Strict handle localization first (ext_xy_max=0.10).
- If fails, looser filter + post-mask filtering to extract clean handle pts.
- Body offset compensation for placement.
- Multi-pass IK descent for release.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def find_object(rgb, depth_img, K, E, prompts, ext_z_max=None, ext_z_min=None,
                ext_xy_max=None, ext_xy_min=None, pos_z_max=None, pos_z_min=None,
                pos_x_min=None, pos_x_max=None, pos_y_min=None, pos_y_max=None,
                top=10, min_score=0.10, return_all=False):
    candidates = []
    for p in prompts:
        masks = segment_sam3_text_prompt(rgb, p)
        if not masks: continue
        for m in masks[:top]:
            score = m.get("score", 0)
            if score < min_score: continue
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 30: continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c, ext = obb["center"], obb["extent"]
            xy_size = max(ext[0], ext[1])
            if ext_z_max is not None and ext[2] > ext_z_max: continue
            if ext_z_min is not None and ext[2] < ext_z_min: continue
            if ext_xy_max is not None and xy_size > ext_xy_max: continue
            if ext_xy_min is not None and xy_size < ext_xy_min: continue
            if pos_z_max is not None and c[2] > pos_z_max: continue
            if pos_z_min is not None and c[2] < pos_z_min: continue
            if pos_x_min is not None and c[0] < pos_x_min: continue
            if pos_x_max is not None and c[0] > pos_x_max: continue
            if pos_y_min is not None and c[1] < pos_y_min: continue
            if pos_y_max is not None and c[1] > pos_y_max: continue
            candidates.append({"score": score, "prompt": p, "center": c, "ext": ext,
                               "pts": pts, "mask": mask})
    if return_all:
        return candidates
    if not candidates:
        return None
    return max(candidates, key=lambda d: d["score"])


def localize_handle_robust(rgb, depth_img, K, E, body_x, body_y):
    """Two-pass handle localization.
    Pass 1: standard find_object with ext_xy_max=0.10
    Pass 2 (fallback): get raw handle masks, filter pts to handle-only region.
    """
    # Pass 1: standard
    for p_set, ext_max in [(["mug handle", "handle"], 0.08),
                           (["mug handle", "handle", "yellow and white mug handle"], 0.10)]:
        cand = find_object(
            rgb, depth_img, K, E, p_set,
            pos_x_min=body_x - 0.10, pos_x_max=body_x + 0.10,
            pos_y_min=body_y - 0.10, pos_y_max=body_y + 0.10,
            pos_z_min=-0.05, pos_z_max=0.10,
            ext_xy_max=ext_max,
            min_score=0.30,
        )
        if cand is not None:
            return {"x": float(cand["center"][0]),
                    "y": float(cand["center"][1]),
                    "top_z": float(cand["pts"][:, 2].max()),
                    "ext": cand["ext"], "score": cand["score"], "prompt": cand["prompt"]}

    # Pass 2: get noisy handle masks; filter pts to remove body
    for p in ["mug handle", "handle"]:
        masks = segment_sam3_text_prompt(rgb, p)
        if not masks: continue
        for m in masks[:5]:
            score = m.get("score", 0)
            if score < 0.30: continue
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 30: continue
            # Filter: keep only pts where distance from mug body axis > 3cm
            d_body = np.sqrt((pts[:, 0] - body_x) ** 2 + (pts[:, 1] - body_y) ** 2)
            handle_pts = pts[d_body > 0.030]
            # Also restrict z range
            if len(handle_pts) > 30:
                z_med = np.median(handle_pts[:, 2])
                handle_pts = handle_pts[(handle_pts[:, 2] < z_med + 0.04) & (handle_pts[:, 2] > z_med - 0.04)]
            if len(handle_pts) < 20: continue
            obb = get_oriented_bounding_box_from_3d_points(handle_pts)
            c, ext = obb["center"], obb["extent"]
            if max(ext[0], ext[1]) > 0.10: continue
            return {"x": float(c[0]), "y": float(c[1]),
                    "top_z": float(handle_pts[:, 2].max()),
                    "ext": ext, "score": score, "prompt": f"{p}+filter"}
    return None


# Step 1: Physics settle
for _ in range(3):
    open_gripper(); close_gripper()
open_gripper()

obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

# Step 2: Localize mug
mug = find_object(
    rgb, depth_img, K, E,
    ["yellow and white mug", "white and yellow mug", "yellow mug",
     "white mug", "ceramic mug", "coffee mug", "mug"],
    pos_x_min=0.45, pos_x_max=0.85,
    pos_y_min=-0.30, pos_y_max=0.30,
    pos_z_min=-0.10, pos_z_max=0.20,
    ext_xy_min=0.06, ext_xy_max=0.20,
    min_score=0.40,
)
if mug is None:
    raise RuntimeError("Mug not found")
mug_pts = mug["pts"]
mug_top = float(mug_pts[:, 2].max())
mug_bot = float(mug_pts[:, 2].min())
mug_height = mug_top - mug_bot

body_pts = mug_pts[(mug_pts[:, 2] > mug_bot + 0.04) & (mug_pts[:, 2] < mug_bot + 0.10)]
if len(body_pts) > 5:
    body_x = float(np.median(body_pts[:, 0]))
    body_y = float(np.median(body_pts[:, 1]))
else:
    body_x = float(mug["center"][0])
    body_y = float(mug["center"][1])
print(f"[MUG] body=({body_x:.3f},{body_y:.3f}) z=[{mug_bot:.3f},{mug_top:.3f}] h={mug_height:.3f}")

# Step 3: Localize handle (robust)
handle = localize_handle_robust(rgb, depth_img, K, E, body_x, body_y)
if handle is None:
    raise RuntimeError("Handle not found")
handle_x, handle_y, handle_top = handle["x"], handle["y"], handle["top_z"]
body_offset = np.array([body_x - handle_x, body_y - handle_y])
print(f"[HANDLE] prompt='{handle['prompt']}' score={handle['score']:.3f}")
print(f"[HANDLE] xy=({handle_x:.3f},{handle_y:.3f}) top={handle_top:.3f} ext={list(handle['ext'])}")
print(f"[BODY_OFFSET] dx={body_offset[0]:.3f} dy={body_offset[1]:.3f}")

# Step 4: Caddy
caddy = find_object(
    rgb, depth_img, K, E,
    ["desk organizer", "wooden organizer", "pencil caddy"],
    ext_xy_min=0.25, ext_xy_max=0.60,
    pos_x_min=0.20, pos_x_max=0.55,
    pos_y_min=-0.40, pos_y_max=0.20,
    min_score=0.40,
)
if caddy is None:
    raise RuntimeError("Caddy not found")
caddy_pts = caddy["pts"]
caddy_x_med = float(np.percentile(caddy_pts[:, 0], 50))
caddy_y_max = float(np.percentile(caddy_pts[:, 1], 95))
caddy_z_top = float(np.percentile(caddy_pts[:, 2], 95))
print(f"[CADDY] x_med={caddy_x_med:.3f} y_max={caddy_y_max:.3f}")


# Step 5: Handle grasp + grip-width retry
def attempt_grasp(yaw_deg, hx, hy, h_top, n_descents=6):
    quat = make_topdown_quat(yaw_deg)
    open_gripper()
    j = solve_ik([hx, hy, h_top + 0.15], quat.tolist())
    if j is not None: move_to_joints(j)
    for _ in range(n_descents):
        j = solve_ik([hx, hy, h_top - 0.003], quat.tolist())
        if j is not None: move_to_joints(j)
    close_gripper()
    obs_c = get_observation()
    gw = float(obs_c['robot_cartesian_pos'][7])
    return quat, gw


def stable_grip_test(hx, hy, h_top, quat, lift_amt=0.25):
    """Lift to h_top + lift_amt and measure grip width — exposes weak grips."""
    j = solve_ik([hx, hy, h_top + lift_amt], quat.tolist())
    if j is not None: move_to_joints(j)
    obs_l = get_observation()
    return float(obs_l['robot_cartesian_pos'][7])


# Try yaw=90 first (gripper closes along world-Y, fits handle thickness)
quat, gw0 = attempt_grasp(90, handle_x, handle_y, handle_top)
print(f"[GRASP-90] gw={gw0:.3f}")

# Early-out: if gw > 0.23 at close, gripper likely caught body wall via handle gap.
# Retry handle grasp slightly offset to ensure clean handle bar grip.
if gw0 > 0.23:
    open_gripper()
    # Retry at handle XY but slightly more outward (away from body)
    body_dir = np.array([handle_x - body_x, handle_y - body_y])
    body_dir = body_dir / (np.linalg.norm(body_dir) + 1e-6)
    hx_retry = handle_x + 0.005 * body_dir[0]
    hy_retry = handle_y + 0.005 * body_dir[1]
    quat, gw0 = attempt_grasp(90, hx_retry, hy_retry, handle_top)
    print(f"[GRASP-90-retry-out] gw={gw0:.3f}")
    if gw0 < 0.23 and gw0 > 0.10:
        handle_x, handle_y = hx_retry, hy_retry

# Aggressive lift test — exposes weak grip before transit
gw_lift = stable_grip_test(handle_x, handle_y, handle_top, quat, lift_amt=0.25)
print(f"[LIFT_LOW-90] grip={gw_lift:.3f}")

# Stable grip is in range [0.15, 0.22]: that's "handle bar" grip.
# gw < 0.15 = empty/handle slip; gw > 0.22 = body wedge (drops on lift)
if gw_lift < 0.15 or gw_lift > 0.22:
    # Lower back to grasp z, retry yaw=0
    j = solve_ik([handle_x, handle_y, handle_top + 0.05], quat.tolist())
    if j is not None: move_to_joints(j)
    open_gripper()
    quat, gw0 = attempt_grasp(0, handle_x, handle_y, handle_top)
    print(f"[GRASP-0] gw={gw0:.3f}")
    gw_lift = stable_grip_test(handle_x, handle_y, handle_top, quat, lift_amt=0.25)
    print(f"[LIFT_LOW-0] grip={gw_lift:.3f}")

if gw_lift < 0.10 or gw_lift > 0.25:
    j = solve_ik([handle_x, handle_y, handle_top + 0.05], quat.tolist())
    if j is not None: move_to_joints(j)
    open_gripper()
    quat, gw0 = attempt_grasp(45, handle_x, handle_y, handle_top)
    print(f"[GRASP-45] gw={gw0:.3f}")
    gw_lift = stable_grip_test(handle_x, handle_y, handle_top, quat, lift_amt=0.25)
    print(f"[LIFT_LOW-45] grip={gw_lift:.3f}")

# Step 6: Lift to safe transit z
lift_z = 0.40
j = solve_ik([handle_x, handle_y, lift_z], quat.tolist())
if j is not None: move_to_joints(j)
obs_lift = get_observation()
print(f"[LIFT] hand={obs_lift['robot_cartesian_pos'][:3]} grip={obs_lift['robot_cartesian_pos'][7]:.3f}")

# Step 7: Plan placement
target_body_x = float(caddy_x_med)
target_body_y = float(caddy_y_max + 0.10)
gripper_target_x = target_body_x - body_offset[0]
gripper_target_y = target_body_y - body_offset[1]
gripper_target_y = min(gripper_target_y, 0.20)
print(f"[TARGET] body=({target_body_x:.3f},{target_body_y:.3f}) gripper=({gripper_target_x:.3f},{gripper_target_y:.3f})")

# Step 8: Mid-waypoint then above
mid_x = 0.5 * (handle_x + gripper_target_x)
mid_y = 0.5 * (handle_y + gripper_target_y)
j = solve_ik([mid_x, mid_y, lift_z], quat.tolist())
if j is not None: move_to_joints(j)

j = solve_ik([gripper_target_x, gripper_target_y, lift_z], quat.tolist())
if j is not None: move_to_joints(j)
obs_above = get_observation()
print(f"[ABOVE] hand={obs_above['robot_cartesian_pos'][:3]} grip={obs_above['robot_cartesian_pos'][7]:.3f}")

# Step 9: Multi-pass descent
finger_to_mug_bot = handle_top - mug_bot
target_hand_z = max(0.20, 0.005 + 0.10 + finger_to_mug_bot)

descent_zs = [0.35, 0.30, 0.25, 0.22, target_hand_z]
for z in descent_zs:
    for _ in range(3):
        j = solve_ik([gripper_target_x, gripper_target_y, z], quat.tolist())
        if j is not None: move_to_joints(j)

obs_pre = get_observation()
print(f"[PRE_RELEASE] hand={obs_pre['robot_cartesian_pos'][:3]} grip={obs_pre['robot_cartesian_pos'][7]:.3f}")

# Step 10: Release + settle
open_gripper()
for _ in range(6): get_observation()
open_gripper()
for _ in range(6): get_observation()

# Step 11: Retreat
j = solve_ik([gripper_target_x, gripper_target_y, 0.45], quat.tolist())
if j is not None: move_to_joints(j)
for _ in range(3): get_observation()

home_pos = [0.30, -0.40, 0.50]
j = solve_ik(home_pos, quat.tolist())
if j is not None: move_to_joints(j)
for _ in range(10): get_observation()

obs_final = get_observation()
print(f"[FINAL] hand={obs_final['robot_cartesian_pos'][:3]}")
