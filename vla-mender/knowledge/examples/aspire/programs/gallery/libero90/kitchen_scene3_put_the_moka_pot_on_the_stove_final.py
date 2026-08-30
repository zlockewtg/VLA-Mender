"""
KITCHEN_SCENE3 — put the moka pot on the stove
Strategy:
  - Localize moka pot ("metal coffee pot" or "coffee pot")
  - Localize stove burner ("stove burner" or "burner")
  - Top-down grasp at the lid level (z = pot top - 0.005); pot has a side handle
    so try multiple yaws to find one where gripper grips body without hitting handle.
  - Lift, transport above burner, lower onto burner (z = burner_top + 0.04 + half_pot_height).
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def localize_object(rgb, depth, K, E, prompts, min_pts=50, z_max=0.20, z_min=-0.02):
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        # try all candidates and pick one that fits in expected workspace
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:6]:
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < min_pts:
                continue
            # filter to workspace (above table, in robot reach)
            pts = pts[(pts[:, 2] > z_min) & (pts[:, 2] < z_max)]
            if len(pts) < min_pts:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c = obb["center"]
            # workspace gate: must be in front of robot
            if not (0.30 < c[0] < 1.10):
                continue
            if not (-0.45 < c[1] < 0.45):
                continue
            print(f"  '{prompt}': score={m['score']:.3f} ctr=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) ext={obb['extent']}", flush=True)
            return c, pts, mask
    return None, None, None


# --- 1. Observe ---
print(f"Task: {env.handle.task_language}", flush=True)
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]

# --- 2. Localize moka pot ---
print("Localizing moka pot...", flush=True)
moka_center, moka_pts, moka_mask = localize_object(
    rgb, depth, K, E,
    ["metal coffee pot", "coffee pot", "silver coffee pot", "moka pot", "coffee maker"],
)
if moka_center is None:
    raise RuntimeError("Moka pot not found")

moka_z_top = float(moka_pts[:, 2].max())
moka_z_bot = float(moka_pts[:, 2].min())
moka_height = moka_z_top - moka_z_bot
moka_obb = get_oriented_bounding_box_from_3d_points(moka_pts)
moka_ext = moka_obb["extent"]
print(f"  moka top_z={moka_z_top:.3f} bot_z={moka_z_bot:.3f} height={moka_height:.3f} ext={moka_ext}", flush=True)

# --- 3. Localize stove burner ---
print("Localizing stove burner...", flush=True)
burner_center, burner_pts, _ = localize_object(
    rgb, depth, K, E,
    ["stove burner", "burner", "stove grate"],
    min_pts=200, z_max=0.10,
)
if burner_center is None:
    # fallback: use stove top
    burner_center, burner_pts, _ = localize_object(
        rgb, depth, K, E,
        ["stove top", "stovetop", "stove", "gas stove"],
        min_pts=200, z_max=0.10,
    )
if burner_center is None:
    raise RuntimeError("Stove burner not found")

burner_top_z = float(burner_pts[:, 2].max())
print(f"  burner ctr=({burner_center[0]:.3f},{burner_center[1]:.3f},{burner_center[2]:.3f}) top_z={burner_top_z:.3f}", flush=True)

# --- 4. Plan grasp ---
# Use top-down grasp at lid level. Try plan_grasp; fall back to manual top-down.
SAFE_Z = max(moka_z_top + 0.20, burner_top_z + 0.30, 0.35)
grasp_z = moka_z_top - 0.01  # 1cm below the very top — grip the narrowing lid
print(f"  grasp_z={grasp_z:.3f}, safe_z={SAFE_Z:.3f}", flush=True)

# XY: use OBB center (moka pot is roughly cylindrical so OBB center is fine)
pot_x = float(moka_center[0])
pot_y = float(moka_center[1])

# Try multiple yaws to handle the side handle
goto_home_joint_position()
open_gripper()

grasp_success = False
used_quat = make_topdown_quat(0)
used_grasp_z = grasp_z

for yaw in [0, 90, 45, -45]:
    quat = make_topdown_quat(yaw_deg=yaw)
    print(f"  Try yaw={yaw}, grasp_z={grasp_z:.3f}", flush=True)
    open_gripper()

    # Pre-grasp: SAFE_Z above the pot
    j = solve_ik([pot_x, pot_y, SAFE_Z], quat.tolist())
    if j is None:
        print(f"    IK fail (safe_z)", flush=True)
        continue
    move_to_joints(j)

    # Approach: 8cm above grasp
    j = solve_ik([pot_x, pot_y, grasp_z + 0.08], quat.tolist())
    if j is None:
        print(f"    IK fail (approach)", flush=True)
        continue
    move_to_joints(j)

    # Descend to grasp height
    j = solve_ik([pot_x, pot_y, grasp_z], quat.tolist())
    if j is None:
        print(f"    IK fail (grasp)", flush=True)
        continue
    move_to_joints(j)
    close_gripper()

    # Lift slightly to verify grasp
    j = solve_ik([pot_x, pot_y, grasp_z + 0.10], quat.tolist())
    if j is not None:
        move_to_joints(j)

    # Verify by gripper width
    obs_c = get_observation()
    gw = float(obs_c["robot_cartesian_pos"][-1])
    print(f"    gripper_width={gw:.3f}", flush=True)

    # Check pot still detected in scene at original location
    rgb_c = obs_c["agentview"]["images"]["rgb"]
    d_c = obs_c["agentview"]["images"]["depth"]
    d_c_img = d_c[:, :, 0] if len(d_c.shape) == 3 else d_c
    K_c = obs_c["agentview"]["intrinsics"]
    E_c = obs_c["agentview"]["pose_mat"]

    moka_c2, moka_p2, _ = localize_object(rgb_c, d_c_img, K_c, E_c,
        ["metal coffee pot", "coffee pot"], min_pts=20)
    moved = False
    if moka_c2 is None:
        print(f"    moka pot not detected → likely grasped", flush=True)
        moved = True
    else:
        dxy = float(np.linalg.norm(np.array(moka_c2[:2]) - np.array(moka_center[:2])))
        dz = float(moka_c2[2] - moka_center[2])
        print(f"    re-localize: dxy={dxy:.3f}, dz={dz:.3f}", flush=True)
        if dz > 0.04 or dxy > 0.04:
            moved = True

    if gw > 0.02 and moved:
        print(f"  GRASP OK with yaw={yaw}", flush=True)
        grasp_success = True
        used_quat = quat
        used_grasp_z = grasp_z
        break
    else:
        print(f"  grasp failed (gw={gw:.3f}, moved={moved}), trying next yaw", flush=True)
        open_gripper()
        # Move clear before retry
        j = solve_ik([pot_x, pot_y, SAFE_Z], make_topdown_quat(0).tolist())
        if j is not None:
            move_to_joints(j)

if not grasp_success:
    print("All yaws failed; trying lower grasp_z fallback", flush=True)
    grasp_z2 = moka_z_top - 0.04
    quat2 = make_topdown_quat(0)
    open_gripper()
    j = solve_ik([pot_x, pot_y, SAFE_Z], quat2.tolist())
    if j is not None: move_to_joints(j)
    j = solve_ik([pot_x, pot_y, grasp_z2 + 0.08], quat2.tolist())
    if j is not None: move_to_joints(j)
    j = solve_ik([pot_x, pot_y, grasp_z2], quat2.tolist())
    if j is not None: move_to_joints(j)
    close_gripper()
    used_quat = quat2
    used_grasp_z = grasp_z2

# --- 5. Lift to safe ---
j = solve_ik([pot_x, pot_y, SAFE_Z], used_quat.tolist())
if j is not None:
    move_to_joints(j)
else:
    safe_q = make_topdown_quat(0)
    j = solve_ik([pot_x, pot_y, SAFE_Z], safe_q.tolist())
    if j is not None:
        used_quat = safe_q
        move_to_joints(j)

# --- 6. Transit above stove burner ---
print(f"Transit to above burner ({burner_center[0]:.3f},{burner_center[1]:.3f})", flush=True)
j = solve_ik([float(burner_center[0]), float(burner_center[1]), SAFE_Z], used_quat.tolist())
if j is None:
    used_quat = make_topdown_quat(0)
    j = solve_ik([float(burner_center[0]), float(burner_center[1]), SAFE_Z], used_quat.tolist())
if j is not None:
    move_to_joints(j)

# --- 7. Lower onto burner ---
# Pot bottom at moka_z_bot when gripper at used_grasp_z. Object bottom is
# (used_grasp_z - moka_z_bot) below gripper. Place so pot bottom is just
# above burner_top_z (burner_top_z + 0.01 to avoid interpenetration).
release_offset = used_grasp_z - moka_z_bot  # how far above pot bottom the gripper sits
# Target: pot bottom should land on burner_top_z; gripper z = burner_top_z + release_offset + small clearance
release_z = burner_top_z + release_offset + 0.005
release_z = max(release_z, burner_top_z + 0.10)  # safety floor — never go below burner+10cm
print(f"  release_offset={release_offset:.3f}, release_z={release_z:.3f}", flush=True)

# Descend in 2 steps
for inter_z in [SAFE_Z * 0.5 + release_z * 0.5, release_z]:
    j = solve_ik([float(burner_center[0]), float(burner_center[1]), float(inter_z)], used_quat.tolist())
    if j is not None:
        move_to_joints(j)

open_gripper()

# Settle
for _ in range(5):
    get_observation()

# Retreat
j = solve_ik([float(burner_center[0]), float(burner_center[1]), SAFE_Z], used_quat.tolist())
if j is not None:
    move_to_joints(j)

goto_home_joint_position()

for _ in range(10):
    get_observation()

print("DONE", flush=True)
