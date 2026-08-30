"""
Task: libero_90 / KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet
Task language: "put the black bowl on top of the cabinet"
Task type: pick-and-place (metallic/silver bowl → +Y cabinet top surface, z≈0.21)

Adapted from KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet (30/30) and
KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer (15/15) — same KS4 bowl
(metallic), but cabinet top instead of drawer.

Key scene facts (seed 51 probe):
- Bowl center (0.705, -0.054, 0.024), ext (0.105, 0.109, 0.060), z_max ≈ 0.054
  Bowl is wide (~10.5cm dia) — gripper (8cm) needs rim grasp.
  "metal bowl" 0.93, "small bowl" 0.91. "black bowl" only 0.31 (UNRELIABLE).
- Cabinet top at (0.670, +0.308, 0.212), thin slab ext_z=0.008, surface_z ≈ 0.216.
  CABINET ON +Y SIDE (different from KS1 -Y side).
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


TOP_DOWN_QUAT = make_topdown_quat(0)


def select_bowl(rgb, depth_img, K, E):
    """Find the metallic/silver bowl on the table (KS4 black-bowl is metallic render)."""
    candidates = []
    # KS4 bowl renders silver — "metal bowl"/"small bowl" score >0.9; "black bowl" only ~0.3
    for prompt in ("small bowl", "metal bowl", "silver bowl", "bowl", "black bowl"):
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:6]:
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 100:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c, e = obb["center"], obb["extent"]
            if c[0] < 0.4 or c[0] > 0.95:
                continue
            if c[2] > 0.15 or c[2] < -0.05:
                continue
            if max(e[0], e[1]) > 0.20 or max(e[0], e[1]) < 0.04:
                continue
            if e[2] > 0.12 or e[2] < 0.02:
                continue
            # Bowl on counter side (Y < 0.05)
            if c[1] > 0.10:
                continue
            candidates.append((m.get("score", 0.0), c, pts, mask))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda x: -x[0])
    _, c, pts, mask = candidates[0]
    return c, pts, mask


def select_cabinet_top(rgb, depth_img, K, E):
    """Find the on-table wooden cabinet top surface (z≈0.21)."""
    candidates = []
    for prompt in ("cabinet top", "top of cabinet", "wooden cabinet top"):
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:8]:
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 200:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c, e = obb["center"], obb["extent"]
            if c[0] < 0.3 or c[0] > 1.0:
                continue
            if c[2] < 0.10:
                continue
            if e[0] > 0.5 or e[1] > 0.5 or e[2] > 0.10:
                continue
            candidates.append((m.get("score", 0.0), c, pts, mask))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda x: -x[0])
    _, c, pts, mask = candidates[0]
    return c, pts, mask


def attempt_grasp_at(grasp_xy, top_z, label, yaw_deg=0.0):
    """Lower to grasp xy, close, lift. Return (gw_after_lift, achieved_wrist_z, gripper_quat).

    Wide bowl (~10.5cm) > gripper (8cm) — must descend deep below the rim so jaws
    straddle the rim wall from inside/outside. Aim for top_z - 0.030 like KS4 drawer.
    """
    open_gripper()
    quat = make_topdown_quat(yaw_deg)

    # Pre-grasp 15cm above
    pre = np.array([grasp_xy[0], grasp_xy[1], top_z + 0.15])
    j = solve_ik(pre.tolist(), quat.tolist())
    if j is not None:
        move_to_joints(j)

    # Deep multi-pass descent. Aim for grasp_z = top_z - 0.030 (well below rim).
    grasp_z = max(top_z - 0.030, 0.005)
    descent_seq = [top_z + 0.10, top_z + 0.05, top_z + 0.02, top_z - 0.005, grasp_z, grasp_z, grasp_z, grasp_z, grasp_z]
    for tz in descent_seq:
        pos = np.array([grasp_xy[0], grasp_xy[1], tz])
        j = solve_ik(pos.tolist(), quat.tolist())
        if j is not None:
            try:
                move_to_joints(j)
            except Exception as e:
                print(f"  [{label}] descent tz={tz:.3f} failed: {e}", flush=True)
                break

    obs_d = get_observation()
    wrist_d = obs_d['robot_cartesian_pos'][:3]
    print(f"  [{label}] yaw={yaw_deg:+.0f} At grasp: wrist=[{wrist_d[0]:.3f},{wrist_d[1]:.3f},{wrist_d[2]:.3f}]", flush=True)

    close_gripper()

    obs_g = get_observation()
    gw = float(obs_g['robot_cartesian_pos'][-1])
    print(f"  [{label}] After close: gw={gw:.3f}", flush=True)

    # Lift to safe z
    lift_pos = np.array([grasp_xy[0], grasp_xy[1], 0.40])
    j = solve_ik(lift_pos.tolist(), quat.tolist())
    if j is not None:
        try:
            move_to_joints(j)
        except Exception as e:
            print(f"  [{label}] lift failed: {e}", flush=True)

    obs_l = get_observation()
    gw_l = float(obs_l['robot_cartesian_pos'][-1])
    wrist_l = obs_l['robot_cartesian_pos'][:3]
    print(f"  [{label}] After lift: gw={gw_l:.3f} wrist=[{wrist_l[0]:.3f},{wrist_l[1]:.3f},{wrist_l[2]:.3f}]", flush=True)
    return gw_l, wrist_d[2], quat


# ---------------- Main ----------------
print(f"Task: {env.handle.task_language}", flush=True)

obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]

bowl_center, bowl_pts, bowl_mask = select_bowl(rgb, depth_img, K, E)
if bowl_center is None:
    raise RuntimeError("Bowl not found")
bowl_obb = get_oriented_bounding_box_from_3d_points(bowl_pts)
bowl_top_z = float(bowl_pts[:, 2].max())
bowl_height = float(bowl_obb["extent"][2])
bowl_radius = float(max(bowl_obb["extent"][0], bowl_obb["extent"][1]) / 2.0)
print(f"[task] BOWL center=[{bowl_center[0]:.3f},{bowl_center[1]:.3f},{bowl_center[2]:.3f}] "
      f"top_z={bowl_top_z:.3f} ext=[{bowl_obb['extent'][0]:.3f},{bowl_obb['extent'][1]:.3f},{bowl_obb['extent'][2]:.3f}] "
      f"radius={bowl_radius:.3f}", flush=True)

tgt_center, tgt_pts, _ = select_cabinet_top(rgb, depth_img, K, E)
if tgt_center is None:
    raise RuntimeError("Cabinet top not found")
surface_z = float(tgt_pts[:, 2].max())
print(f"[task] CABINET TOP center=[{tgt_center[0]:.3f},{tgt_center[1]:.3f},{tgt_center[2]:.3f}] "
      f"surface_z={surface_z:.3f}", flush=True)

# ---------------- Grasp ----------------
# Try GraspNet first; if its XY is too far from OBB center (rim grasp clustered outside),
# fall back to forced rim positions at OBB radius.

grasp_poses, grasp_scores = plan_grasp(depth, K, bowl_mask)
print(f"[task] GraspNet returned {len(grasp_poses) if grasp_poses is not None else 0} grasps", flush=True)

# Bowl ~10.5cm > gripper 8cm. Lead with FORCED RIM POSITIONS (KS4 drawer recipe).
# Cabinet at +Y, so prefer -Y rim (closest to robot, away from cabinet during transit).
attempts = []
r = 0.95 * bowl_radius
attempts.append((bowl_center[0], bowl_center[1] - r, "rim_-Y"))   # Y closest to robot
attempts.append((bowl_center[0], bowl_center[1] + r, "rim_+Y"))   # Y closest to cabinet
attempts.append((bowl_center[0] - r, bowl_center[1], "rim_-X"))
attempts.append((bowl_center[0] + r, bowl_center[1], "rim_+X"))
attempts.append((bowl_center[0], bowl_center[1], "obb_center"))   # last resort

GW_SUCCESS = 0.040
gw_final = 0.0
chosen_xy = None
chosen_quat = None

for i, (gx, gy, label) in enumerate(attempts):
    print(f"\n=== Attempt {i+1} [{label}]: XY=[{gx:.3f},{gy:.3f}] ===", flush=True)
    yaw_seq = [0.0, 45.0, 90.0, -45.0]  # multi-yaw for low-Z IK robustness
    success = False
    for yaw in yaw_seq:
        gw, achieved_z, quat_used = attempt_grasp_at((gx, gy), bowl_top_z, label, yaw_deg=yaw)
        if gw >= GW_SUCCESS:
            gw_final = gw
            chosen_xy = (gx, gy)
            chosen_quat = quat_used
            success = True
            print(f"GRASP SUCCESS [{label}] yaw={yaw:+.0f}: gw={gw:.3f}", flush=True)
            break
        # Reset for next yaw
        try:
            goto_home_joint_position()
            open_gripper()
        except Exception:
            pass
    if success:
        break
    # Re-localize bowl after a failed attempt — bowl may have shifted
    if i < len(attempts) - 1:
        try:
            goto_home_joint_position()
            open_gripper()
            obs2 = get_observation()
            r2 = obs2["agentview"]
            bc2, bp2, _ = select_bowl(r2["images"]["rgb"],
                                       r2["images"]["depth"][:,:,0] if r2["images"]["depth"].ndim == 3 else r2["images"]["depth"],
                                       r2["intrinsics"], r2["pose_mat"])
            if bc2 is not None:
                dx = bc2[0] - bowl_center[0]
                dy = bc2[1] - bowl_center[1]
                if abs(dx) > 0.005 or abs(dy) > 0.005:
                    print(f"  Bowl moved: dx={dx:.3f} dy={dy:.3f} → shifting future attempts", flush=True)
                    for k in range(i + 1, len(attempts)):
                        ox, oy, ol = attempts[k]
                        attempts[k] = (ox + dx, oy + dy, ol)
                    bowl_center = bc2
                    bowl_top_z = float(bp2[:, 2].max())
        except Exception as e:
            print(f"  Re-localize failed: {e}", flush=True)

if chosen_xy is None:
    raise RuntimeError("Failed to grasp bowl after all attempts")

# Bowl-vs-gripper offset (for placement compensation in case we used rim grasp)
offset_x = bowl_center[0] - chosen_xy[0]
offset_y = bowl_center[1] - chosen_xy[1]
print(f"\n[task] Bowl offset from gripper: dx={offset_x:.3f} dy={offset_y:.3f}", flush=True)

# ---------------- Transport ----------------
# Target gripper XY = cabinet top center MINUS bowl-vs-gripper offset
# (so bowl ends up at target despite rim-grip offset)
tgt_gripper_x = tgt_center[0] - offset_x
tgt_gripper_y = tgt_center[1] - offset_y

# High lift: clear cabinet face (~0.21m) plus margin. Bowl height ~6cm.
lift_z = max(0.40, surface_z + 0.20)
print(f"[task] lift_z={lift_z:.3f}, target_gripper=[{tgt_gripper_x:.3f},{tgt_gripper_y:.3f}]", flush=True)

# Already at z≈0.40 from grasp lift — move laterally above target.
above = np.array([tgt_gripper_x, tgt_gripper_y, lift_z])
j = solve_ik(above.tolist(), chosen_quat.tolist())
if j is not None:
    move_to_joints(j)

obs_t = get_observation()
print(f"[task] above target: wrist=[{obs_t['robot_cartesian_pos'][0]:.3f},"
      f"{obs_t['robot_cartesian_pos'][1]:.3f},{obs_t['robot_cartesian_pos'][2]:.3f}] "
      f"gw={obs_t['robot_cartesian_pos'][-1]:.3f}", flush=True)

# Descend to release height: surface_z + bowl_height + small margin
# Bowl rim must clear cabinet surface; bowl bottom should land just above surface.
release_z = surface_z + bowl_height + 0.005
release_pos = np.array([tgt_gripper_x, tgt_gripper_y, release_z])
j = solve_ik(release_pos.tolist(), chosen_quat.tolist())
if j is not None:
    move_to_joints(j)
# Multi-pass descent to combat IK convergence issues
for _ in range(3):
    j = solve_ik(release_pos.tolist(), chosen_quat.tolist())
    if j is not None:
        move_to_joints(j)

obs_r = get_observation()
print(f"[task] at release: wrist=[{obs_r['robot_cartesian_pos'][0]:.3f},"
      f"{obs_r['robot_cartesian_pos'][1]:.3f},{obs_r['robot_cartesian_pos'][2]:.3f}] "
      f"gw={obs_r['robot_cartesian_pos'][-1]:.3f}", flush=True)

open_gripper()

# Settle
for _ in range(5):
    get_observation()

# Retreat upward
retreat_pos = np.array([tgt_gripper_x, tgt_gripper_y, surface_z + 0.25])
j = solve_ik(retreat_pos.tolist(), chosen_quat.tolist())
if j is not None:
    move_to_joints(j)

for _ in range(3):
    get_observation()

obs_end = get_observation()
print(f"[task] DONE. Final wrist=[{obs_end['robot_cartesian_pos'][0]:.3f},"
      f"{obs_end['robot_cartesian_pos'][1]:.3f},{obs_end['robot_cartesian_pos'][2]:.3f}]", flush=True)
