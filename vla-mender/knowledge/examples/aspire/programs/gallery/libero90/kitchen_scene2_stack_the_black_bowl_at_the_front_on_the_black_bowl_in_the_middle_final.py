"""
KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle

Strategy:
- 3 black bowls in row arrangement (along X axis).
- "Front" = max X. "Middle" = mid X. "Back" = min X.
- GraspNet grasp on front bowl rim, lift, transit through home_joint_position to avoid IK singularities,
  then approach above middle bowl, descend & release with XY correction for grasp offset.
"""
import numpy as np
from scipy.spatial.transform import Rotation


TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, top-down


def get_cam(obs):
    cam = obs["agentview"]
    rgb = cam["images"]["rgb"]
    depth = cam["images"]["depth"]
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    return rgb, depth_img, cam["intrinsics"], cam["pose_mat"]


def detect_three_bowls(rgb, depth_img, K, E):
    """Find black bowls in scene; return up to 3 unique candidates by score."""
    candidates = []
    for prompt in ["black bowl", "bowl"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:10]:
            if m["score"] < 0.10:
                continue
            mask = m["mask"].astype(np.uint8)
            n_pix = int(mask.sum())
            if n_pix < 800 or n_pix > 8000:
                continue
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 100:
                continue
            ctr = np.median(pts, axis=0)
            if not (0.30 < ctr[0] < 0.95 and -0.40 < ctr[1] < 0.40 and -0.05 < ctr[2] < 0.10):
                continue
            candidates.append((ctr, pts, mask, float(m["score"])))
        if len(candidates) >= 3:
            break

    if not candidates:
        return []
    unique = []
    for ctr, pts, mask, score in sorted(candidates, key=lambda c: c[3], reverse=True):
        too_close = any(np.linalg.norm(ctr[:2] - u[0][:2]) < 0.06 for u in unique)
        if not too_close:
            unique.append((ctr, pts, mask, score))
        if len(unique) == 3:
            break
    return unique


def safe_move(pos, quat):
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat, dtype=float)
    joints = solve_ik(pos.tolist(), quat.tolist())
    if joints is not None:
        move_to_joints(joints)
        return True
    return False


def get_ee_pos(obs):
    return np.array(obs["robot_cartesian_pos"][:3])


def get_gripper_norm(obs):
    return float(obs["robot_cartesian_pos"][7])


def run():
    obs = get_observation()
    rgb, depth_img, K, E = get_cam(obs)

    bowls = detect_three_bowls(rgb, depth_img, K, E)
    print(f"Found {len(bowls)} bowls:", flush=True)
    for i, (c, p, m, s) in enumerate(bowls):
        z90 = float(np.percentile(p[:, 2], 90))
        print(f"  bowl {i}: ctr=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) z90={z90:.3f} score={s:.3f}", flush=True)

    if len(bowls) < 2:
        print("ERROR: need at least 2 bowls", flush=True)
        return

    xs = np.array([b[0][0] for b in bowls])
    ys = np.array([b[0][1] for b in bowls])
    sort_axis = 0 if xs.ptp() >= ys.ptp() else 1
    print(f"Sort axis={sort_axis} (X spread={xs.ptp():.3f}, Y spread={ys.ptp():.3f})", flush=True)
    bowls.sort(key=lambda b: b[0][sort_axis])

    middle_bowl = bowls[1] if len(bowls) >= 3 else bowls[0]
    front_bowl = bowls[-1]

    front_ctr, front_pts, front_mask, _ = front_bowl
    middle_ctr, middle_pts, middle_mask, _ = middle_bowl
    front_top_z = float(np.percentile(front_pts[:, 2], 90))
    middle_top_z = float(np.percentile(middle_pts[:, 2], 90))
    print(f"FRONT (pick): ({front_ctr[0]:.3f},{front_ctr[1]:.3f},{front_ctr[2]:.3f}) top_z={front_top_z:.3f}", flush=True)
    print(f"MIDDLE (place): ({middle_ctr[0]:.3f},{middle_ctr[1]:.3f},{middle_ctr[2]:.3f}) top_z={middle_top_z:.3f}", flush=True)

    open_gripper()
    quat = TOP_DOWN_QUAT.copy()

    # ---- Pre-grasp ----
    safe_move([front_ctr[0], front_ctr[1], 0.30], quat)

    # ---- GraspNet ----
    grasp_pos = None
    try:
        gposes, gscores = plan_grasp(obs["agentview"]["images"]["depth"], K, front_mask)
        if gposes is not None and len(gposes) > 0:
            best_world, _ = select_top_down_grasp(gposes, gscores, E)
            if best_world is None:
                best_world = E @ gposes[gscores.argmax()]
            gp = best_world[:3, 3]
            if np.linalg.norm(gp[:2] - front_ctr[:2]) < 0.08:
                grasp_pos = gp.copy()
                print(f"GraspNet grasp: {grasp_pos}", flush=True)
    except Exception as e:
        print(f"GraspNet fail: {e}", flush=True)

    if grasp_pos is None:
        grasp_z = max(front_top_z - 0.005, 0.005)
        grasp_pos = np.array([front_ctr[0], front_ctr[1], grasp_z])
        print(f"Centroid fallback: {grasp_pos}", flush=True)

    grasp_offset = grasp_pos[:2] - front_ctr[:2]
    print(f"Grasp XY offset (arm-bowl): ({grasp_offset[0]:.3f},{grasp_offset[1]:.3f})", flush=True)

    # ---- Approach + close ----
    safe_move([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.10], quat)
    safe_move([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.04], quat)
    safe_move(grasp_pos, quat)

    close_gripper()
    obs2 = get_observation()
    g_norm = get_gripper_norm(obs2)
    print(f"After close: gripper_norm={g_norm:.3f}", flush=True)
    if g_norm < 0.02:
        print("WARN: bowl not held (gripper fully closed)", flush=True)

    # ---- Lift straight up ----
    safe_move([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.06], quat)
    safe_move([grasp_pos[0], grasp_pos[1], 0.30], quat)

    # ---- Re-detect held bowl in air to refine grasp_offset ----
    obs_air = get_observation()
    arm_air = get_ee_pos(obs_air)
    rgb_a = obs_air["agentview"]["images"]["rgb"]
    d_a_full = obs_air["agentview"]["images"]["depth"]
    d_a = d_a_full[:, :, 0] if len(d_a_full.shape) == 3 else d_a_full
    K_a = obs_air["agentview"]["intrinsics"]
    E_a = obs_air["agentview"]["pose_mat"]
    held_offset = grasp_offset.copy()
    try:
        masks_held = segment_sam3_text_prompt(rgb_a, "black bowl")
        for m in sorted(masks_held, key=lambda d: d["score"], reverse=True)[:6]:
            if m["score"] < 0.15:
                continue
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, d_a, K_a, E_a)
            if pts is None or len(pts) < 50:
                continue
            # Held bowl is elevated (Z>0.10)
            elev = pts[pts[:, 2] > 0.10]
            if len(elev) < 30:
                continue
            held_ctr = np.median(elev, axis=0)
            d_to_arm = np.linalg.norm(held_ctr[:2] - arm_air[:2])
            if d_to_arm < 0.15:
                held_offset = arm_air[:2] - held_ctr[:2]
                print(f"Held bowl re-detected: ctr={held_ctr}, refined offset=({held_offset[0]:.3f},{held_offset[1]:.3f})", flush=True)
                break
    except Exception as e:
        print(f"Held re-detect skipped: {e}", flush=True)

    # ---- Transit via home (avoid high-Y singularity) ----
    print("Transit via home", flush=True)
    goto_home_joint_position()

    # ---- Approach above middle bowl with XY offset correction (use refined held_offset) ----
    target_x = middle_ctr[0] + held_offset[0]
    target_y = middle_ctr[1] + held_offset[1]
    above = np.array([target_x, target_y, 0.30])
    print(f"Move above middle (corrected): {above}", flush=True)
    safe_move(above, quat)
    obs_a = get_observation()
    ee_a = get_ee_pos(obs_a)
    print(f"  EE after approach={ee_a}", flush=True)

    # If IK overshot Z, retry without the X correction (just direct above middle)
    if ee_a[2] > 0.40 or abs(ee_a[2] - 0.30) > 0.05:
        print("Retry with home_reset", flush=True)
        goto_home_joint_position()
        # Less aggressive approach
        safe_move([target_x - 0.05, target_y, 0.30], quat)
        safe_move([target_x, target_y, 0.25], quat)

    # ---- Re-detect held bowl AT placement position to refine offset post-transit ----
    obs_above = get_observation()
    arm_above = get_ee_pos(obs_above)
    rgb_b = obs_above["agentview"]["images"]["rgb"]
    d_b_full = obs_above["agentview"]["images"]["depth"]
    d_b = d_b_full[:, :, 0] if len(d_b_full.shape) == 3 else d_b_full
    K_b = obs_above["agentview"]["intrinsics"]
    E_b = obs_above["agentview"]["pose_mat"]
    try:
        masks_held2 = segment_sam3_text_prompt(rgb_b, "black bowl")
        for m in sorted(masks_held2, key=lambda d: d["score"], reverse=True)[:6]:
            if m["score"] < 0.15:
                continue
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, d_b, K_b, E_b)
            if pts is None or len(pts) < 50:
                continue
            elev = pts[pts[:, 2] > 0.10]
            if len(elev) < 30:
                continue
            held_ctr2 = np.median(elev, axis=0)
            d_to_arm = np.linalg.norm(held_ctr2[:2] - arm_above[:2])
            if d_to_arm < 0.15:
                # Compute true delta needed: bowl currently at held_ctr2, arm at arm_above
                # We want bowl at middle_ctr — so move arm by (middle_ctr - held_ctr2)
                delta_xy = middle_ctr[:2] - held_ctr2[:2]
                target_x = arm_above[0] + delta_xy[0]
                target_y = arm_above[1] + delta_xy[1]
                print(f"Post-transit held: ctr={held_ctr2}, arm={arm_above[:2]}, delta_to_target={delta_xy}, new_target=({target_x:.3f},{target_y:.3f})", flush=True)
                # Move to corrected position
                safe_move([target_x, target_y, 0.30], quat)
                break
    except Exception as e:
        print(f"Above re-detect skipped: {e}", flush=True)

    # ---- Multi-step descent ----
    # IK clamps Z to ~0.27 due to high-X kinematic floor; ask for low Z to push it down.
    # Final EE_z ≈ 0.27 → fingertips at 0.17 → bowl drops from ~13cm above rim. Empirically OK.
    descent_zs = [0.25, 0.20, 0.16, 0.12]
    last_ee_z = None
    for z in descent_zs:
        ok = safe_move([target_x, target_y, z], quat)
        obs_d = get_observation()
        ee_d = get_ee_pos(obs_d)
        print(f"  descend Z={z:.3f} → EE_z={ee_d[2]:.3f} EE_xy=({ee_d[0]:.3f},{ee_d[1]:.3f}) ok={ok}", flush=True)
        if last_ee_z is not None and abs(ee_d[2] - last_ee_z) < 0.005:
            print(f"  IK plateau at Z={ee_d[2]:.3f} — stopping", flush=True)
            break
        last_ee_z = ee_d[2]

    open_gripper()
    print("Released", flush=True)
    for _ in range(15):
        get_observation()

    safe_move([target_x, target_y, 0.32], quat)
    goto_home_joint_position()


run()
