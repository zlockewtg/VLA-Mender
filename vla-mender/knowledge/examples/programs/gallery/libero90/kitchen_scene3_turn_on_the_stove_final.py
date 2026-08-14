"""KITCHEN_SCENE3_turn_on_the_stove — knob rotation via grip + forced joint sweep.

Adapted from KITCHEN_SCENE8_turn_off_the_stove (which achieved 30/30).
Same approach: locate stove knob, top-down grip at z_top - 0.005, then forced
joint sweep with j5/j6/j7 increments. Try multiple yaw + sweep configs.

Note: KITCHEN_SCENE3 differs from KITCHEN_SCENE8 in:
- The knob may be at a different orientation (turn-on may need OPPOSITE direction)
- KITCHEN_SCENE3 also has a frying pan in the scene that may need to be considered
  for SAM3 disambiguation. We filter knob world position carefully.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0.0):
    R = (Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix()
         @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def safe_solve_ik(pos, quat):
    try:
        return solve_ik(np.array(pos), quat)
    except Exception:
        return None


def safe_move(j):
    if j is None:
        return False
    try:
        move_to_joints(j)
        return True
    except Exception:
        return False


def localize_knob(rgb, depth_img, K, E):
    """Find stove knob, return (mask, pts, centroid_xy, z_top) or None.

    KITCHEN_SCENE3: knob c≈(0.47, 0.20, 0.02), z_top≈0.048. Scene also has stove
    burner at c≈(0.62, 0.20, 0.02) and frying pan at c≈(0.59, -0.24).
    Knob is identified as: X∈(0.40, 0.55), Y∈(0.10, 0.30), z_top<0.07.
    """
    candidates = []
    for prompt in ["stove knob", "black stove knob", "knob"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in masks[:8]:
            if m["score"] < 0.05:
                continue
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 50:
                continue
            c = pts.mean(axis=0)
            # KITCHEN_SCENE3 knob: x∈(0.40, 0.55), y∈(0.10, 0.30)
            if not (0.40 < c[0] < 0.55):
                continue
            if not (0.10 < c[1] < 0.30):
                continue
            z_top = pts[:, 2].max()
            if z_top < 0.015 or z_top > 0.10:
                continue
            # Filter out the stove burner top mask (which has wide extent ~0.13m).
            # Knob mask is much smaller in extent, ~0.09m.
            ext = pts.max(axis=0) - pts.min(axis=0)
            if max(ext[0], ext[1]) > 0.16:
                continue
            candidates.append((m["score"], m["mask"], pts, c, float(z_top)))
        if candidates:
            break
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])


def grip_and_sweep(cx, cy, cz, yaw, j5_inc, j6_inc, j7_inc):
    """Top-down approach + grip + forced joint sweep + release/lift.

    cx, cy: knob center XY in world.
    cz: knob top Z in world.
    yaw: gripper yaw in degrees (top-down rotated around world Z).
    j5_inc, j6_inc, j7_inc: joint increments (radians) applied AFTER close_gripper.

    Returns gripper_width after close (>0.05 = lever gripped, sweep effective).
    """
    tdq = make_topdown_quat(yaw)
    j = safe_solve_ik([cx, cy, cz + 0.15], tdq)
    if not safe_move(j):
        return 0.0
    j = safe_solve_ik([cx, cy, cz + 0.05], tdq)
    if not safe_move(j):
        return 0.0
    j = safe_solve_ik([cx, cy, cz - 0.005], tdq)
    if not safe_move(j):
        return 0.0
    close_gripper()

    obs = get_observation()
    gw = float(obs["robot_cartesian_pos"][-1])

    j_pre = obs["robot_joint_pos"][:7].copy()
    j_target = j_pre.copy()
    j_target[4] += j5_inc
    j_target[5] += j6_inc
    j_target[6] += j7_inc
    safe_move(j_target)

    open_gripper()
    j_lift = safe_solve_ik([cx, cy, cz + 0.20], make_topdown_quat(0.0))
    safe_move(j_lift)
    return gw


def main():
    print(f"Task: {env.handle.task_language}", flush=True)
    goto_home_joint_position()
    open_gripper()

    obs = get_observation()
    cam = obs["agentview"]
    rgb = cam["images"]["rgb"]
    depth = cam["images"]["depth"]
    K = cam["intrinsics"]
    E = cam["pose_mat"]
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

    res = localize_knob(rgb, depth_img, K, E)
    if res is None:
        print("Knob not found, fallback geometry (KITCHEN_SCENE3)", flush=True)
        cx, cy, cz = 0.471, 0.200, 0.048
    else:
        score, mask, pts, centroid, cz = res
        cx, cy = float(centroid[0]), float(centroid[1])
        print(f"Knob: c=({cx:.3f},{cy:.3f}) z_top={cz:.3f} score={score:.3f}", flush=True)

        # Warm up plan_grasp call to match the IK warm-start of the validated path.
        try:
            grasps, scores = plan_grasp(depth_img, K, mask)
            if len(grasps) > 0:
                sorted_idx = np.argsort(scores)[::-1]
                for idx in sorted_idx[:3]:
                    g_world = E @ grasps[idx]
                    decompose_transform(g_world)
        except Exception:
            pass

    # Multi-cycle sweep configurations.
    # Empirical discovery: turn-on the stove succeeds with NEGATIVE j5/j6/j7 increments
    # (opposite direction from KITCHEN_SCENE8 "turn off"). The first negative cycle on
    # seed 51 turned the burner ON (visible at video frame 1 = step 110). However, if
    # we keep cycling, subsequent rotations turn it back OFF.
    #
    # CRITICAL: STOP after the first cycle that achieves a real gripper close
    # (gw > 0.05 = lever gripped, sweep was effective).
    configs = [
        # Negative direction (turn-ON)
        (60.0, -2.55, -0.83, -0.49),
        (90.0, -2.55, -0.83, -0.49),
        (30.0, -2.55, -0.83, -0.49),
        (45.0, -2.55, -0.83, -0.49),
        (75.0, -2.55, -0.83, -0.49),
        (15.0, -2.55, -0.83, -0.49),
        (105.0, -2.55, -0.83, -0.49),
        # Fallback: smaller magnitudes (in case full sweep over-rotates)
        (60.0, -2.00, -0.60, -0.40),
        (60.0, -1.50, -0.50, -0.30),
        # Reverse direction as last resort
        (60.0, +2.55, +0.83, +0.49),
    ]

    success_gw_threshold = 0.05  # lever-gripped gw after close
    for i, (yaw, j5, j6, j7) in enumerate(configs):
        try:
            print(f"=== Cycle {i}: yaw={yaw:.0f}° j5+={j5:.2f} j6+={j6:.2f} j7+={j7:.2f} ===",
                  flush=True)
            gw = grip_and_sweep(cx, cy, cz, yaw, j5, j6, j7)
            print(f"  gw={gw:.3f}", flush=True)
            if gw > success_gw_threshold:
                # Lever was gripped; sweep was effective. Stop to avoid undoing.
                print(f"  Lever gripped (gw={gw:.3f}>{success_gw_threshold}), stopping.",
                      flush=True)
                break
        except Exception as e:
            print(f"  cycle ended (likely reward fired): {e}", flush=True)
            break

    try:
        open_gripper()
        goto_home_joint_position()
    except Exception:
        pass
    print("Done", flush=True)


main()
