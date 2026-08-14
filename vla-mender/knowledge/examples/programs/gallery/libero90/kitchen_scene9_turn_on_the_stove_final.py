"""KITCHEN_SCENE9_turn_on_the_stove — knob rotation via top-down grip + wrist-yaw IK.

Scene layout: knob at (0.474, 0.300), z_top=0.048. Frying pan and bowl are at -X side
of stove, well separated from knob. Stove burner at (0.62, 0.31).

Approach (wrist-yaw IK rotation, validated 19/30 for KS9 in skill registry):
  1. Localize stove knob via SAM3 ("black stove knob"/"stove knob").
  2. Top-down approach with gripper, descend to z_top - 0.005, close gripper.
  3. Apply 3-step wrist yaw sweep: +60°, +120°, -60° (net +120° CW), using solve_ik
     with rotated quaternion at the same XY position.
  4. If first attempt fails, try with different starting yaw and different sweep magnitudes.

Why wrist-yaw IK (NOT direct joint sweep):
  Direct joint sweep with j5+=2.55 (validated for KS8 turn-off at -Y knob) catastrophically
  knocks the knob OFF the stove in KS9 because the +Y knob position requires a different
  arm trajectory. solve_ik(pos, rotated_quat) keeps the wrist over the knob during rotation.
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
    """Find stove knob, return (score, mask, pts, centroid_xy, z_top) or None."""
    candidates = []
    for prompt in ["black stove knob", "stove knob", "knob"]:
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
            # KS9 stove knob is at +Y side (~0.30), in front of burner at (0.62, 0.31).
            if not (0.35 < c[0] < 0.55):
                continue
            if not (0.20 < c[1] < 0.40):
                continue
            z_top = pts[:, 2].max()
            if z_top < 0.015 or z_top > 0.10:
                continue
            candidates.append((m["score"], m["mask"], pts, c, float(z_top)))
        if candidates:
            break
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])


def grip_and_yaw_sweep(cx, cy, cz, start_yaw, sweep_yaws):
    """Top-down approach + grip + wrist-yaw IK sweep.

    cx, cy: knob XY in world.
    cz: knob top Z.
    start_yaw: initial gripper yaw (deg) for descent and grip.
    sweep_yaws: list of yaws (deg) to apply AFTER grip — gripper rotates through them in sequence.
    """
    tdq = make_topdown_quat(start_yaw)
    j = safe_solve_ik([cx, cy, cz + 0.15], tdq)
    if not safe_move(j):
        return False
    j = safe_solve_ik([cx, cy, cz + 0.05], tdq)
    if not safe_move(j):
        return False
    j = safe_solve_ik([cx, cy, cz - 0.005], tdq)
    if not safe_move(j):
        return False
    close_gripper()

    # Apply wrist-yaw rotation sequence — quaternion only, position pinned at knob center.
    for yaw_deg in sweep_yaws:
        q = make_topdown_quat(yaw_deg)
        j = safe_solve_ik([cx, cy, cz - 0.005], q)
        safe_move(j)

    open_gripper()
    j_lift = safe_solve_ik([cx, cy, cz + 0.20], make_topdown_quat(0.0))
    safe_move(j_lift)
    return True


def main():
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
        print("Knob not found, fallback geometry", flush=True)
        cx, cy, cz = 0.474, 0.300, 0.048
    else:
        score, mask, pts, centroid, cz = res
        cx, cy = float(centroid[0]), float(centroid[1])
        print(f"Knob: c=({cx:.3f},{cy:.3f}) z_top={cz:.3f} score={score:.3f}", flush=True)

    # Multiple wrist-yaw rotation strategies. Episode terminates on reward=1.
    # Each tuple: (start_yaw, [sweep_yaws]).
    # Net rotation matters for the predicate; we try positive and negative directions.
    # CW sweep: yaws decrease. CCW: yaws increase. Net rotation magnitude varies.
    configs = [
        (0.0,   [60.0, 120.0, -60.0]),    # 3-step CW net +120 (skill-validated for KS9)
        (0.0,   [-60.0, -120.0, 60.0]),   # 3-step CCW net -120
        (0.0,   [90.0, 180.0, -180.0]),   # large positive then back (full 180)
        (0.0,   [-90.0, -180.0, 180.0]),  # full 180 negative
        (30.0,  [90.0, 150.0, -30.0]),    # rotated start
        (-30.0, [60.0, 120.0, 60.0]),     # alt start
        (60.0,  [120.0, 180.0, 0.0]),     # large net forward
        (0.0,   [45.0, 90.0, 135.0, 180.0]),  # gradual increase
        (0.0,   [-45.0, -90.0, -135.0, -180.0]),  # gradual decrease
        (90.0,  [150.0, 210.0, 90.0]),    # alt start with rot
    ]

    for i, (start_yaw, sweep) in enumerate(configs):
        try:
            print(f"=== Cycle {i}: start_yaw={start_yaw:.0f}° sweep={sweep} ===", flush=True)
            grip_and_yaw_sweep(cx, cy, cz, start_yaw, sweep)
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
