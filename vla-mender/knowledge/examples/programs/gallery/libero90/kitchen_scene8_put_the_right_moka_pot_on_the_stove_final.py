# =====================================================================
# Fix: libero_90 / KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove
#
# Scene:
#   - 2 moka pots: silver (LEFT, lower Y) and silver/red (RIGHT, higher Y)
#   - "right moka pot" SAM3 prompt directly returns RIGHT pot (score ~0.76)
#   - Stove burner: ctr=(~0.61, ~-0.20, ~0.02), reachable
#   - Right moka pot height ~0.14m
#
# KEY INSIGHT (v3):
#   - Top-down body grasp DROPS during lift (kinematic + body shape)
#   - GRASP THE TOP-KNOB INSTEAD: SAM3 mask "top region" (Z > z_max-0.04)
#     gives a XY ~(0.622, 0.233) — slightly toward robot from body center
#   - Top-down at this XY at target_z=0.13 grips the lid+knob firmly
#   - Lift survives 25+cm without dropping (gw stable at ~0.16)
#
# Approach:
#   1. SAM3 'right moka pot' → top-score mask
#   2. Compute top-knob XY from points where Z > z_max - 0.04
#   3. Top-down grasp at target_z = z_max - 0.012 (just below very top)
#   4. Lift in stages, transport to stove, place pot base on burner
# =====================================================================
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


TOP_DOWN_QUAT = make_topdown_quat(0.0)


def localize_right_moka(rgb, depth_img, K, E):
    """Localize right moka pot. Returns (top_xy, pot_pts, pot_top_z).
    Strategy: 'right moka pot' SAM3 prompt; fallback to 'moka pot' max-Y.
    """
    masks = segment_sam3_text_prompt(rgb, "right moka pot")
    best_pts = None

    if masks:
        top = max(masks, key=lambda m: m["score"])
        pts0 = mask_to_world_points(top["mask"].astype(np.uint8), depth_img, K, E)
        if pts0 is not None and len(pts0) > 200:
            ctr0 = pts0.mean(axis=0)
            # Sanity: right moka pot is at +Y, on table
            if ctr0[1] > 0.05 and 0.4 < ctr0[0] < 0.85 and 0.0 < ctr0[2] < 0.20:
                best_pts = pts0
                print(f"  'right moka pot' direct: ctr={ctr0.round(3)} score={top['score']:.3f}", flush=True)

    if best_pts is None:
        # Fallback: 'moka pot' max-Y selection
        masks2 = segment_sam3_text_prompt(rgb, "moka pot")
        cands = []
        for m in sorted(masks2, key=lambda x: x["score"], reverse=True)[:8]:
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 200: continue
            c = pts.mean(axis=0)
            zmin, zmax = pts[:,2].min(), pts[:,2].max()
            h = zmax - zmin
            if h < 0.05 or h > 0.20: continue
            if c[2] < 0.0 or c[2] > 0.20: continue
            if c[0] < 0.4 or c[0] > 0.85: continue
            cands.append((m, pts, c))
        if not cands:
            # Last fallback: 'coffee maker'
            masks3 = segment_sam3_text_prompt(rgb, "coffee maker")
            for m in sorted(masks3, key=lambda x: x["score"], reverse=True)[:8]:
                pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
                if pts is None or len(pts) < 200: continue
                c = pts.mean(axis=0)
                zmin, zmax = pts[:,2].min(), pts[:,2].max()
                h = zmax - zmin
                if h < 0.05 or h > 0.20: continue
                if c[2] < 0.0 or c[2] > 0.20: continue
                if c[0] < 0.4 or c[0] > 0.85: continue
                cands.append((m, pts, c))
        if not cands:
            return None, None, None
        # Right = max-Y
        _, best_pts, _ = max(cands, key=lambda c: c[2][1])
        print(f"  Fallback max-Y: ctr={best_pts.mean(axis=0).round(3)}", flush=True)

    # Compute top-knob XY (where the gripper grips most reliably)
    z_max = best_pts[:, 2].max()
    top_pts = best_pts[best_pts[:, 2] > z_max - 0.04]
    if len(top_pts) < 30:
        # Fallback to all points
        top_pts = best_pts

    top_x = (top_pts[:, 0].min() + top_pts[:, 0].max()) / 2
    top_y = (top_pts[:, 1].min() + top_pts[:, 1].max()) / 2

    return np.array([top_x, top_y]), best_pts, z_max


def localize_stove_burner(rgb, depth_img, K, E):
    """Localize the stove burner. Returns (center_xy, top_z)."""
    for prompt in ["stove burner", "burner", "stove top", "stove"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks: continue
        best = max(masks, key=lambda m: m["score"])
        if best["score"] < 0.4: continue
        pts = mask_to_world_points(best["mask"].astype(np.uint8), depth_img, K, E)
        if pts is None or len(pts) < 100: continue
        ctr = pts.mean(axis=0)
        # Stove is at -Y side, low Z
        if ctr[1] > 0.05: continue
        if ctr[2] > 0.10: continue
        cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
        cy = (pts[:, 1].min() + pts[:, 1].max()) / 2
        cz = np.percentile(pts[:, 2], 95)
        print(f"  Stove '{prompt}': ctr=({cx:.3f},{cy:.3f},{cz:.3f}) score={best['score']:.3f}", flush=True)
        return np.array([cx, cy]), cz
    return None, None


# =====================================================================
# MAIN
# =====================================================================
print("=== Right moka pot on stove (v3) ===", flush=True)

obs0 = get_observation()
cam = obs0["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]

# === LOCALIZE OBJECTS (BEFORE moving to keep view clean) ===
top_xy, pot_pts, pot_top_z = localize_right_moka(rgb, depth_img, K, E)
if top_xy is None:
    raise RuntimeError("Right moka pot not found")
print(f"Right moka pot top-knob XY: {top_xy.round(3)}, top_z={pot_top_z:.3f}", flush=True)

stove_xy, stove_z = localize_stove_burner(rgb, depth_img, K, E)
if stove_xy is None:
    print("Stove not found via SAM3; using fallback", flush=True)
    stove_xy = np.array([0.613, -0.202])
    stove_z = 0.020
print(f"Stove center XY: {stove_xy.round(3)}, top_z={stove_z:.3f}", flush=True)

# === GRASP ===
gx, gy = float(top_xy[0]), float(top_xy[1])
# target_z = wrist Z target. IK clamps but the resulting pos is good for grip on top knob.
# Validated: target_z=0.13 → arm wrist_Z≈0.24, gripper closes around top knob, lifts firmly.
# Use pot_top_z - 0.012 (just below very top) — works robustly across seeds.
GRASP_Z = float(np.clip(pot_top_z - 0.012, 0.10, 0.16))

print(f"Grasp at ({gx:.3f}, {gy:.3f}, target_z={GRASP_Z:.3f})", flush=True)

open_gripper()
# Multi-step descent for IK warm-start
goto_pose([gx, gy, 0.30], TOP_DOWN_QUAT.tolist())
goto_pose([gx, gy, 0.20], TOP_DOWN_QUAT.tolist())
goto_pose([gx, gy, GRASP_Z], TOP_DOWN_QUAT.tolist())

obs_pre = get_observation()
arm_pre = np.array(obs_pre["robot_cartesian_pos"][:3])
print(f"Pre-close arm wrist: {arm_pre.round(3)}", flush=True)

close_gripper()

obs_g = get_observation()
gw = obs_g["robot_cartesian_pos"][-1]
arm_g = np.array(obs_g["robot_cartesian_pos"][:3])
print(f"After grasp: arm={arm_g.round(3)}, gw={gw:.3f}", flush=True)

# Retry once if air grasp
if gw < 0.04:
    print(f"  Retry: gw={gw:.3f} too low", flush=True)
    open_gripper()
    for retry_z_off in [-0.02, +0.02, -0.04]:
        retry_z = float(np.clip(GRASP_Z + retry_z_off, 0.08, 0.16))
        goto_pose([gx, gy, retry_z + 0.05], TOP_DOWN_QUAT.tolist())
        goto_pose([gx, gy, retry_z], TOP_DOWN_QUAT.tolist())
        close_gripper()
        obs_g = get_observation()
        gw = obs_g["robot_cartesian_pos"][-1]
        print(f"    retry_z={retry_z:.3f}: gw={gw:.3f}", flush=True)
        if gw >= 0.04:
            GRASP_Z = retry_z
            break
        open_gripper()

# === LIFT ===
TRANSIT_Z = 0.40
print(f"Lifting to TRANSIT_Z={TRANSIT_Z}...", flush=True)
for step_z in [GRASP_Z + 0.04, GRASP_Z + 0.10, GRASP_Z + 0.18, TRANSIT_Z]:
    goto_pose([gx, gy, step_z], TOP_DOWN_QUAT.tolist())

obs_l = get_observation()
gw_l = obs_l["robot_cartesian_pos"][-1]
arm_l = np.array(obs_l["robot_cartesian_pos"][:3])
print(f"After lift: arm={arm_l.round(3)}, gw={gw_l:.3f}", flush=True)

# === TRANSPORT TO STOVE ===
sx, sy = float(stove_xy[0]), float(stove_xy[1])
print(f"Transport to stove ({sx:.3f}, {sy:.3f})", flush=True)

# Multi-step horizontal traversal at TRANSIT_Z
for frac in [0.33, 0.66, 1.0]:
    wp_x = arm_l[0] * (1 - frac) + sx * frac
    wp_y = arm_l[1] * (1 - frac) + sy * frac
    goto_pose([wp_x, wp_y, TRANSIT_Z], TOP_DOWN_QUAT.tolist())

obs_t = get_observation()
gw_t = obs_t["robot_cartesian_pos"][-1]
arm_t = np.array(obs_t["robot_cartesian_pos"][:3])
print(f"Above stove: arm={arm_t.round(3)}, gw={gw_t:.3f}", flush=True)

# === LOWER TO PLACE ===
# We grasped at fingertips Z ≈ pot_top_z (~0.14). Pot bottom is at Z=0 (table).
# Pot height is ~pot_top_z. So fingertips offset above pot bottom = pot_top_z.
# To place pot bottom on stove (Z=stove_z), fingertips need to be at stove_z + pot_top_z.
# Wrist Z = fingertips + 0.10.
# Target_z (for goto_pose) is the wrist position. So target_z = stove_z + pot_top_z + 0.10.
# Add 1-3cm release clearance.
release_target_z = stove_z + pot_top_z + 0.13  # wrist clearance for release
release_target_z = float(np.clip(release_target_z, 0.18, 0.30))
print(f"Release target_z (wrist) = {release_target_z:.3f}", flush=True)

# Multi-step descent
for descend_z in [0.32, 0.27, 0.22, release_target_z]:
    if descend_z <= release_target_z:
        descend_z = release_target_z
    goto_pose([sx, sy, descend_z], TOP_DOWN_QUAT.tolist())
    if descend_z <= release_target_z + 0.001:
        break

obs_r = get_observation()
gw_r = obs_r["robot_cartesian_pos"][-1]
arm_r = np.array(obs_r["robot_cartesian_pos"][:3])
print(f"At release pos: arm={arm_r.round(3)}, gw={gw_r:.3f}", flush=True)

open_gripper()

# Settle
for _ in range(10):
    get_observation()

# Retreat upward
goto_pose([sx, sy, 0.35], TOP_DOWN_QUAT.tolist())
goto_pose([sx, sy, 0.45], TOP_DOWN_QUAT.tolist())
goto_home_joint_position()

print("Done.", flush=True)
