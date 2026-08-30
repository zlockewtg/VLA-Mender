"""
LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray
Task type: pick-and-place

Pick: salad dressing bottle (~11cm tall, narrow X ~3cm, wide Y ~6cm).
Place: wooden tray on the right side of the table.

Strategy:
1. Move arm aside (deocclude). Localize bottle and tray with SAM3.
2. Compute body XY centroid (inlier-filtered).
3. Gradual descent (z = 0.30 -> 0.25 -> ... -> 0.135) to keep IK in topdown
   solution branch, avoiding silent fallback to side-approach.
4. After reaching grasp z, observe achieved XY drift; if >1cm, issue an
   XY-correction (same z, small target shift) to align fingers with bottle.
5. Try yaw=90 first (close along narrower X); if grip width still empty,
   retract and try yaw=0.

The bottle has body Y-extent ~6cm (above gripper opening 8cm) and X-extent
~3cm at the cap. yaw=90 gives the most consistent grip across most seeds.

Critical: Franka workspace boundary — for bottle X<0.28, top-down IK at
z=0.135 silently falls back to side-approach. Gradual descent + XY-correction
keeps the topdown solution viable across all 30 seeds 51-80.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


def localize_object(rgb, depth, K, E, prompts, min_score=0.0):
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
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


def gradual_descent_with_correction(target_xy, target_z, quat, max_correction_iter=2):
    """Descend gradually to (target_xy, target_z) using warm-started IK.
    After reaching the bottom, apply XY corrections if achieved XY is off
    target by more than 1cm (within IK feasibility).

    Returns (achieved_x, achieved_y, achieved_tips_z).
    """
    cx, cy = target_xy
    # Multi-step descent
    descent_path = [0.30, 0.25, 0.22, 0.20, 0.18, 0.16, target_z]
    for z in descent_path:
        joints = solve_ik([cx, cy, z], quat.tolist())
        if joints is not None:
            move_to_joints(joints)

    # Read achieved
    obs = get_observation()
    cart = obs["robot_cartesian_pos"]
    tips_z = cart[2] - 0.110

    # XY correction loop: shift target to compensate for drift
    correction_x = correction_y = 0.0
    for it in range(max_correction_iter):
        drift_x = cart[0] - cx
        drift_y = cart[1] - cy
        if abs(drift_x) < 0.008 and abs(drift_y) < 0.008:
            break
        # Pre-compensate by shifting target opposite to drift
        correction_x -= drift_x
        correction_y -= drift_y
        joints = solve_ik([cx + correction_x, cy + correction_y, target_z], quat.tolist())
        if joints is not None:
            move_to_joints(joints)
        obs = get_observation()
        cart = obs["robot_cartesian_pos"]
        tips_z = cart[2] - 0.110
        print(f"[INFO] correction it={it+1}: target=({cx+correction_x:.3f},{cy+correction_y:.3f}) achieved=({cart[0]:.3f},{cart[1]:.3f}) tips_z={tips_z:.3f}", flush=True)

    return cart[0], cart[1], tips_z


# === Step 1: Move arm aside to deocclude ===
quat_topdown = make_topdown_quat(0)
side_pos = np.array([0.4, -0.4, 0.3])
joints = solve_ik(side_pos.tolist(), quat_topdown.tolist())
if joints is not None:
    move_to_joints(joints)
else:
    for fallback_pos in [[0.3, -0.4, 0.35], [0.3, 0.4, 0.35], [0.5, -0.3, 0.3]]:
        joints = solve_ik(fallback_pos, quat_topdown.tolist())
        if joints is not None:
            move_to_joints(joints)
            break

for _ in range(3):
    obs = get_observation()

# === Step 2: Localize bottle and tray ===
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

salad_prompts = ["salad dressing bottle", "small bottle", "bottle", "tall bottle",
                 "condiment bottle", "salad dressing"]
obj_center, obj_pts, obj_mask, obj_score = localize_object(
    rgb, depth, K, E, salad_prompts, min_score=0.4
)
if obj_center is None:
    raise RuntimeError("Salad dressing not found")
print(f"[INFO] salad score={obj_score:.3f} center={obj_center.tolist()}", flush=True)

tray_prompts = ["wooden tray", "tray", "serving tray"]
tgt_center, tgt_pts, _, tgt_score = localize_object(rgb, depth, K, E, tray_prompts, min_score=0.3)
if tgt_center is None:
    raise RuntimeError("Tray not found")
tray_rim_z = tgt_pts[:, 2].max()
print(f"[INFO] tray score={tgt_score:.3f} center={tgt_center.tolist()} rim_z={tray_rim_z:.3f}", flush=True)

# === Step 3: Body inlier centroid ===
body_pts = obj_pts[obj_pts[:, 2] > 0.04]
if len(body_pts) < 30:
    body_pts = obj_pts
xy = body_pts[:, :2]
cx, cy = float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))
for _ in range(3):
    dist = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    inliers = body_pts[dist < 0.04]
    if len(inliers) < 5:
        break
    cx, cy = float(inliers[:, 0].mean()), float(inliers[:, 1].mean())

inlier_z_max = float(inliers[:, 2].max()) if len(inliers) > 0 else float(body_pts[:, 2].max())
inlier_z_min = float(inliers[:, 2].min()) if len(inliers) > 0 else float(body_pts[:, 2].min())
print(f"[INFO] body=({cx:.3f},{cy:.3f}) z=[{inlier_z_min:.3f},{inlier_z_max:.3f}]", flush=True)

target_grasp_z = inlier_z_max - 0.03
target_grasp_z = max(target_grasp_z, inlier_z_min + 0.04)


def attempt_grasp_with_yaw(yaw):
    """Approach + close. Returns achieved gripper width."""
    quat = make_topdown_quat(yaw)
    open_gripper()
    achieved = gradual_descent_with_correction((cx, cy), target_grasp_z, quat)
    close_gripper()
    obs = get_observation()
    width = float(obs["robot_cartesian_pos"][7])
    return width, quat


# Try yaw=90 first (best for narrow X bottle dimension)
print(f"[INFO] === Try yaw=90 ===", flush=True)
width, quat = attempt_grasp_with_yaw(90)
print(f"[INFO] yaw=90 grasp width: {width:.3f}", flush=True)

if width < 0.20:
    # Retract and try yaw=0
    print(f"[INFO] yaw=90 failed; retract and try yaw=0", flush=True)
    open_gripper()
    # Lift up high using yaw=0 quat (more stable for high z)
    quat_lift = make_topdown_quat(0)
    for z_lift in [0.20, 0.30]:
        joints = solve_ik([cx, cy, z_lift], quat_lift.tolist())
        if joints is not None:
            move_to_joints(joints)

    # Re-localize bottle (it might have been disturbed)
    obs = get_observation()
    cam2 = obs["agentview"]
    rgb2 = cam2["images"]["rgb"]
    depth2 = cam2["images"]["depth"]
    depth_img2 = depth2[:, :, 0] if len(depth2.shape) == 3 else depth2
    obj_center2, obj_pts2, _, _ = localize_object(
        rgb2, depth2, cam2["intrinsics"], cam2["pose_mat"], salad_prompts, min_score=0.4
    )
    if obj_center2 is not None:
        body_pts2 = obj_pts2[obj_pts2[:, 2] > 0.04]
        if len(body_pts2) > 30:
            xy2 = body_pts2[:, :2]
            cx2, cy2 = float(np.median(xy2[:, 0])), float(np.median(xy2[:, 1]))
            for _ in range(3):
                dist2 = np.sqrt((xy2[:, 0] - cx2) ** 2 + (xy2[:, 1] - cy2) ** 2)
                inliers2 = body_pts2[dist2 < 0.04]
                if len(inliers2) < 5:
                    break
                cx2, cy2 = float(inliers2[:, 0].mean()), float(inliers2[:, 1].mean())
            cx, cy = cx2, cy2
            inlier_z_max = float(inliers2[:, 2].max())
            target_grasp_z = max(inlier_z_max - 0.03, body_pts2[:, 2].min() + 0.04)
            print(f"[INFO] re-localized: ({cx:.3f},{cy:.3f}) z_max={inlier_z_max:.3f}", flush=True)

    width, quat = attempt_grasp_with_yaw(0)
    print(f"[INFO] yaw=0 grasp width: {width:.3f}", flush=True)

# === Step 5: Lift gradually ===
lift_z = 0.30
for z in [target_grasp_z + 0.05, 0.20, 0.25, lift_z]:
    if z > target_grasp_z:
        joints = solve_ik([cx, cy, z], quat.tolist())
        if joints is not None:
            move_to_joints(joints)

obs = get_observation()
print(f"[INFO] after lift width={obs['robot_cartesian_pos'][7]:.3f}", flush=True)

# === Step 6: Lateral transit and place ===
joints = solve_ik([tgt_center[0], tgt_center[1], lift_z], quat.tolist())
if joints is not None:
    move_to_joints(joints)

release_pos = np.array([tgt_center[0], tgt_center[1], tray_rim_z + 0.08])
joints = solve_ik(release_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

open_gripper()
for _ in range(5):
    get_observation()
