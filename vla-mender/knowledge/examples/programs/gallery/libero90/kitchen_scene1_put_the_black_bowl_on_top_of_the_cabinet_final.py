import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


def localize_object(rgb, depth_img, K, E, prompts):
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        best = max(masks, key=lambda d: d["score"])
        mask = best["mask"].astype(np.uint8)
        pts = mask_to_world_points(mask, depth_img, K, E)
        if pts is None or len(pts) < 10:
            continue
        center = get_oriented_bounding_box_from_3d_points(pts)["center"]
        return center, pts, mask
    return None, None, None


def select_cabinet_top_mask(rgb, depth_img, K, E):
    """Find the small cabinet's top surface (high z, modest extent)."""
    candidates = []
    for prompt in ("cabinet top", "top of cabinet", "wooden cabinet top"):
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:8]:
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 100:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            center, extent = obb["center"], obb["extent"]
            if center[0] < 0.3 or center[0] > 1.0:
                continue
            if center[2] < 0.10:
                continue
            if extent[0] > 0.5 or extent[1] > 0.5:
                continue
            if extent[2] > 0.10:
                continue
            candidates.append((m["score"], center, pts, mask))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, center, pts, mask = candidates[0]
    return center, pts, mask


def select_bowl_mask(rgb, depth_img, K, E):
    """Find the black bowl on the table."""
    candidates = []
    for prompt in ("black bowl", "dark bowl", "bowl"):
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:6]:
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 50:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            center, extent = obb["center"], obb["extent"]
            if center[0] < 0.3 or center[0] > 1.0:
                continue
            if center[2] > 0.15:
                continue
            if max(extent[0], extent[1]) > 0.20:
                continue
            if max(extent[0], extent[1]) < 0.04:
                continue
            candidates.append((m["score"], center, pts, mask))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, center, pts, mask = candidates[0]
    return center, pts, mask


# ---------------- Main ----------------
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]

bowl_center, bowl_pts, bowl_mask = select_bowl_mask(rgb, depth_img, K, E)
if bowl_center is None:
    bowl_center, bowl_pts, bowl_mask = localize_object(
        rgb, depth_img, K, E, ["black bowl", "dark bowl", "bowl"])
if bowl_center is None:
    raise RuntimeError("Black bowl not found")

bowl_obb = get_oriented_bounding_box_from_3d_points(bowl_pts)
bowl_top_z = float(bowl_pts[:, 2].max())
bowl_height = float(bowl_obb["extent"][2])

tgt_center, tgt_pts, _ = select_cabinet_top_mask(rgb, depth_img, K, E)
if tgt_center is None:
    tgt_center, tgt_pts, _ = localize_object(
        rgb, depth_img, K, E,
        ["cabinet top", "top of cabinet", "wooden cabinet top"])
if tgt_center is None:
    raise RuntimeError("Cabinet top not found")

surface_z = float(tgt_pts[:, 2].max())
print(f"[task] bowl_center={bowl_center} bowl_top_z={bowl_top_z:.3f} "
      f"tgt_center={tgt_center} surface_z={surface_z:.3f}", flush=True)

# ---------------- Grasp the bowl ----------------
grasp_poses, grasp_scores = plan_grasp(depth, K, bowl_mask)
best_grasp, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
if best_grasp is None:
    best_grasp = E @ grasp_poses[grasp_scores.argmax()]
grasp_pos, grasp_quat = decompose_transform(best_grasp)
grasp_pos = np.asarray(grasp_pos, dtype=float)
grasp_quat = np.asarray(grasp_quat, dtype=float)

# Sanity-check: if the grasp x/y is far from the bowl center, the grasp planner
# likely picked an off-center contact point on the bowl rim.  Snap to the bowl
# center to make sure the gripper actually closes around the bowl.
bowl_xy = np.array([bowl_center[0], bowl_center[1]])
dist_xy = np.linalg.norm(grasp_pos[:2] - bowl_xy)
bowl_radius = max(float(bowl_obb["extent"][0]), float(bowl_obb["extent"][1])) / 2.0
if dist_xy > 0.5 * bowl_radius:
    print(f"[task] snapping grasp xy {grasp_pos[:2]} -> bowl center {bowl_xy} "
          f"(was {dist_xy:.3f} m off, bowl_radius={bowl_radius:.3f})", flush=True)
    grasp_pos[0] = bowl_xy[0]
    grasp_pos[1] = bowl_xy[1]

# Also clamp grasp z so we go down to the bowl rim (catch the rim) but not below
# the table.  Aim for the bowl's top minus a small margin so the fingers close
# on the rim of the bowl.
target_grasp_z = bowl_top_z - 0.005   # just under the rim
if grasp_pos[2] > bowl_top_z + 0.02 or grasp_pos[2] < bowl_top_z - bowl_height:
    print(f"[task] adjusting grasp z {grasp_pos[2]:.3f} -> {target_grasp_z:.3f}", flush=True)
    grasp_pos[2] = target_grasp_z

# Use a topdown quaternion if the planner gave something extreme.
# decompose_transform's quat is wxyz.  If the implied gripper is far from
# top-down, replace it with the canonical top-down quat.
R_default = Rotation.from_quat([grasp_quat[1], grasp_quat[2], grasp_quat[3], grasp_quat[0]]).as_matrix()
gripper_z_world = R_default @ np.array([0, 0, 1])
if gripper_z_world[2] > -0.7:    # not pointing mostly downward
    print(f"[task] replacing grasp quat with top-down (z_world_z={gripper_z_world[2]:.2f})", flush=True)
    grasp_quat = make_topdown_quat(0)

open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.15)
goto_pose(grasp_pos, grasp_quat)
close_gripper()

# ---------------- Lift well above the cabinet top ----------------
clearance = 0.18
lift_z = max(grasp_pos[2] + 0.20, surface_z + clearance)

lift_pos = np.array([grasp_pos[0], grasp_pos[1], lift_z])
joints = solve_ik(lift_pos.tolist(), grasp_quat.tolist())
if joints is not None:
    move_to_joints(joints)

above = np.array([tgt_center[0], tgt_center[1], lift_z])
joints = solve_ik(above.tolist(), grasp_quat.tolist())
if joints is not None:
    move_to_joints(joints)

# Lower to release height (just above cabinet top surface).
release_pos = np.array([tgt_center[0], tgt_center[1], surface_z + 0.05])
joints = solve_ik(release_pos.tolist(), grasp_quat.tolist())
if joints is not None:
    move_to_joints(joints)

open_gripper()

# Retreat upward so the gripper does not knock the bowl off when it withdraws.
retreat_pos = np.array([tgt_center[0], tgt_center[1], surface_z + 0.20])
joints = solve_ik(retreat_pos.tolist(), grasp_quat.tolist())
if joints is not None:
    move_to_joints(joints)

for _ in range(3):
    get_observation()
