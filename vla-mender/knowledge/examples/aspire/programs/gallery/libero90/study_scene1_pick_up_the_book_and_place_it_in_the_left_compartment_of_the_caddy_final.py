"""
STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy
Task type: pick-and-place

Pick: book — standing upright on table (~16cm wide x 14cm tall x 3.6cm thick).
      OBB center near (0.53, +0.15, 0.04), z range -0.03 to 0.11 (vertical).
Place: LEFT compartment of a desk organizer / caddy with 3 compartments along world-y.
       Leftmost in agentview = lowest world-y (~ -0.28).

Strategy (copied verbatim from SS2/SS3 left-compartment 30/30 + 29/30 solutions):
  - Grasp top of book: top_z - 0.010 (top spine grip).
  - yaw=0: fingers close along world-y, gripping book's thickness.
  - Sort compartments by world-y ascending; index 0 = leftmost.
  - Release: above leftmost compartment, descend to just above rim, open.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


def find_object(rgb, depth_img, K, E, prompts, ext_z_max=None, ext_z_min=None,
                ext_xy_max=None, ext_xy_min=None, pos_z_max=None, pos_z_min=None,
                pos_x_min=None, pos_x_max=None, pos_y_min=None, pos_y_max=None,
                top=10, min_score=0.10, return_all=False, depth_filter=False):
    """Try each prompt, return best (or all) candidate(s) matching geometry filters.
    depth_filter: if True, also store a 'pts_filt' with points within ±4cm of median X
                  (front face of upright object, removes mask-edge depth leakage).
                  Geometry filters are applied to the filtered pts; original pts saved as 'pts_full'.
    """
    candidates = []
    for p in prompts:
        masks = segment_sam3_text_prompt(rgb, p)
        if not masks:
            continue
        for m in masks[:top]:
            score = m.get("score", 0)
            if score < min_score:
                continue
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 50:
                continue
            pts_full = pts.copy()
            if depth_filter:
                # Keep only points near median X (closest to camera = front face of book)
                med_x = np.median(pts[:, 0])
                keep = np.abs(pts[:, 0] - med_x) < 0.04
                if keep.sum() < 30:
                    keep = np.abs(pts[:, 0] - med_x) < 0.06
                pts = pts[keep]
                if len(pts) < 30:
                    continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c, ext = obb["center"], obb["extent"]
            ext_z, xy_size = ext[2], max(ext[0], ext[1])
            if ext_z_max is not None and ext_z > ext_z_max:
                continue
            if ext_z_min is not None and ext_z < ext_z_min:
                continue
            if ext_xy_max is not None and xy_size > ext_xy_max:
                continue
            if ext_xy_min is not None and xy_size < ext_xy_min:
                continue
            if pos_z_max is not None and c[2] > pos_z_max:
                continue
            if pos_z_min is not None and c[2] < pos_z_min:
                continue
            if pos_x_min is not None and c[0] < pos_x_min:
                continue
            if pos_x_max is not None and c[0] > pos_x_max:
                continue
            if pos_y_min is not None and c[1] < pos_y_min:
                continue
            if pos_y_max is not None and c[1] > pos_y_max:
                continue
            candidates.append({"score": score, "prompt": p, "center": c, "ext": ext,
                                "pts": pts, "pts_full": pts_full, "mask": mask, "box": m.get("box")})
    if return_all:
        return candidates
    if not candidates:
        return None
    return max(candidates, key=lambda d: d["score"])


def dedupe_3d(cands, radius=0.05):
    cands_sorted = sorted(cands, key=lambda d: -d["score"])
    kept = []
    for c in cands_sorted:
        if any(np.linalg.norm(c["center"][:2] - k["center"][:2]) < radius for k in kept):
            continue
        kept.append(c)
    return kept


# === Step 1: Settle physics ===
for _ in range(3):
    open_gripper()
    close_gripper()
open_gripper()

obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

# === Step 2: Localize book ===
# In SS1 the book sits at y≈+0.15 (right of camera). Widen pos_y_max to 0.25.
book_prompts = ["book", "novel", "binder", "black book",
                "black object", "small black box", "rectangular object",
                "thin book", "textbook"]
book_cands = find_object(
    rgb, depth_img, K, E, book_prompts,
    ext_xy_min=0.06, ext_xy_max=0.20,    # tight again now that depth_filter is on
    pos_x_min=0.40, pos_x_max=0.75,
    pos_y_min=-0.25, pos_y_max=0.35,
    pos_z_min=-0.05, pos_z_max=0.15,
    min_score=0.25, return_all=True,
    depth_filter=True,                    # filter mask edge leakage onto background
)
book_cands = dedupe_3d(book_cands, radius=0.06)
if not book_cands:
    raise RuntimeError("Book not found")
book = max(book_cands, key=lambda d: d["score"])
book_center = book["center"]
book_pts = book["pts"]
book_top_z = float(book_pts[:, 2].max())
book_bottom_z = float(book_pts[:, 2].min())
book_height = book_top_z - book_bottom_z
print(f"[BOOK] prompt='{book['prompt']}' score={book['score']:.3f}")
print(f"[BOOK] center={book_center.tolist()} ext={book['ext'].tolist()}")
print(f"[BOOK] top_z={book_top_z:.3f} bottom_z={book_bottom_z:.3f} height={book_height:.3f}")

book_xy = book_center[:2]

# === Step 2b: Compute book yaw via PCA on top edge of book ===
# Strategy: take points near book TOP (within 2cm of top_z), use FULL pts (mask projection).
# Top of upright book = a horizontal line along book WIDTH (visible from camera).
# PCA: long axis = book width direction.
# Gripper X axis (perpendicular to fingers) should align with long axis.
# At yaw=0, gripper X = world X. So yaw_deg = atan2(long_y, long_x) (mod 180).
book_pts_full = book["pts_full"]  # unfiltered for orientation
top_pts = book_pts_full[book_pts_full[:, 2] > book_top_z - 0.02]
if len(top_pts) < 20:
    top_pts = book_pts_full[book_pts_full[:, 2] > book_top_z - 0.04]
xy_top = top_pts[:, :2]
xy_centered = xy_top - xy_top.mean(axis=0, keepdims=True)
cov = np.cov(xy_centered.T)
evals, evecs = np.linalg.eigh(cov)
long_axis = evecs[:, -1]
long_angle_deg = np.degrees(np.arctan2(long_axis[1], long_axis[0]))
# Normalize to [-90, 90] (axis direction ambiguous)
yaw_deg = long_angle_deg
while yaw_deg > 90:
    yaw_deg -= 180
while yaw_deg < -90:
    yaw_deg += 180
print(f"[BOOK_YAW] n_top={len(top_pts)} long_angle={long_angle_deg:.1f}° → yaw_deg={yaw_deg:.1f}°")
print(f"[BOOK_YAW] eigenvalues sqrt={np.sqrt(np.maximum(evals, 0)).round(3).tolist()}")

# === Step 3: Localize caddy compartments ===
masks = segment_sam3_text_prompt(rgb, "compartment")
comps_raw = []
for m in masks[:10]:
    score = m.get("score", 0)
    if score < 0.50:
        continue
    mask = m["mask"].astype(np.uint8)
    pts = mask_to_world_points(mask, depth_img, K, E)
    if pts is None or len(pts) < 100:
        continue
    obb = get_oriented_bounding_box_from_3d_points(pts)
    c, ext = obb["center"], obb["extent"]
    if c[0] < 0.20 or c[0] > 0.55:
        continue
    if c[1] < -0.45 or c[1] > 0.10:
        continue
    if max(ext[0], ext[1]) > 0.30:
        continue
    if max(ext[0], ext[1]) < 0.08:
        continue
    comps_raw.append({"score": score, "center": c, "ext": ext, "pts": pts, "mask": mask})

comps = dedupe_3d(comps_raw, radius=0.06)
comps.sort(key=lambda d: d["center"][1])
print(f"[COMPS] {len(comps_raw)} raw → {len(comps)} after dedupe")
for i, c in enumerate(comps):
    print(f"   [{i}] y={c['center'][1]:.3f} center={c['center'].tolist()} ext={c['ext'].tolist()}")

if not comps:
    raise RuntimeError("No compartments found")

left_comp = comps[0]
left_center = left_comp["center"]
left_pts = left_comp["pts"]
print(f"[LEFT_COMP] center={left_center.tolist()} ext={left_comp['ext'].tolist()}")

# === Step 4: Caddy rim height ===
caddy_prompts = ["desk organizer", "pencil caddy", "tray with compartments"]
caddy = find_object(rgb, depth_img, K, E, caddy_prompts,
                    ext_xy_min=0.25, ext_xy_max=0.60,
                    pos_x_min=0.20, pos_x_max=0.55,
                    pos_y_min=-0.30, pos_y_max=0.20,
                    min_score=0.40)
if caddy is None:
    print("[CADDY] not found via prompts; using compartment p95")
    caddy_rim_z = float(np.percentile(left_pts[:, 2], 95))
else:
    caddy_pts = caddy["pts"]
    caddy_rim_z = float(np.percentile(caddy_pts[:, 2], 95))
    print(f"[CADDY] prompt='{caddy['prompt']}' rim_z(p95)={caddy_rim_z:.3f}")

left_floor_z = float(np.percentile(left_pts[:, 2], 5))
print(f"[LEFT_COMP] floor_z={left_floor_z:.3f}, rim_z={caddy_rim_z:.3f}")

# === Step 5: Plan grasp on book ===
quat_grasp = make_topdown_quat(yaw_deg)
# Place quat: align gripper X with world X so book-width fits along compartment long axis (~0.176)
# and book-thickness along world-Y (compartment short axis ~0.131).
quat_place = make_topdown_quat(0)
grasp_z = book_top_z - 0.010
grasp_pos = np.array([book_xy[0], book_xy[1], grasp_z])
print(f"[GRASP] yaw={yaw_deg:.1f}° grasp_pos={grasp_pos.tolist()}")

# === Step 6: Pre-grasp + descend + close ===
goto_pose(grasp_pos, quat_grasp, z_approach=0.18)
goto_pose(grasp_pos, quat_grasp)
close_gripper()

obs2 = get_observation()
rcp = obs2.get("robot_cartesian_pos", [])
gw = rcp[7] if len(rcp) >= 8 else 0
print(f"[GRIP] width={gw:.3f}")

# === Step 7: Lift +0.30 (clears caddy rim during transit) — keep grasp orientation ===
lift_z = grasp_z + 0.30
lift_pos = np.array([grasp_pos[0], grasp_pos[1], lift_z])
joints = solve_ik(lift_pos.tolist(), quat_grasp.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 7b: Re-orient to placement quat (yaw=0) above book — book width to world X ===
joints = solve_ik(lift_pos.tolist(), quat_place.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 8: Move above left compartment via 2-step (Y first then X) for stable IK ===
# Drop Y first (avoiding workspace boundary near robot), then approach X.
mid = np.array([grasp_pos[0], left_center[1], lift_z])
joints = solve_ik(mid.tolist(), quat_place.tolist())
if joints is not None:
    move_to_joints(joints)
above = np.array([left_center[0], left_center[1], lift_z])
joints = solve_ik(above.tolist(), quat_place.tolist())
if joints is not None:
    move_to_joints(joints)

# Re-observe to check book is still in gripper
obs_check = get_observation()
rcp = obs_check.get("robot_cartesian_pos", [])
gw_check = rcp[7] if len(rcp) >= 8 else 0
print(f"[CHECK_AFTER_TRANSIT] gripper_width={gw_check:.3f} pos={rcp[:3] if len(rcp)>=3 else 'N/A'}")

# === Step 9: Lower to release via gradual descent ===
# Book hangs 14cm below gripper; want book bottom near compartment floor before release.
# Gradual descent prevents bouncing/swing.
target_release_z = left_floor_z + 0.02 + book_height - 0.012
release_z = max(target_release_z, caddy_rim_z + 0.020)
release_pos = np.array([left_center[0], left_center[1], release_z])
print(f"[RELEASE] target={target_release_z:.3f} chosen={release_z:.3f} (rim={caddy_rim_z:.3f}, floor={left_floor_z:.3f}, book_h={book_height:.3f})")
# Multi-step descent
for step_z in [lift_z - 0.05, lift_z - 0.10, lift_z - 0.15, release_z + 0.05, release_z]:
    pos = np.array([left_center[0], left_center[1], step_z])
    joints = solve_ik(pos.tolist(), quat_place.tolist())
    if joints is not None:
        move_to_joints(joints)
# Repeat at release_z to ensure convergence
for _ in range(2):
    joints = solve_ik(release_pos.tolist(), quat_place.tolist())
    if joints is not None:
        move_to_joints(joints)

# === Step 10: Release ===
open_gripper()

# === Step 11: Retreat upward ===
retreat_pos = np.array([release_pos[0], release_pos[1], release_pos[2] + 0.15])
joints = solve_ik(retreat_pos.tolist(), quat_place.tolist())
if joints is not None:
    move_to_joints(joints)

# Settle physics
for _ in range(5):
    get_observation()
