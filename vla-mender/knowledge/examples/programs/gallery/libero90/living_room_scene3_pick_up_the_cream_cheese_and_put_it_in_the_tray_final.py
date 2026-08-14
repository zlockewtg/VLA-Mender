"""
LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray
Task type: pick-and-place

Pick: cream cheese — small flat blue/purple rectangular box, ~7-8cm x 4cm x 1.7cm,
      sits at z≈0.03. RGB ≈ (64,70,86), R/B ≈ 0.74.
Place: wooden tray on the right side of the table (rim z≈0.10).

Scene caveats (mirrors LR3 butter_tray):
- Robot arm at HOME may occlude front-of-table objects → deocclude first.
- Two visually-similar flat boxes are present (both ext ≈ 8x4x1.8cm):
    * cream cheese box (blue-purple): RGB ≈ (64,70,86), R/B ≈ 0.74
    * butter package    (warm orange):  RGB ≈ (92,53,35), R/B ≈ 2.60
  SAM3 prompts ("butter package", "small box", "rectangular package", "flat box")
  return BOTH boxes with scores ~0.91–0.95. Disambiguate by R/B ratio:
  cream cheese has R/B < 1.0 (validated 0.74);
  butter has R/B > 1.5 (validated 2.60).
- Other distractors filtered by ext_z<0.04 (cans = ~6cm tall) and z<0.06.
- Tray prompt "wooden tray" scores ≈0.93 reliably (already validated for tomato_tray, butter_tray).
- Release height: tray_rim_z + 0.05 (matches butter_tray pattern; avoids bounce).
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


def find_flat_box_candidates(rgb, depth_img, K, E, prompts,
                              ext_z_max=0.04, ext_xy_min=0.04, ext_xy_max=0.12,
                              pos_z_max=0.06, pos_z_min=0.0, min_score=0.30):
    """Return list of unique flat-box candidates (deduped by 3D proximity).
    Each candidate has score, rgb mean, center, ext, mask, pts."""
    cands = []
    for p in prompts:
        masks = segment_sam3_text_prompt(rgb, p)
        for m in masks[:8]:
            sc = m.get("score", 0)
            if sc < min_score:
                continue
            mk = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mk, depth_img, K, E)
            if pts is None or len(pts) < 50:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c, ext = obb["center"], obb["extent"]
            if ext[2] > ext_z_max:
                continue
            xy = max(ext[0], ext[1])
            if xy > ext_xy_max or xy < ext_xy_min:
                continue
            if c[2] > pos_z_max or c[2] < pos_z_min:
                continue
            bool_mask = m["mask"].astype(bool)
            rgb_pixels = rgb[bool_mask]
            if len(rgb_pixels) < 20:
                continue
            r = float(rgb_pixels[:, 0].mean())
            g = float(rgb_pixels[:, 1].mean())
            b = float(rgb_pixels[:, 2].mean())
            # Dedupe by 3D proximity; keep highest score
            replaced = False
            for prev in cands:
                if np.linalg.norm(np.array(c[:2]) - np.array(prev["center"][:2])) < 0.05:
                    if sc > prev["score"]:
                        prev.update({"score": sc, "prompt": p,
                                     "rgb": (r, g, b),
                                     "center": c, "ext": ext,
                                     "mask": mk, "pts": pts})
                    replaced = True
                    break
            if not replaced:
                cands.append({"score": sc, "prompt": p,
                              "rgb": (r, g, b),
                              "center": c, "ext": ext,
                              "mask": mk, "pts": pts})
    return cands


def localize_object_top1(rgb, depth_img, K, E, prompts, min_score=0.30):
    """Top-1 SAM3 localizer; tries prompts in order, returns first viable hit."""
    if isinstance(prompts, str):
        prompts = [prompts]
    for p in prompts:
        masks = segment_sam3_text_prompt(rgb, p)
        if not masks:
            continue
        best = max(masks, key=lambda d: d["score"])
        if best["score"] < min_score:
            continue
        mk = best["mask"].astype(np.uint8)
        pts = mask_to_world_points(mk, depth_img, K, E)
        if pts is None or len(pts) < 10:
            continue
        c = get_oriented_bounding_box_from_3d_points(pts)["center"]
        return c, pts, mk, best["score"]
    return None, None, None, 0.0


# === Step 1: Move arm aside to deocclude scene ===
quat_topdown = make_topdown_quat(0)
side_pos = np.array([0.4, -0.4, 0.3])
joints = solve_ik(side_pos.tolist(), quat_topdown.tolist())
if joints is not None:
    move_to_joints(joints)
else:
    for fb in [[0.3, -0.4, 0.35], [0.3, 0.4, 0.35], [0.5, -0.3, 0.3]]:
        joints = solve_ik(fb, quat_topdown.tolist())
        if joints is not None:
            move_to_joints(joints)
            break

obs = None
for _ in range(3):
    obs = get_observation()

# === Step 2: Localize cream cheese (blue flat box, R/B < 1.0) ===
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

box_prompts = ["butter package", "small box", "rectangular package", "flat box",
               "rectangular box", "cream cheese box", "blue box"]
candidates = find_flat_box_candidates(rgb, depth_img, K, E, box_prompts)
if not candidates:
    raise RuntimeError("No flat-box candidates found")

print(f"[CANDS] {len(candidates)} flat-box candidate(s):")
for c in candidates:
    rgb_v = c["rgb"]
    rb = rgb_v[0] / max(rgb_v[2], 1e-3)
    print(f"  prompt='{c['prompt']}' score={c['score']:.3f} center={[float(x) for x in c['center']]} "
          f"ext={[float(x) for x in c['ext']]} RGB=({rgb_v[0]:.0f},{rgb_v[1]:.0f},{rgb_v[2]:.0f}) R/B={rb:.2f}")

# Pick LOWEST R/B (blue/cool-colored one is cream cheese)
def rb_ratio(c):
    return c["rgb"][0] / max(c["rgb"][2], 1e-3)
cream = min(candidates, key=rb_ratio)
cream_rb = rb_ratio(cream)
print(f"[CREAM] selected prompt='{cream['prompt']}' R/B={cream_rb:.2f}")
if cream_rb > 1.2:
    # Couldn't distinguish — fallback: explicit blue prompts
    print("[WARN] R/B too high for cream cheese; fallback search for blue flat box…")
    cands2 = find_flat_box_candidates(rgb, depth_img, K, E,
                                       ["blue box", "cream cheese box",
                                        "purple box", "blue rectangular box",
                                        "small box", "butter package"],
                                       min_score=0.05)
    if cands2:
        cream = min(cands2, key=rb_ratio)
        cream_rb = rb_ratio(cream)
        print(f"[CREAM-FB] prompt='{cream['prompt']}' R/B={cream_rb:.2f}")

cream_center = cream["center"]
cream_pts = cream["pts"]
cream_mask = cream["mask"]

# === Step 3: Localize tray ===
tgt_center, tgt_pts, _, tgt_score = localize_object_top1(
    rgb, depth_img, K, E,
    ["wooden tray", "tray", "serving tray"], min_score=0.30
)
if tgt_center is None:
    raise RuntimeError("Tray not found")
tray_rim_z = float(tgt_pts[:, 2].max())
print(f"[TRAY] score={tgt_score:.3f} center={[float(x) for x in tgt_center]} rim_z={tray_rim_z:.3f}")

# === Step 4: Plan grasp ===
# Cream cheese is a thin flat box (~1.7cm). Top-down quat. Use plan_grasp + OBB-snap fallback.
quat = make_topdown_quat(0)

# Use plan_grasp if it returns sensible XY; otherwise fall back to OBB center.
obj_obb = get_oriented_bounding_box_from_3d_points(cream_pts)
obb_xy = obj_obb["center"][:2]

grasp_pos = None
try:
    grasp_poses, grasp_scores = plan_grasp(depth_img, K, cream_mask)
except Exception as e:
    print(f"[GRASP] plan_grasp exception: {e}")
    grasp_poses, grasp_scores = None, None

if grasp_poses is not None and len(grasp_poses) > 0:
    best_T, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
    if best_T is None:
        best_T = E @ grasp_poses[grasp_scores.argmax()]
    gp, _gq = decompose_transform(best_T)
    dist_xy = np.linalg.norm(gp[:2] - obb_xy)
    if dist_xy > 0.025:
        print(f"[GRASP] plan_grasp XY off by {dist_xy:.3f}m → snap to OBB center")
        gp = np.array([obb_xy[0], obb_xy[1], gp[2]])
    grasp_pos = gp
    print(f"[GRASP] plan_grasp grasp_pos={grasp_pos.tolist()}")
else:
    print("[GRASP] plan_grasp empty → use OBB center directly")
    grasp_pos = np.array([obb_xy[0], obb_xy[1], cream_pts[:, 2].max()])

# Force grasp z to (top - 0.005). Thin object → fingers hug sides at this z.
top_z = float(cream_pts[:, 2].max())
grasp_z = top_z - 0.005
grasp_pos = np.array([float(grasp_pos[0]), float(grasp_pos[1]), grasp_z])
print(f"[GRASP] final grasp_pos={grasp_pos.tolist()}, top_z={top_z:.3f}")

# === Step 5: Execute pick ===
open_gripper()
goto_pose(grasp_pos, quat, z_approach=0.15)
goto_pose(grasp_pos, quat)
close_gripper()

# Check grip width
obs2 = get_observation()
rcp = obs2.get("robot_cartesian_pos", [0]*8)
gw = rcp[7] if len(rcp) >= 8 else 0
print(f"[GRIP] width={gw:.3f}")

# === Step 6: Lift ===
lift_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.20])
joints = solve_ik(lift_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 7: Move above tray ===
above_tgt = np.array([tgt_center[0], tgt_center[1], lift_pos[2]])
joints = solve_ik(above_tgt.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 8: Lower to release just above rim ===
release_pos = np.array([tgt_center[0], tgt_center[1], tray_rim_z + 0.05])
joints = solve_ik(release_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 9: Release ===
open_gripper()

# === Step 10: Retreat upward ===
retreat_pos = np.array([release_pos[0], release_pos[1], release_pos[2] + 0.15])
joints = solve_ik(retreat_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# Settle
for _ in range(5):
    get_observation()
