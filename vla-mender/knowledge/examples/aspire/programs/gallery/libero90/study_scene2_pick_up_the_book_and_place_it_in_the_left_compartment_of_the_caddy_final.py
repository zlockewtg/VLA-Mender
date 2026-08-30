"""
STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy
Task type: pick-and-place

Pick: book — standing upright on table, cover facing camera (~12cm wide x 14cm tall x 3.7cm thick).
      OBB center near (0.53, -0.05, 0.04), with z range -0.03 to 0.11 (book is vertical).
Place: LEFT compartment of a desk organizer / caddy with 3 compartments.
       The leftmost compartment in agentview = lowest world-y (~ -0.28).

Strategy:
  - Grasp top of book: top_z = max(book_pts[:,2]) ≈ 0.11. Grasp z = top_z - 0.012 (near top spine).
  - yaw=0: fingers close along world-y, gripping book's thickness (~3.7cm).
  - Release: above leftmost compartment center, descend to just above rim, open gripper.

Disambiguation:
  - "compartment" returns 3 large compartments + 1 small false hit. Filter by:
      ext_z > 0.05 (real compartment depth, false hits ~0.07 too)
    Better: "compartment" + filter by 2D bbox area (real compartments are large) + dedup by 3D pos,
    then sort by world-y ascending, take leftmost (index 0).
  - "book", "novel", "binder" all hit the book (score 0.81 for book, 0.81 for novel).
  - "desk organizer" (0.95) gives the caddy as a whole; use it for caddy_z (rim percentile).
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
                top=10, min_score=0.10, return_all=False):
    """Try each prompt, return best (or all) candidate(s) matching geometry filters."""
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
                                "pts": pts, "mask": mask, "box": m.get("box")})
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
# The book is dark/black, upright on edge. Some seeds rotate it 90° so
# generic SAM3 prompts ("book", "novel") drop in score. Use "black object"
# and "small black box" as fallbacks; filter by geometry to reject the caddy/cup.
book_prompts = ["book", "novel", "binder", "black book",
                "black object", "small black box", "rectangular object",
                "thin book", "textbook"]
book_cands = find_object(
    rgb, depth_img, K, E, book_prompts,
    ext_xy_min=0.06, ext_xy_max=0.20,        # book ~12-15cm wide x 12-14cm tall (visible)
    pos_x_min=0.40, pos_x_max=0.75,           # book sits in front of caddy (caddy is x≈0.36)
    pos_y_min=-0.20, pos_y_max=0.20,
    pos_z_min=-0.05, pos_z_max=0.15,
    min_score=0.25, return_all=True,
)
# Dedupe & rank by score
book_cands = dedupe_3d(book_cands, radius=0.06)
if not book_cands:
    raise RuntimeError("Book not found")
book = max(book_cands, key=lambda d: d["score"])
book_center = book["center"]
book_pts = book["pts"]
book_mask = book["mask"]
book_top_z = float(book_pts[:, 2].max())
book_bottom_z = float(book_pts[:, 2].min())
book_height = book_top_z - book_bottom_z
print(f"[BOOK] prompt='{book['prompt']}' score={book['score']:.3f}")
print(f"[BOOK] center={book_center.tolist()} ext={book['ext'].tolist()}")
print(f"[BOOK] top_z={book_top_z:.3f} bottom_z={book_bottom_z:.3f} height={book_height:.3f}")

# Use OBB center XY as primary; book is symmetric flat so OBB is robust.
book_xy = book_center[:2]

# === Step 3: Localize caddy compartments ===
# "compartment" returns multiple hits. Collect, dedupe in 3D, filter to real compartments
# (sufficient ext, table-area position), sort by world-y, take leftmost.
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
    # Filter to real compartments: located on table area, reasonable size, not the whole caddy
    if c[0] < 0.20 or c[0] > 0.55:
        continue
    if c[1] < -0.45 or c[1] > 0.10:
        continue
    if max(ext[0], ext[1]) > 0.30:  # exclude full-caddy hit (ext ~0.45)
        continue
    if max(ext[0], ext[1]) < 0.08:  # exclude noise
        continue
    comps_raw.append({"score": score, "center": c, "ext": ext, "pts": pts, "mask": mask})

# Dedupe by 3D position
comps = dedupe_3d(comps_raw, radius=0.06)
# Sort by world-y (ascending = left in image)
comps.sort(key=lambda d: d["center"][1])
print(f"[COMPS] {len(comps_raw)} raw → {len(comps)} after dedupe")
for i, c in enumerate(comps):
    print(f"   [{i}] y={c['center'][1]:.3f} center={c['center'].tolist()} ext={c['ext'].tolist()}")

if not comps:
    raise RuntimeError("No compartments found")

# Leftmost = lowest world-y
left_comp = comps[0]
left_center = left_comp["center"]
left_pts = left_comp["pts"]
print(f"[LEFT_COMP] center={left_center.tolist()} ext={left_comp['ext'].tolist()}")

# === Step 4: Caddy rim height (for transit clearance and release height) ===
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

# Floor of compartment ≈ z minimum of compartment pts (or use percentile to be robust)
left_floor_z = float(np.percentile(left_pts[:, 2], 5))
print(f"[LEFT_COMP] floor_z={left_floor_z:.3f}, rim_z={caddy_rim_z:.3f}")

# === Step 5: Plan grasp on book ===
# Yaw selection: book width along world-X (~0.16m), depth/thickness along world-Y (~0.04m).
# yaw=0 → fingers close along world-Y, gripping the thin axis. (Standard top-down.)
quat = make_topdown_quat(0)

# Grasp at top of book - ~1cm (grip top spine). Book is upright so this is the top edge.
grasp_z = book_top_z - 0.010
grasp_pos = np.array([book_xy[0], book_xy[1], grasp_z])
print(f"[GRASP] yaw=0 grasp_pos={grasp_pos.tolist()}")

# === Step 6: Pre-grasp + descend + close ===
goto_pose(grasp_pos, quat, z_approach=0.18)
goto_pose(grasp_pos, quat)
close_gripper()

obs2 = get_observation()
rcp = obs2.get("robot_cartesian_pos", [])
gw = rcp[7] if len(rcp) >= 8 else 0
print(f"[GRIP] width={gw:.3f}")

# === Step 7: Lift (book is tall, lift higher than usual) ===
# Need book bottom to clear caddy rim during transit. Book is 14cm tall when held by top.
# Gripper at z = grasp_z + 0.20 → book bottom at z = grasp_z - book_height + 0.20 = grasp_z + 0.06.
# Caddy rim ~0.142 → need book bottom > 0.142 + safety. With grasp_z=0.10, lifted gripper z=0.30,
# book bottom at 0.16 (clears rim by 2cm). Use lift +0.30 for safer 12cm clearance.
lift_z = grasp_z + 0.30
lift_pos = np.array([grasp_pos[0], grasp_pos[1], lift_z])
joints = solve_ik(lift_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 8: Move above left compartment at lift height ===
above = np.array([left_center[0], left_center[1], lift_z])
joints = solve_ik(above.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 9: Lower to release. ===
# Goal: book bottom slightly inside compartment.
# Gripper tcp at z = release_z. Book is held with top at gripper z + 0.012 (grasp was top - 0.012).
# So book bottom at z = (release_z + 0.012) - book_height.
# Want book bottom ≈ left_floor_z + 0.02 (just above floor):
#   release_z = left_floor_z + 0.02 + book_height - 0.012
# Constraint: release_z must keep gripper above caddy rim (avoid finger collision):
#   release_z >= caddy_rim_z + 0.04 (gripper above rim)
target_release_z = left_floor_z + 0.02 + book_height - 0.012
release_z = max(target_release_z, caddy_rim_z + 0.04)
release_pos = np.array([left_center[0], left_center[1], release_z])
print(f"[RELEASE] target={target_release_z:.3f} chosen={release_z:.3f} (rim={caddy_rim_z:.3f}, floor={left_floor_z:.3f}, book_h={book_height:.3f})")
joints = solve_ik(release_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# === Step 10: Release ===
open_gripper()

# === Step 11: Retreat upward ===
retreat_pos = np.array([release_pos[0], release_pos[1], release_pos[2] + 0.15])
joints = solve_ik(retreat_pos.tolist(), quat.tolist())
if joints is not None:
    move_to_joints(joints)

# Settle physics
for _ in range(5):
    get_observation()
