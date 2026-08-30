"""STUDY_SCENE4: pick up the LEFT book and place it ON TOP OF the shelf.

Scene findings (seed 51 probe):
- 3 books standing upright. Sorted by world-y ascending:
    cy=-0.247: LEFT (top_z=0.096, ext_x=0.130, ext_y=0.106)  <-- TARGET
    cy=-0.153: middle (top_z=0.110, ext_x=0.144, ext_y=0.117)
    cy=+0.001: right (top_z=0.096, ext_x=0.146, ext_y=0.128)

- Cabinet/shelf block: x=[0.389, 0.736], y=[0.208, 0.398], top at z=0.185.
- "Top of the shelf" = exterior top surface (NOT internal cabinet shelf — different predicate).
- Reachable in IK (z=0.185 < workspace ceiling).

Strategy v1:
- Localize all "book" candidates, filter by world-y < 0 region, sort by world-y, take min-y.
- Grasp top of book at top_z - 0.010 with yaw=0 (book standing upright, ext_x>ext_y for left book).
- Lift to z=0.45.
- Transit above cabinet center (sx=0.56, sy=0.30).
- Lower so book base is just above cabinet top (release_z = 0.185 + book_height - 0.005).
- Open gripper, settle, retreat.

Note: yaw selection per-book — "left" book has ext_x=0.13, ext_y=0.106 (closer to symmetric).
Use yaw such that fingers close on the SHORTER axis. yaw=0 closes fingers along world-y, so use
yaw=0 if ext_x > ext_y (close from sides along y). For left book: ext_x=0.13>0.106 -> yaw=0.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def get_scene():
    obs = get_observation()
    cam = obs["agentview"]
    rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
    K, E = cam["intrinsics"], cam["pose_mat"]
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    return rgb, depth_img, K, E


def find_books(rgb, depth_img, K, E):
    """Return list of book candidates with body geometry. Sorted by world-y ascending."""
    candidates = []
    for prompt in ["book", "books", "novel", "textbook", "black object", "small black box"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in masks[:20]:
            if m['score'] < 0.30:
                continue
            pts = mask_to_world_points(m['mask'].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 50:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            c = obb['center']
            if not (0.30 < c[0] < 0.85 and -0.45 < c[1] < 0.20):
                continue  # books are at y in [-0.25, 0]; tighter range
            body = pts[(pts[:, 2] > 0.005) & (pts[:, 2] < 0.20)]
            if len(body) < 100:
                continue
            bx_min, bx_max = float(body[:, 0].min()), float(body[:, 0].max())
            by_min, by_max = float(body[:, 1].min()), float(body[:, 1].max())
            ext_x = bx_max - bx_min
            ext_y = by_max - by_min
            if ext_x > 0.20 or ext_y > 0.20:
                continue
            cx = (bx_min + bx_max) / 2
            cy = (by_min + by_max) / 2
            top_z = float(body[:, 2].max())
            top_band = body[body[:, 2] > top_z - 0.015]
            if len(top_band) >= 10:
                top_cx = float(np.mean(top_band[:, 0]))
                top_cy = float(np.mean(top_band[:, 1]))
                top_z = float(np.percentile(top_band[:, 2], 90))
            else:
                top_cx, top_cy = cx, cy
            candidates.append({
                'center': np.array([cx, cy, c[2]]),
                'top_pos': np.array([top_cx, top_cy, top_z]),
                'ext_x': float(ext_x),
                'ext_y': float(ext_y),
                'top_z': top_z,
                'mask': m['mask'],
                'score': float(m['score']),
            })
        if len(candidates) >= 3:
            break

    # Dedupe by 3D center
    unique = []
    for c in candidates:
        is_dup = False
        for u in unique:
            if np.linalg.norm(c['center'][:2] - u['center'][:2]) < 0.05:
                is_dup = True
                if c['score'] > u['score']:
                    u.update(c)
                break
        if not is_dup:
            unique.append(c)
    unique.sort(key=lambda b: b['center'][1])  # ascending world-y
    return unique


def find_shelf(rgb, depth_img, K, E):
    """Find the cabinet block and its TOP surface (exterior top)."""
    all_pts = mask_to_world_points(np.ones(depth_img.shape, dtype=np.uint8), depth_img, K, E)
    if all_pts is None:
        return None
    cab = all_pts[(all_pts[:, 0] > 0.35) & (all_pts[:, 0] < 0.80) &
                  (all_pts[:, 1] > 0.10) & (all_pts[:, 1] < 0.50)]
    if len(cab) < 500:
        return None
    # Top surface: dense layer at z>0.16
    top = cab[(cab[:, 2] > 0.16) & (cab[:, 2] < 0.22)]
    if len(top) < 200:
        return None
    top_z = float(np.percentile(top[:, 2], 90))
    sx_min = float(np.percentile(top[:, 0], 5))
    sx_max = float(np.percentile(top[:, 0], 95))
    sy_min = float(np.percentile(top[:, 1], 5))
    sy_max = float(np.percentile(top[:, 1], 95))
    place_x = (sx_min + sx_max) / 2
    place_y = (sy_min + sy_max) / 2
    return {
        'top_z': top_z, 'top_x': place_x, 'top_y': place_y,
        'top_x_min': sx_min, 'top_x_max': sx_max,
        'top_y_min': sy_min, 'top_y_max': sy_max,
    }


# ===== MAIN =====
goto_home_joint_position()
open_gripper()
# Physics settle
for _ in range(3):
    close_gripper()
    open_gripper()

rgb, depth_img, K, E = get_scene()
books = find_books(rgb, depth_img, K, E)
if len(books) < 1:
    raise RuntimeError(f"Need books, found {len(books)}")
print(f"Found {len(books)} books (sorted by world-y):", flush=True)
for b in books:
    print(f"  cx={b['center'][0]:.3f} cy={b['center'][1]:.3f} top_z={b['top_z']:.3f} "
          f"ext=({b['ext_x']:.3f},{b['ext_y']:.3f}) score={b['score']:.2f}", flush=True)

# LEFT = index 0 of books sorted by y ascending (min-y)
left_book = books[0]
print(f"LEFT book: top={left_book['top_pos'].round(3)}", flush=True)

bx = float(left_book['top_pos'][0])
by = float(left_book['top_pos'][1])
book_top_z = float(left_book['top_pos'][2])
book_height = max(book_top_z - 0.005, 0.060)
ext_x = left_book['ext_x']
ext_y = left_book['ext_y']

# yaw=0 closes along world-y (fingers close on the y faces of book).
# Want fingers to close on the SHORTER axis (i.e. close along narrow edge).
# If ext_x > ext_y: book is longer in x -> fingers should close along y -> yaw=0.
# If ext_y > ext_x: book is longer in y -> fingers should close along x -> yaw=90.
yaw_deg = 0 if ext_x >= ext_y else 90
quat = make_topdown_quat(yaw_deg)
print(f"yaw={yaw_deg} (ext_x={ext_x:.3f} ext_y={ext_y:.3f})", flush=True)

shelf = find_shelf(rgb, depth_img, K, E)
if shelf is None:
    raise RuntimeError("Shelf not found")
shelf_z = shelf['top_z']
sx = shelf['top_x']
sy = shelf['top_y']
print(f"Shelf top z={shelf_z:.3f}, place=({sx:.3f},{sy:.3f})", flush=True)

# ===== GRASP =====
grasp_z = book_top_z - 0.010
goto_pose([bx, by, max(grasp_z + 0.20, 0.30)], quat.tolist())
goto_pose([bx, by, grasp_z + 0.10], quat.tolist())
goto_pose([bx, by, grasp_z], quat.tolist(), z_approach=0.06)
close_gripper()
gw_g = get_observation()["robot_cartesian_pos"][-1]
print(f"After grasp: gw={gw_g:.4f}", flush=True)

# ===== LIFT =====
lift_z = max(grasp_z + 0.30, shelf_z + 0.20, 0.40)
goto_pose([bx, by, lift_z], quat.tolist())
gw_l = get_observation()["robot_cartesian_pos"][-1]
print(f"Lifted: gw={gw_l:.4f}", flush=True)

# ===== TRANSPORT =====
goto_pose([sx, sy, lift_z], quat.tolist())

# ===== LOWER & RELEASE =====
# Drop book onto shelf top: gripper finger tips should be at shelf_z + book_height - 0.010
# (book bottom touches shelf surface, book stands upright)
release_finger_z = shelf_z + book_height - 0.005
goto_pose([sx, sy, release_finger_z + 0.10], quat.tolist())
goto_pose([sx, sy, release_finger_z], quat.tolist(), z_approach=0.05)
gw_p = get_observation()["robot_cartesian_pos"][-1]
print(f"At release: gw={gw_p:.4f}", flush=True)

open_gripper()
for _ in range(15):
    get_observation()

# Retreat
goto_pose([sx, sy, release_finger_z + 0.20], quat.tolist())
goto_pose([sx - 0.15, sy - 0.30, release_finger_z + 0.25], quat.tolist())

for _ in range(20):
    get_observation()

print("Task complete", flush=True)
