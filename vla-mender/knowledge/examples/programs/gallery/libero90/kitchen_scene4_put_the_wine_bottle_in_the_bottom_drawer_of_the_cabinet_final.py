import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def localize_object(rgb, depth, K, E, prompts):
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
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


def body_xy_centroid(pts, radius=0.04, iters=3):
    xy = pts[:, :2]
    cx, cy = float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))
    for _ in range(iters):
        dist = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
        inliers = pts[dist < radius]
        if len(inliers) < 5:
            break
        cx, cy = float(inliers[:, 0].mean()), float(inliers[:, 1].mean())
    return cx, cy


# ── Step 1: Observe ──
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
E = cam["pose_mat"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

# ── Step 2: Localize wine bottle ──
bottle_center, bottle_pts, bottle_mask = localize_object(
    rgb, depth, K, E,
    ["wine bottle", "dark bottle", "green bottle", "bottle"],
)
if bottle_center is None:
    raise RuntimeError("Wine bottle not found")

bx, by = body_xy_centroid(bottle_pts)
top_z = bottle_pts[:, 2].max()
grasp_z = top_z - 0.04   # 4cm below the top of the bottle
quat = make_topdown_quat(0)

print(f"[bottle] body_xy=({bx:.3f},{by:.3f}) top_z={top_z:.3f} grasp_z={grasp_z:.3f}", flush=True)

# ── Step 3: Localize the open drawer interior BEFORE grasping (uses initial scene observation) ──
# The bottom drawer is already open in the scene. Find its handle and interior.
def find_drawer_interior(rgb, depth_img, K, E):
    """Find the open drawer interior using multiple cues."""
    # Cue 1: drawer handle (front of drawer = lower x where drawer face is)
    handle_pos = None
    for prompt in ["drawer handle", "lower drawer handle", "lower handle"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in masks[:6]:
            mk = m["mask"].astype(np.uint8)
            p = mask_to_world_points(mk, depth_img, K, E)
            if p is None or len(p) < 20:
                continue
            cx = float(np.median(p[:, 0]))
            cy = float(np.median(p[:, 1]))
            cz = float(np.median(p[:, 2]))
            # Bottom drawer handle: lower z (~0.04), forward (smaller y), right side of table (cx 0.55-0.80)
            if 0.55 < cx < 0.80 and -0.1 < cy < 0.30 and 0.0 < cz < 0.08:
                handle_pos = (cx, cy, cz, p)
                print(f"  [handle '{prompt}'] sc={m['score']:.3f} pos=({cx:.3f},{cy:.3f},{cz:.3f})", flush=True)
                break
        if handle_pos is not None:
            break

    # Cue 2: drawer interior masks
    interior_pts = None
    interior_score = -1
    for prompt in ["drawer interior", "open drawer", "drawer"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in masks[:6]:
            mk = m["mask"].astype(np.uint8)
            p = mask_to_world_points(mk, depth_img, K, E)
            if p is None or len(p) < 80:
                continue
            cx = float(np.median(p[:, 0]))
            cy = float(np.median(p[:, 1]))
            cz_med = float(np.median(p[:, 2]))
            cz_min = float(np.percentile(p[:, 2], 10))
            # The drawer FLOOR should be near table height (z 0.0-0.08)
            # and inside the cabinet area (cx 0.50-0.80, cy 0.05-0.25)
            if 0.50 < cx < 0.80 and 0.05 < cy < 0.25 and 0.0 < cz_med < 0.08:
                if m["score"] > interior_score:
                    interior_pts = p
                    interior_score = m["score"]
                    print(f"  [interior '{prompt}'] sc={m['score']:.3f} med=({cx:.3f},{cy:.3f},{cz_med:.3f})", flush=True)
        if interior_pts is not None:
            break
    return handle_pos, interior_pts


print("[localize drawer]", flush=True)
handle_pos, interior_pts = find_drawer_interior(rgb, depth_img, K, E)

# Compute drop target. The bottom drawer is open, with handle at the front (small y).
# The drawer interior extends from handle_y (front, small y) toward larger y (back of drawer, deeper inside the cabinet).
# Best drop target: drawer center, biased toward the back to avoid front-rim collisions.
floor_z = None
if interior_pts is not None and len(interior_pts) > 50:
    # Use percentile center of the floor for floor_z reading
    floor_z = float(np.percentile(interior_pts[:, 2], 20))

if handle_pos is not None:
    # Anchor on handle: drawer center is 10cm "into" the drawer from handle.
    hx, hy, hz, hpts = handle_pos
    dx = hx                  # use handle x (drawer is centered around handle x in this scene)
    dy = hy + 0.10           # 10cm into the drawer from the handle (toward back)
    if floor_z is None:
        floor_z = hz - 0.01
elif interior_pts is not None and len(interior_pts) > 50:
    # No handle: use interior 75th percentile bounds (push toward back of drawer)
    dx = float(np.percentile(interior_pts[:, 0], 60))
    dy = float(np.percentile(interior_pts[:, 1], 60))
else:
    # Last-resort fallback to a known-good position
    print("[drawer] no localization; using fallback", flush=True)
    dx, dy = 0.65, 0.15
    floor_z = 0.04

# Sanity-clip
dx = float(np.clip(dx, 0.60, 0.72))
dy = float(np.clip(dy, 0.10, 0.20))
floor_z = float(np.clip(floor_z, 0.01, 0.08))
print(f"[drawer] drop target=({dx:.3f},{dy:.3f},{floor_z:.3f})", flush=True)

# ── Step 4: Pre-grasp + grasp ──
open_gripper()
goto_pose(np.array([bx, by, grasp_z]), quat, z_approach=0.15)
goto_pose(np.array([bx, by, grasp_z]), quat)
close_gripper()
for _ in range(2):
    get_observation()

# ── Step 5: Lift bottle clear ──
lift_z = max(grasp_z + 0.20, 0.30)
joints = solve_ik([bx, by, lift_z], quat.tolist())
if joints is not None:
    move_to_joints(joints)

# ── Step 6: Carry over the drawer ──
above_z = max(lift_z, floor_z + 0.30)
joints = solve_ik([dx, dy, above_z], quat.tolist())
if joints is not None:
    move_to_joints(joints)

# ── Step 7: Lower into the drawer ──
release_z = floor_z + 0.05
joints = solve_ik([dx, dy, release_z], quat.tolist())
if joints is None:
    release_z = floor_z + 0.10
    joints = solve_ik([dx, dy, release_z], quat.tolist())
if joints is not None:
    move_to_joints(joints)

# ── Step 8: Release ──
open_gripper()
for _ in range(3):
    get_observation()

# Retreat
retreat_z = release_z + 0.20
joints = solve_ik([dx, dy, retreat_z], quat.tolist())
if joints is not None:
    move_to_joints(joints)
for _ in range(3):
    get_observation()
