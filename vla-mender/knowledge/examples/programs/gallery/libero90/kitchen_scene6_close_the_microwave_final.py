"""
LIBERO-90 KITCHEN_SCENE6: close the microwave

Approach:
- The microwave body is on the right side of the table, with its front face at world y≈0.24.
- The door is hinged at one end of the body's front face (around world (0.48, 0.24)) and rotates outward.
- The free edge varies per seed (some seeds: door wide open, free edge near (0.40, 0.01); others: door
  slightly open with free edge near (0.78, 0.24)).
- To close: with closed gripper, sweep along the natural door arc from outside the open door to past
  the body's front face. The closed gripper pushes the door face as it sweeps.
- Avoid re-localizing after the arm enters the workspace (occlusion).
"""

import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


def localize_object(rgb, depth, K, E, prompts, score_thresh=0.0):
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        for m in sorted(masks, key=lambda d: -d["score"])[:5]:
            if m["score"] < score_thresh:
                continue
            mask = m["mask"].astype(np.uint8)
            if mask.sum() < 100:
                continue
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 50:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts)
            return obb["center"], pts, mask, m["score"], prompt
    return None, None, None, 0.0, None


# ---------- Begin ----------
print("Task: close the microwave")

# Settle physics; ensure gripper closed
open_gripper()
close_gripper()
for _ in range(2):
    get_observation()

obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]

# Localize the open door — try multiple prompts; collect ALL hits then filter to door region.
# Door region: world-y in approximately [-0.10, 0.27], world-x in [0.30, 0.85], world-z in [0.0, 0.25]
# The microwave body itself is at y≥0.23, so anything at y<0.20 is the door (plate angled out).
all_door_pts = []
for prompt in ["open microwave door", "microwave door", "open microwave", "microwave"]:
    masks = segment_sam3_text_prompt(rgb, prompt)
    if not masks:
        continue
    for m in sorted(masks, key=lambda d: -d["score"])[:5]:
        mask = m["mask"].astype(np.uint8)
        if mask.sum() < 100:
            continue
        pts = mask_to_world_points(mask, depth_img, K, E)
        if pts is None or len(pts) < 30:
            continue
        # filter to door region
        sel = (
            (pts[:, 0] > 0.30) & (pts[:, 0] < 0.85) &
            (pts[:, 1] > -0.10) & (pts[:, 1] < 0.27) &
            (pts[:, 2] > 0.005) & (pts[:, 2] < 0.25)
        )
        pts_f = pts[sel]
        if len(pts_f) > 30:
            all_door_pts.append((prompt, m["score"], pts_f))

if not all_door_pts:
    print("WARN: no door candidates from SAM3; using image fallback region")
    H, W = rgb.shape[:2]
    m = np.zeros((H, W), dtype=np.uint8)
    m[140:260, 400:600] = 1
    pts = mask_to_world_points(m, depth_img, K, E)
    sel = (
        (pts[:, 0] > 0.30) & (pts[:, 0] < 0.85) &
        (pts[:, 1] > -0.10) & (pts[:, 1] < 0.27) &
        (pts[:, 2] > 0.005) & (pts[:, 2] < 0.25)
    )
    door_pts = pts[sel]
else:
    # Use the candidate with the most points in the door region (largest valid mask)
    all_door_pts.sort(key=lambda c: -len(c[2]))
    used_prompt, used_score, door_pts = all_door_pts[0]
    print(f"Using door prompt='{used_prompt}' score={used_score:.3f} npts={len(door_pts)}")

# Find the free edge (lowest y point - sticking out away from body)
# Use percentile to be robust
y_p5 = float(np.percentile(door_pts[:, 1], 5))
free_edge_pts = door_pts[door_pts[:, 1] < y_p5 + 0.03]
free_edge_x = float(np.median(free_edge_pts[:, 0]))
free_edge_y = float(np.median(free_edge_pts[:, 1]))
free_edge_z = float(np.median(free_edge_pts[:, 2]))

print(f"Door free edge: ({free_edge_x:.3f}, {free_edge_y:.3f}, {free_edge_z:.3f})")
print(f"Door y span: [{door_pts[:,1].min():.3f}, {door_pts[:,1].max():.3f}]")
print(f"Door x span: [{door_pts[:,0].min():.3f}, {door_pts[:,0].max():.3f}]")
print(f"Door z span: [{door_pts[:,2].min():.3f}, {door_pts[:,2].max():.3f}]")

# Body front face at y ≈ 0.24 (constant from observed seeds)
BODY_FACE_Y = 0.24
# Hinge approximately at body's front-left corner (low x, high y front face)
# From observation: body x range starts at ~0.48
HINGE_X = 0.48

# Push setup
push_z = float(np.clip(free_edge_z + 0.02, 0.06, 0.16))
quat = make_topdown_quat(0)

# Make sure gripper is closed
close_gripper()

# Trajectory: sweep along an arc from outside the door (low y, low x) toward inside (high y, high x).
# Use 4-5 waypoints.
# Start: outside the door's free edge (low y, x near free_edge_x)
# End: past the body face (y > 0.30, x near hinge_x + door_length = 0.74)
# Mid: arc through (mid_x, mid_y)

# Calculate the angle-from-closed of the open door
dx = free_edge_x - HINGE_X
dy = free_edge_y - BODY_FACE_Y
open_angle_rad = np.arctan2(dy, dx)
print(f"Door open angle: {np.degrees(open_angle_rad):.1f} deg from closed")

# Build interpolated arc waypoints
# Door radius (from hinge to free edge)
r = np.sqrt(dx*dx + dy*dy)
r = float(np.clip(r, 0.20, 0.30))
print(f"Estimated door length: {r:.3f}")

# Closed angle = 0; open angle = open_angle_rad
# Sweep from open_angle_rad to small positive (slightly past closed)
n_wp = 5
sweep_angles = np.linspace(open_angle_rad, np.radians(8.0), n_wp + 1)
# Approach the door from outside: each waypoint just outside the door face along arc, at slightly
# larger radius so the gripper is on the OUTER side of the door.
SAFE_OFFSET = 0.04  # gripper offset outside door face plane in normal direction

waypoints = []
# Pre-push start (above free edge area, safe-z)
start_above = (free_edge_x, free_edge_y - 0.07, 0.30)
waypoints.append(("safe-above-start", start_above))
# Pre-push at push_z
pre_push = (free_edge_x, free_edge_y - 0.05, push_z)
waypoints.append(("pre-push", pre_push))

# Arc waypoints: gripper position along door face arc, from open angle toward closed.
# At each angle, the door free_edge is at (HINGE_X + r*cos(a), BODY_FACE_Y + r*sin(a)).
# Gripper applies pressure on the OUTER side of door face. Outer normal in xy:
# rotated 90° CW from radius direction:
#   radius = (cos(a), sin(a))
#   outer = (sin(a), -cos(a))   for door "below-y" of hinge with body at +y
# We push the gripper a bit ahead of the next angle's free_edge position, on the outer side.
for i, ang in enumerate(sweep_angles):
    fx = HINGE_X + r * np.cos(ang)
    fy = BODY_FACE_Y + r * np.sin(ang)
    # Gripper position: at the door free-edge XY (no offset — gripper is a "pusher")
    # Slight offset on the outer side along +x, -y direction (depending on door angle)
    nx, ny = np.sin(ang), -np.cos(ang)  # outer normal rotated to be along door face
    # We want to push DEEPER inside than the door face current position so we make contact
    push_depth = 0.02
    px = fx + push_depth * (-nx)  # push into the door
    py = fy + push_depth * (-ny)
    waypoints.append((f"arc[{i}]", (float(px), float(py), push_z)))

# Final push past closed: move toward body face center to ensure latch
# Push to a position past body face along +y direction with the gripper at the far side of the door
final_push = (HINGE_X + r * 0.6, BODY_FACE_Y + 0.10, push_z)
waypoints.append(("final-push", final_push))

# Hold push for stability - small back-and-forth at body face to ensure latch
hold_push = (HINGE_X + r * 0.7, BODY_FACE_Y + 0.08, push_z)
waypoints.append(("hold-push", hold_push))

# Retreat: lift straight up first (avoid catching the door), THEN move sideways
retreat_up = (HINGE_X + r * 0.7, BODY_FACE_Y + 0.08, 0.35)
waypoints.append(("retreat-up", retreat_up))
retreat_away = (free_edge_x, BODY_FACE_Y - 0.10, 0.35)
waypoints.append(("retreat-away", retreat_away))

# Execute waypoints
for name, (tx, ty, tz) in waypoints:
    print(f"  WP[{name}]: ({tx:.3f}, {ty:.3f}, {tz:.3f})")
    j = solve_ik([tx, ty, tz], quat.tolist())
    if j is not None:
        move_to_joints(j)
    else:
        # Try slightly relaxed
        j2 = solve_ik([tx, ty, max(tz, 0.10)], quat.tolist())
        if j2 is not None:
            move_to_joints(j2)
        else:
            print(f"    IK failed at ({tx:.3f}, {ty:.3f}, {tz:.3f})")

# Settle
for _ in range(5):
    get_observation()

print("Done.")
