# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
T_cam_world = cam["pose_mat"]

quat_down = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)

# Segment the brown spill with fallback
masks = segment_sam3_text_prompt(rgb, "brown spill")
if not masks:
    pt_result = point_prompt_molmo(rgb, "brown spill")
    p = list(pt_result.values())[0] if pt_result else (None, None)
    if p[0] is not None and p[1] is not None:
        masks = segment_sam3_point_prompt(rgb, (float(p[0]), float(p[1])))

if not masks:
    raise RuntimeError("Could not segment brown spill.")

best = max(masks, key=lambda d: d.get("score", 0.0))
mask = best["mask"].astype(bool)

# Convert to world points
pts_world = mask_to_world_points(mask.astype(np.uint8), depth, K, T_cam_world)
if pts_world.shape[0] == 0:
    raise RuntimeError("No 3D points found for spill.")

# Filter invalid points
valid = np.isfinite(pts_world).all(axis=1)
pts_world = pts_world[valid]
if pts_world.shape[0] == 0:
    raise RuntimeError("No valid world points for spill.")

x_min, y_min = pts_world[:, 0].min(), pts_world[:, 1].min()
x_max, y_max = pts_world[:, 0].max(), pts_world[:, 1].max()

# Small inward margin
margin = 0.005
x_min += margin; x_max -= margin
y_min += margin; y_max -= margin

cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
if x_max <= x_min:
    x_min, x_max = cx - 0.01, cx + 0.01
if y_max <= y_min:
    y_min, y_max = cy - 0.01, cy + 0.01

# Build serpentine wiping path (horizontal pass)
ny = max(3, min(6, int(np.ceil((y_max - y_min) / 0.03)) + 1))
nx = max(3, min(5, int(np.ceil((x_max - x_min) / 0.04)) + 1))

x_samples = np.linspace(x_min, x_max, nx)
y_samples = np.linspace(y_min, y_max, ny)

waypoints = []
for i, y in enumerate(y_samples):
    xs = x_samples if i % 2 == 0 else x_samples[::-1]
    for x in xs:
        waypoints.append(np.array([float(x), float(y), 0.0], dtype=float))

# Second pass (vertical direction) for better coverage on curved spills
for j, x in enumerate(x_samples):
    ys = y_samples if j % 2 == 0 else y_samples[::-1]
    for y in ys:
        waypoints.append(np.array([float(x), float(y), 0.0], dtype=float))

# Densify with small interpolation steps
dense_path = [waypoints[0]]
for i in range(len(waypoints) - 1):
    seg = interpolate_segment(waypoints[i], waypoints[i + 1], step=0.02)
    if len(seg) > 1:
        dense_path.extend(seg[1:])

# Execute wiping
for p in dense_path:
    joints = solve_ik(np.array([p[0], p[1], 0.0], dtype=float), quat_down)
    move_to_joints(joints)

print({
    "spill_bounds": {"x_min": float(x_min), "x_max": float(x_max),
                     "y_min": float(y_min), "y_max": float(y_max)},
    "num_waypoints": len(dense_path),
})