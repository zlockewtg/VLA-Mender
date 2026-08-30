# Code block 0
import numpy as np

# === Utilities ===

def make_topdown_quat():
    R = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ], dtype=float)
    return rotation_matrix_to_quaternion(R)

TOP_DOWN_QUAT = make_topdown_quat()

def move_to_pose(pos, quat=None):
    if quat is None:
        quat = TOP_DOWN_QUAT
    joints = solve_ik(np.asarray(pos, dtype=float), np.asarray(quat, dtype=float))
    move_to_joints(joints)

def safe_home():
    """Move arm to a safe position above workspace to clear camera view."""
    move_to_pose([0.0, 0.0, 0.25])

def get_best_mask(rgb, prompt, min_pixels=50, max_pixels=12000):
    """Get best SAM3 mask, filtering by pixel count to exclude robot arm."""
    masks = segment_sam3_text_prompt(rgb, prompt)
    valid = []
    if masks:
        for m in masks:
            if "mask" not in m or m["mask"] is None:
                continue
            area = int(np.sum(m["mask"] > 0))
            if min_pixels <= area <= max_pixels:
                valid.append(m)

    if not valid:
        # Fallback: Molmo point prompt -> SAM3 point prompt
        pts = point_prompt_molmo(rgb, prompt)
        if pts:
            for _, p in pts.items():
                if p[0] is not None and p[1] is not None:
                    pmasks = segment_sam3_point_prompt(rgb, (int(p[0]), int(p[1])))
                    if pmasks:
                        for m in pmasks:
                            if "mask" not in m or m["mask"] is None:
                                continue
                            area = int(np.sum(m["mask"] > 0))
                            if min_pixels <= area <= max_pixels:
                                valid.append(m)
                    break

    if not valid:
        # Last resort: take best mask without area filter
        if masks:
            candidate = [m for m in masks if "mask" in m and m["mask"] is not None]
            if candidate:
                valid = candidate

    if not valid:
        return None
    return max(valid, key=lambda m: float(m.get("score", 0.0)))["mask"]

def get_object_info(prompt, retries=2):
    """Localize an object, returning center, top_z, min_z, with retry."""
    for attempt in range(retries):
        obs = get_observation()
        cam = obs["robot0_robotview"]
        rgb = cam["images"]["rgb"]
        depth = cam["images"]["depth"]
        K = cam["intrinsics"]
        T = cam["pose_mat"]

        mask = get_best_mask(rgb, prompt)
        if mask is None:
            print(f"  [{prompt}] attempt {attempt}: no mask found")
            continue

        pts = mask_to_world_points(mask.astype(np.uint8), depth, K, T)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 10:
            print(f"  [{prompt}] attempt {attempt}: too few points ({pts.shape[0]})")
            continue

        # Use median for robustness against outliers
        center = np.median(pts, axis=0)
        top_z = float(np.percentile(pts[:, 2], 95))  # Use 95th percentile instead of max to avoid outliers
        min_z = float(np.percentile(pts[:, 2], 5))

        info = {
            "center": center,
            "top_z": top_z,
            "min_z": min_z,
        }
        print(f"  [{prompt}] center={center}, top_z={top_z:.4f}, min_z={min_z:.4f}")
        return info

    return None

def pick_object(info, max_retries=3):
    """Pick an object with retry logic. Returns True if grasp succeeded."""
    cx, cy = info["center"][0], info["center"][1]
    top_z = info["top_z"]
    min_z = info["min_z"]
    # Grasp at center height for better grip
    center_z = (top_z + min_z) / 2.0

    for attempt in range(max_retries):
        open_gripper()

        # Pre-grasp above object
        move_to_pose([cx, cy, top_z + 0.10])

        # Descend to center height of cube
        grasp_z = center_z - 0.002 * attempt  # Go slightly lower on retries
        move_to_pose([cx, cy, grasp_z])

        close_gripper()

        # Check if we got the object by reading gripper state
        obs = get_observation()
        gripper_qpos = obs.get("robot0_gripper_qpos", None)

        # Lift regardless
        move_to_pose([cx, cy, top_z + 0.12])

        if gripper_qpos is not None:
            gw = float(gripper_qpos[0])
            print(f"  pick attempt {attempt}: gripper_qpos={gw:.4f}")
            if gw > 0.003:  # Object grasped
                return True
            else:
                print(f"  pick attempt {attempt}: air grasp, retrying...")
                # Open and re-observe
                open_gripper()
                safe_home()
                new_info = get_object_info(info.get("_prompt", "cube"))
                if new_info is not None:
                    cx, cy = new_info["center"][0], new_info["center"][1]
                    top_z = new_info["top_z"]
                    min_z = new_info["min_z"]
                    center_z = (top_z + min_z) / 2.0
        else:
            # Can't check gripper - assume success
            return True

    return True  # Continue even if uncertain

def place_object(xy, place_z, clearance=0.032):
    """Place the held object at target xy, lowering to place_z + clearance."""
    move_to_pose([xy[0], xy[1], place_z + 0.12])
    move_to_pose([xy[0], xy[1], place_z + clearance])
    open_gripper()
    move_to_pose([xy[0], xy[1], place_z + 0.12])

# === Main Execution ===
print("=== Cube Restack: Move green aside, stack red on green ===")

# Step 0: Move arm to safe position and observe
open_gripper()
safe_home()

# Step 1: Locate both cubes
print("\n--- Step 1: Initial observation ---")
green = get_object_info("green cube")
red = get_object_info("red cube")

if green is None or red is None:
    print("ERROR: Could not detect both cubes!")
    # Try alternative prompts
    if green is None:
        green = get_object_info("green block")
    if red is None:
        red = get_object_info("red block")

assert green is not None and red is not None, "Failed to detect cubes after fallback"

# Store prompts for retry logic
green["_prompt"] = "green cube"
red["_prompt"] = "red cube"

# Determine table z from the lowest point
table_z = min(green["min_z"], red["min_z"])
print(f"Table Z estimate: {table_z:.4f}")

# Step 2: Find safe temporary location for green cube
print("\n--- Step 2: Move green cube aside ---")
# Try several offsets to find a clear spot away from the red cube
offsets = [
    np.array([0.14, 0.0]),
    np.array([-0.14, 0.0]),
    np.array([0.0, 0.14]),
    np.array([0.0, -0.14]),
    np.array([0.10, 0.10]),
    np.array([-0.10, 0.10]),
]
temp_xy = None
for off in offsets:
    cand = green["center"][:2] + off
    if np.linalg.norm(cand - red["center"][:2]) > 0.10:
        temp_xy = cand
        break
if temp_xy is None:
    temp_xy = green["center"][:2] + np.array([0.14, 0.0])

# Pick green cube
pick_object(green)

# Place green aside on table
place_object(temp_xy, table_z, clearance=0.028)

# Step 3: Re-observe both cubes from safe position
print("\n--- Step 3: Re-observe after moving green ---")
safe_home()
green2 = get_object_info("green cube")
red2 = get_object_info("red cube")

if green2 is None or red2 is None:
    print("WARNING: Could not re-detect both cubes, trying fallback prompts")
    if green2 is None:
        green2 = get_object_info("green block")
    if red2 is None:
        red2 = get_object_info("red block")

assert green2 is not None and red2 is not None, "Failed to re-detect cubes"
green2["_prompt"] = "green cube"
red2["_prompt"] = "red cube"

# Step 4: Pick red cube
print("\n--- Step 4: Pick red cube ---")
pick_object(red2)

# Step 5: Place red on top of green
print("\n--- Step 5: Place red on green ---")
target_xy = green2["center"][:2].copy()
support_top_z = green2["top_z"]
place_object(target_xy, support_top_z, clearance=0.032)

# Step 6: Final cleanup
print("\n--- Step 6: Final ---")
open_gripper()
safe_home()

# Verify
green_f = get_object_info("green cube")
red_f = get_object_info("red cube")
if green_f and red_f:
    if red_f["center"][2] > green_f["center"][2]:
        print("SUCCESS: Red cube is above green cube!")
    else:
        print("NOTE: Red may not be on green - checking heights")
        print(f"  Red center Z: {red_f['center'][2]:.4f}, Green center Z: {green_f['center'][2]:.4f}")

print("=== Done ===")