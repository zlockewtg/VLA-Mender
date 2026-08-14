"""
Task code for: libero_90 / KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet
Task language: "open the top drawer of the cabinet"

Strategy (drawer-only, no placement):
1. Localize top drawer handle via SAM3 "drawer handle" → filter to cabinet workspace
   x∈[0.4,0.85], y∈[-0.30,-0.10], z∈[0.05,0.25]; pick highest Z = top drawer.
2. Approach handle horizontally (gripper -y facing) using quat_h.
3. Push past handle bar, close gripper, pull in +y direction (>= 0.20m).
4. Two-step escape and goto_home.

Validated reference: aspire_goal_task_actor open_the_top_drawer_and_put_the_bowl_inside.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat():
    R = np.column_stack([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


tdq = make_topdown_quat()
quat_h = np.array([0.707, 0.707, 0.0, 0.0])  # gripper points -y (into cabinet face)
TRANSIT_Z = 0.45


print(f"Task: {env.handle.task_language}", flush=True)

# Phase 0: Settle physics
for _ in range(3):
    open_gripper()
    close_gripper()
open_gripper()

# Phase 1: Observe scene
obs = get_observation()
c = obs["agentview"]
rgb = c["images"]["rgb"]
d = c["images"]["depth"]
if d.ndim == 3:
    d = d[:, :, 0]
K = c["intrinsics"]
E = c["pose_mat"]


def localize_top_handle(rgb, d, K, E):
    """Find top drawer handle in cabinet workspace, take highest Z."""
    candidates = []  # (score, centroid, n_pts)
    for prompt in ["drawer handle", "cabinet handle", "handle"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks[:20]:
            if m['score'] < 0.3:
                continue
            pts = mask_to_world_points(m['mask'].astype(np.uint8), d, K, E)
            if pts is None or len(pts) < 30:
                continue
            cxyz = pts.mean(0)
            cx, cy, cz = float(cxyz[0]), float(cxyz[1]), float(cxyz[2])
            # Filter to cabinet workspace
            if not (0.40 < cx < 0.85):
                continue
            if not (-0.30 < cy < -0.10):
                continue
            if not (0.04 < cz < 0.25):
                continue
            candidates.append({'c': cxyz, 's': float(m['score']), 'n': len(pts)})
    return candidates


cands = localize_top_handle(rgb, d, K, E)
print(f"Found {len(cands)} valid handle candidates", flush=True)
for k, h in enumerate(cands):
    print(f"  [{k}] s={h['s']:.3f} N={h['n']:4d} xyz=[{h['c'][0]:.3f},{h['c'][1]:.3f},{h['c'][2]:.3f}]", flush=True)

# Filter to top drawer: z in [0.14, 0.22] (top drawer handle ≈ 0.176-0.180)
top_cands = [h for h in cands if 0.14 <= float(h['c'][2]) <= 0.22]

if not top_cands:
    # Fallback to highest Z among all candidates
    if cands:
        cands.sort(key=lambda x: x['c'][2])
        hc = cands[-1]['c']
        print(f"No handle in z=[0.14,0.22]; using highest Z fallback at [{hc[0]:.3f},{hc[1]:.3f},{hc[2]:.3f}]", flush=True)
    else:
        # Last resort fixed estimate
        hc = np.array([0.665, -0.195, 0.180])
        print(f"No candidates; using fixed fallback [{hc[0]:.3f},{hc[1]:.3f},{hc[2]:.3f}]", flush=True)
else:
    # Pick the candidate with highest score among top-drawer candidates
    top_cands.sort(key=lambda x: x['s'], reverse=True)
    hc = top_cands[0]['c']
    print(f"Using top handle (score-best within z[0.14,0.22]) at [{hc[0]:.3f},{hc[1]:.3f},{hc[2]:.3f}]", flush=True)

hc = np.array([float(hc[0]), float(hc[1]), float(hc[2])])
hc_init_y = hc[1]

# Phase 2: Approach from front (+Y), push past handle bar, grip, pull open
open_gripper()

# Approach: dy decreasing to handle, with gripper pointing -y (4-step gradual descent)
for dy in [0.12, 0.08, 0.04, 0.01]:
    pos = hc.copy(); pos[1] += dy
    j = solve_ik(pos, quat_h)
    if j is not None:
        try:
            move_to_joints(j)
        except Exception as e:
            print(f"approach dy={dy} failed: {e}", flush=True)

# Push slightly past handle bar
for dy in [-0.02, -0.04]:
    pos = hc.copy(); pos[1] += dy
    j = solve_ik(pos, quat_h)
    if j is not None:
        try:
            move_to_joints(j)
        except Exception as e:
            print(f"push past dy={dy} failed: {e}", flush=True)

close_gripper()
gw_h = float(get_observation()['robot_cartesian_pos'][-1])
print(f"Handle gripped: gw={gw_h:.3f}", flush=True)

# Phase 3: Pull in +Y direction in steps (target: hc_y + 0.28)
for dy in [0.0, 0.06, 0.12, 0.16, 0.20, 0.24, 0.28]:
    pos = hc.copy(); pos[1] = hc_init_y + dy
    j = solve_ik(pos, quat_h)
    if j is not None:
        try:
            move_to_joints(j)
        except Exception as e:
            print(f"pull dy={dy} failed: {e}", flush=True)
            break

open_gripper()

# Phase 4: Two-step escape (avoid sweeping through drawer)
try:
    j_escape = solve_ik(np.array([hc[0] + 0.003, hc_init_y + 0.38, hc[2] + 0.05]), quat_h)
    if j_escape is not None:
        move_to_joints(j_escape)
    print("Escape step 1 OK", flush=True)
except Exception as e:
    print(f"Escape step 1 failed: {e}", flush=True)

try:
    j_transit = solve_ik(np.array([hc[0] + 0.003, hc_init_y + 0.38, TRANSIT_Z]), tdq)
    if j_transit is not None:
        move_to_joints(j_transit)
    print("Escape step 2 OK", flush=True)
except Exception as e:
    print(f"Escape step 2 failed: {e}", flush=True)

# Final home
try:
    goto_home_joint_position()
except Exception as e:
    print(f"goto_home failed: {e}", flush=True)

print("Done", flush=True)
