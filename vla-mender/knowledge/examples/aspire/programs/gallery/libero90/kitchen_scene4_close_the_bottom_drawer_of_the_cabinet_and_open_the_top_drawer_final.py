"""
KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer

v9 strategy: phase 1 (3 passes, 5 wp each) + verify + extra push, phase 2.

Scene (cabinet at +Y):
  Cabinet face: y≈0.213
  Bottom drawer (OPEN): face Y≈0.07, handle Y≈0.05, z=0.04
  Top drawer (closed): handle bar Y≈0.190, z=0.18
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_topdown_quat(yaw_deg=0.0):
    R = (Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix()
         @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


QUAT_H_PLUSY = np.array([0.7071068, -0.7071068, 0.0, 0.0])
QUAT_H_MINUSY = np.array([0.7071068, 0.7071068, 0.0, 0.0])
TOP_DOWN_QUAT = make_topdown_quat(0.0)


def get_obs_arrays():
    obs = get_observation()
    cam = obs["agentview"]
    rgb = cam["images"]["rgb"]
    depth = cam["images"]["depth"]
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    return obs, rgb, depth_img, cam["intrinsics"], cam["pose_mat"]


def localize_handles(rgb, depth_img, K, E):
    candidates = []
    seen_centers = []
    for prompt in ["drawer handle", "metal handle", "handle"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks: continue
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:6]:
            if m["score"] < 0.30: continue
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < 30: continue
            c = pts.mean(0)
            if not (0.50 < c[0] < 0.85 and -0.15 < c[1] < 0.30 and -0.05 < c[2] < 0.28): continue
            dup = any(np.linalg.norm(c - sc) < 0.04 for sc in seen_centers)
            if dup: continue
            seen_centers.append(c)
            z80 = np.percentile(pts[:, 2], 80)
            bar_pts = pts[pts[:, 2] >= z80]
            if len(bar_pts) < 5: bar_pts = pts
            bar_c = bar_pts.mean(0)
            candidates.append({'pts': pts, 'c': c, 'bar_c': bar_c, 'bar_pts': bar_pts,
                                'score': m['score'], 'prompt': prompt})
    candidates.sort(key=lambda h: h['c'][2])
    return candidates


# ============= START =============
print(f"Task: {env.handle.task_language}", flush=True)
goto_home_joint_position()
close_gripper()

obs0, rgb0, depth0, K0, E0 = get_obs_arrays()
handles = localize_handles(rgb0, depth0, K0, E0)
print(f"Found {len(handles)} handles", flush=True)
for h in handles:
    print(f"  z={h['c'][2]:.3f} y={h['c'][1]:.3f}", flush=True)

if len(handles) < 2:
    raise RuntimeError("Need at least 2 handles")

bottom = handles[0]
top = handles[-1]
bx_h = float(bottom['bar_c'][0])
by_h = float(bottom['bar_c'][1])

top_bar_y = float(top['bar_c'][1])
cabinet_face_y = top_bar_y + 0.025

# ============= PHASE 1: CLOSE BOTTOM DRAWER =============
print("\n=== PHASE 1: Close bottom drawer ===", flush=True)
push_x = bx_h
push_quat = make_topdown_quat(yaw_deg=90.0)
target_y = cabinet_face_y + 0.04
pre_y = -0.05

push_z_values = [0.025, 0.040, 0.055]

for pass_idx, push_z in enumerate(push_z_values):
    print(f"  pass {pass_idx} z={push_z:.3f}", flush=True)
    high_pre = [push_x, pre_y, 0.30]
    j = solve_ik(high_pre, push_quat.tolist())
    if j is not None:
        try: move_to_joints(j)
        except Exception: break
    j = solve_ik([push_x, pre_y, push_z], push_quat.tolist())
    if j is not None:
        try: move_to_joints(j)
        except Exception: continue
    for ty_p in [0.05, 0.12, 0.18, 0.22, target_y]:
        j = solve_ik([push_x, ty_p, push_z], push_quat.tolist())
        if j is not None:
            try: move_to_joints(j)
            except Exception: break

# Retreat
j = solve_ik([push_x, pre_y, 0.40], push_quat.tolist())
if j is not None:
    try: move_to_joints(j)
    except Exception: pass

# Verify bottom drawer closed; if not, do extra push
obs_v = get_observation()
cam_v = obs_v["agentview"]
rgb_v = cam_v["images"]["rgb"]
depth_v_raw = cam_v["images"]["depth"]
depth_v = depth_v_raw[:, :, 0] if len(depth_v_raw.shape) == 3 else depth_v_raw
K_v = cam_v["intrinsics"]
E_v = cam_v["pose_mat"]
handles_v = localize_handles(rgb_v, depth_v, K_v, E_v)
bottom_y_after = None
for h in handles_v:
    if h['c'][2] < 0.08:
        bottom_y_after = float(h['c'][1])
        break
print(f"After phase 1: bottom handle Y={bottom_y_after}", flush=True)

if bottom_y_after is not None and bottom_y_after < cabinet_face_y - 0.03:
    print(f"  Not fully closed, extra push", flush=True)
    extra_z = 0.045
    j = solve_ik([push_x, pre_y, 0.30], push_quat.tolist())
    if j is not None:
        try: move_to_joints(j)
        except Exception: pass
    j = solve_ik([push_x, pre_y, extra_z], push_quat.tolist())
    if j is not None:
        try: move_to_joints(j)
        except Exception: pass
    for ty_p in [0.10, 0.18, 0.22, target_y]:
        j = solve_ik([push_x, ty_p, extra_z], push_quat.tolist())
        if j is not None:
            try: move_to_joints(j)
            except Exception: break
    j = solve_ik([push_x, pre_y, 0.40], push_quat.tolist())
    if j is not None:
        try: move_to_joints(j)
        except Exception: pass

goto_home_joint_position()

# ============= PHASE 2: OPEN TOP DRAWER =============
print("\n=== PHASE 2: Open top drawer ===", flush=True)

obs2, rgb2, depth2, K2, E2 = get_obs_arrays()
handles2 = localize_handles(rgb2, depth2, K2, E2)
print(f"Re-localized {len(handles2)} handles", flush=True)

if not handles2:
    raise RuntimeError("Lost handles")

top2 = max(handles2, key=lambda h: h['c'][2])
tx = float(top2['bar_c'][0])
ty = float(top2['bar_c'][1])
tz = float(top2['bar_c'][2])
top_bar_pts = top2['bar_pts']
ty_max = float(top_bar_pts[:, 1].max())
ty_min = float(top_bar_pts[:, 1].min())
tz_min = float(top_bar_pts[:, 2].min())
tz_max = float(top_bar_pts[:, 2].max())

quat_h = QUAT_H_PLUSY
j_test = solve_ik([tx, ty - 0.10, tz], quat_h.tolist())
if j_test is None:
    quat_h = QUAT_H_MINUSY

open_gripper()

grip_z = float((tz_min + tz_max) / 2)
push_past_y = ty_max + 0.005

approach_seq = [
    [tx, ty - 0.08, grip_z],
    [tx, ty - 0.02, grip_z],
    [tx, push_past_y, grip_z],
]
for pos in approach_seq:
    j = solve_ik(pos, quat_h.tolist())
    if j is not None:
        try: move_to_joints(j)
        except Exception: break

close_gripper()
obs_g = get_observation()
gw = float(obs_g["robot_cartesian_pos"][-1])
print(f"Grasp gw={gw:.3f}", flush=True)

# Pull -Y
for dy in [0.0, -0.05, -0.10, -0.16, -0.22, -0.28]:
    pos = [tx, ty + dy, grip_z]
    j = solve_ik(pos, quat_h.tolist())
    if j is not None:
        try: move_to_joints(j)
        except Exception: break

open_gripper()

escape1 = [tx, ty - 0.32, grip_z + 0.05]
j = solve_ik(escape1, quat_h.tolist())
if j is not None:
    try: move_to_joints(j)
    except Exception: pass

print("Done", flush=True)
