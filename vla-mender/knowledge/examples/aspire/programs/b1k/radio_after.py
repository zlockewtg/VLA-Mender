# Code block 0
# fix_code_interactive_radio.py — Agent-3 Interactive Policy for pick_up_radio
# v4: Fix perception failure for low-z radios (seed 15: z=0.33 below table z=0.41)
#
# Key patterns:
#   1. find_object_base_rotate → reset_torso (fix depth pipeline)
#   2. get_object_pose for BOTH radio AND table → get_navigation_pose
#   3. Multi-angle fallback if get_navigation_pose fails
#   4. find_object_torso_rotate at close range → reset_torso before grasp
#   5. arm=1 first for table objects, then arm=0 — BUT skip arm=1 if stuck at same position
#   6. Navigate to NEW angle between grasp rounds: back up 1.0m first, then reapproach
#   7. Back up if distance < 0.45m (cuRobo hang zone)
#   8. Max 1 pose per arm per round (2 attempts, not 4) — saves time for recovery
#   9. v3: If reposition fails, arm=0 only — arm=1 from same position triggers cuRobo hang
#  10. v4: When sample_grasp_pose returns None, navigate to different side (not base_rotate)
#  11. v4: Skip find_object_torso_rotate when <0.55m — saves 100-200s
import numpy as np
import time

START_TIME = time.time()
TIME_BUDGET = 870  # 900s total minus 30s safety

def elapsed():
    return time.time() - START_TIME

def remaining():
    return TIME_BUDGET - elapsed()

def log(tag, msg):
    print(f"[{tag}] (t={elapsed():.0f}s, left={remaining():.0f}s) {msg}")

def safe_navigate(goal, label=""):
    """Navigate and verify movement."""
    pos_before, _, _ = get_robot_position()
    try:
        navigate_to_pose([float(goal[0]), float(goal[1]), float(goal[2])])
    except Exception as e:
        log("NAV", f"{label}: error: {e}")
    pos_after, _, _ = get_robot_position()
    moved = float(np.linalg.norm(np.array(pos_after[:2]) - np.array(pos_before[:2])))
    if moved > 0.05:
        log("NAV", f"{label}: moved {moved:.2f}m -> [{pos_after[0]:.2f}, {pos_after[1]:.2f}]")
        return True, pos_after
    else:
        log("NAV", f"{label}: stuck (moved {moved:.3f}m)")
        return False, pos_before

def safe_get_pose(name, retries=3):
    """get_object_pose with QHull retry."""
    for attempt in range(retries):
        try:
            result = get_object_pose(name)
            if result[0] is not None:
                return result
        except Exception as e:
            log("POSE", f"'{name}' attempt {attempt+1}: {e}")
        if attempt < retries - 1:
            try:
                reset_torso()
            except:
                pass
            get_env_observation()
    return (None, None, None, None, None)

def get_dist(pos1, pos2):
    return float(np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2])))

def robot_dist_to(target_pos):
    rp, _, _ = get_robot_position()
    return get_dist(rp, target_pos)

RADIO_NAMES = ["red radio", "radio", "portable radio"]
TABLE_NAMES = ["table", "coffee table", "wooden table", "dining table", "side table", "grey table", "desk"]

# =========================================================================
# BLOCK 1: Initial observation + find radio
# =========================================================================
log("INIT", "=== BLOCK 1: Observe + Find Radio ===")
robot_pos, robot_quat, robot_yaw = get_robot_position()
log("INIT", f"Robot at [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}], yaw={robot_yaw:.2f}")

rgb, depth = get_env_observation()
save_current_observation("start")

# Find radio
found_name = None
for name in RADIO_NAMES:
    if remaining() < 700:
        break
    log("SEARCH", f"find_object_base_rotate('{name}')...")
    try:
        found = find_object_base_rotate(name)
        if found:
            found_name = name
            log("SEARCH", f"Found as '{name}'!")
            break
    except Exception as e:
        log("SEARCH", f"Error: {e}")

if found_name is None and remaining() > 600:
    log("SEARCH", "Trying torso_rotate...")
    for name in RADIO_NAMES[:2]:
        try:
            found = find_object_torso_rotate(name)
            if found:
                found_name = name
                log("SEARCH", f"Torso found as '{name}'!")
                break
        except:
            continue

if found_name is None:
    found_name = "red radio"
    log("SEARCH", "Defaulting to 'red radio'")

# CRITICAL: reset_torso after find to fix depth pipeline
try:
    reset_torso()
except:
    pass
rgb, depth = get_env_observation()
save_current_observation("post_find")

log("SEARCH", f"Using name: '{found_name}'")

# =========================================================================
# BLOCK 2: Get poses + navigate to radio
# =========================================================================
log("NAV", "=== BLOCK 2: Get Poses + Navigate ===")

# Get radio pose
radio_result = safe_get_pose(found_name, retries=3)
radio_pos = radio_result[0]
P_radio = radio_result[3]
initial_radio_pos = None

if radio_pos is not None:
    initial_radio_pos = [float(radio_pos[0]), float(radio_pos[1]), float(radio_pos[2])]
    log("POSE", f"Radio at [{radio_pos[0]:.2f}, {radio_pos[1]:.2f}, z={radio_pos[2]:.2f}]")
    log("POSE", f"Distance: {robot_dist_to(radio_pos):.2f}m")
else:
    log("POSE", "WARNING: Radio pose FAILED")

# Get table pose — try multiple names
P_table = None
table_pos = None
table_name_found = None
for tname in TABLE_NAMES:
    t_result = safe_get_pose(tname, retries=2)
    if t_result[0] is not None:
        t_pts = t_result[3]
        if t_pts is not None:
            t_pts_np = np.asarray(t_pts)
            if len(t_pts_np) > 50:
                P_table = t_pts_np
                table_pos = t_result[0]
                table_name_found = tname
                log("POSE", f"Table '{tname}' at [{table_pos[0]:.2f}, {table_pos[1]:.2f}, z={table_pos[2]:.2f}], pts={len(t_pts_np)}")
                break
            else:
                log("POSE", f"Table '{tname}' too sparse ({len(t_pts_np)} pts)")

if P_table is None:
    log("POSE", "WARNING: No table found")

# Navigate using get_navigation_pose (THE critical API for table approach)
nav_success = False
if P_table is not None and P_radio is not None:
    try:
        P_radio_np = np.asarray(P_radio)
        if len(P_radio_np) > 0 and len(P_table) > 50:
            nav_goal = get_navigation_pose(P_table, P_radio_np)
            log("NAV", f"get_navigation_pose -> [{nav_goal[0]:.2f}, {nav_goal[1]:.2f}, yaw={nav_goal[2]:.2f}]")
            moved, _ = safe_navigate(nav_goal, "table_nav")
            if moved:
                nav_success = True
                d = robot_dist_to(radio_pos)
                log("NAV", f"After table_nav: dist={d:.2f}m")
    except Exception as e:
        log("NAV", f"get_navigation_pose error: {e}")

# Fallback: direct approach if table nav failed or no table
if not nav_success and radio_pos is not None:
    dist = robot_dist_to(radio_pos)
    if dist > 0.6:
        log("NAV", f"Direct approach (dist={dist:.2f}m)")
        rp, _, _ = get_robot_position()
        direction = np.array(radio_pos[:2]) - np.array(rp[:2])
        d = float(np.linalg.norm(direction))
        if d > 0.01:
            unit = direction / d
            for offset in [0.50, 0.40, 0.60]:
                target_xy = np.array(radio_pos[:2]) - unit * offset
                face_yaw = float(np.arctan2(direction[1], direction[0]))
                moved, _ = safe_navigate([float(target_xy[0]), float(target_xy[1]), face_yaw], f"direct_{offset:.2f}")
                if moved:
                    nd = robot_dist_to(radio_pos)
                    log("NAV", f"After direct: dist={nd:.2f}m")
                    if nd < 0.8:
                        break

# v4: Hop navigation for nav-blocked seeds (seed 25 pattern: stuck at 1.37m)
if radio_pos is not None:
    dist = robot_dist_to(radio_pos)
    if dist > 0.8 and remaining() > 500:
        log("NAV", f"Still far ({dist:.2f}m). Trying hop navigation...")
        for _hop in range(5):
            if remaining() < 450:
                break
            _rp_hop, _, _ = get_robot_position()
            _hop_dir = np.array(radio_pos[:2]) - np.array(_rp_hop[:2])
            _hop_d = float(np.linalg.norm(_hop_dir))
            if _hop_d < 0.8:
                break
            _hop_unit = _hop_dir / _hop_d
            _hop_target = np.array(_rp_hop[:2]) + _hop_unit * 0.40
            _hop_yaw = float(np.arctan2(_hop_dir[1], _hop_dir[0]))
            _hop_moved, _ = safe_navigate([float(_hop_target[0]), float(_hop_target[1]), _hop_yaw], f"hop_{_hop}")
            if not _hop_moved:
                log("NAV", f"  Hop {_hop}: stuck")
                break
            _hop_nd = robot_dist_to(radio_pos)
            log("NAV", f"  Hop {_hop}: dist={_hop_nd:.2f}m")
            if _hop_nd < 0.8:
                break

# Multi-angle fallback if still far
if radio_pos is not None:
    dist = robot_dist_to(radio_pos)
    if dist > 0.8 and remaining() > 500:
        log("NAV", f"Still far ({dist:.2f}m). Multi-angle approach...")
        rx, ry = float(radio_pos[0]), float(radio_pos[1])
        # v4: Try close positions first, then wider circle for nav-blocked seeds
        _nav_candidates = [
            (0, -0.50, "S_0.5"), (0, 0.50, "N_0.5"), (0.50, 0, "E_0.5"), (-0.50, 0, "W_0.5"),
            (-0.40, -0.40, "SW_0.5"), (0.40, -0.40, "SE_0.5"), (-0.40, 0.40, "NW_0.5"), (0.40, 0.40, "NE_0.5"),
        ]
        # v4: If still far after close circle, try wider ring (for nav-blocked seeds)
        for _r in [0.80, 1.0, 1.2]:
            for _a_deg in [0, 90, 180, 270, 45, 135, 225, 315]:
                _a_rad = np.radians(_a_deg)
                _nav_candidates.append((_r * np.cos(_a_rad), _r * np.sin(_a_rad), f"{_a_deg}d_{_r:.1f}m"))
        # Sort by distance from ROBOT (nearest first — more likely reachable)
        rp_now, _, _ = get_robot_position()
        _nav_candidates.sort(key=lambda c: (rx + c[0] - rp_now[0])**2 + (ry + c[1] - rp_now[1])**2)
        _blocked_count = 0
        for dx, dy, label in _nav_candidates:
            if remaining() < 400 or _blocked_count >= 5:
                break
            ax, ay = rx + dx, ry + dy
            fy = float(np.arctan2(ry - ay, rx - ax))
            moved, _ = safe_navigate([ax, ay, fy], f"angle_{label}")
            if moved:
                _blocked_count = 0
                nd = robot_dist_to(radio_pos)
                log("NAV", f"  {label}: dist={nd:.2f}m")
                if nd < 0.8:
                    break
            else:
                _blocked_count += 1

# Back up if too close (cuRobo hang zone < 0.45m)
if radio_pos is not None:
    dist = robot_dist_to(radio_pos)
    if dist < 0.40:
        log("NAV", f"Too close ({dist:.2f}m)! Backing up...")
        rp, _, _ = get_robot_position()
        direction = np.array(radio_pos[:2]) - np.array(rp[:2])
        d = float(np.linalg.norm(direction))
        if d > 0.01:
            unit = direction / d
            backup = np.array(rp[:2]) - unit * 0.15
            face_yaw = float(np.arctan2(direction[1], direction[0]))
            safe_navigate([float(backup[0]), float(backup[1]), face_yaw], "backup")

robot_pos, _, _ = get_robot_position()
final_nav_dist = robot_dist_to(radio_pos) if radio_pos is not None else float('inf')
log("NAV", f"Block 2 done: robot=[{robot_pos[0]:.2f}, {robot_pos[1]:.2f}], dist={final_nav_dist:.2f}m")

rgb, depth = get_env_observation()
save_current_observation("post_nav")

# =========================================================================
# BLOCK 3: Re-find at close range + Grasp
# =========================================================================
log("GRASP", "=== BLOCK 3: Re-find + Grasp ===")

# Re-find radio at close range
# v4: Skip find_object_torso_rotate when very close (<0.55m) — it wastes 100-200s
#     and can move camera AWAY from the radio (seed 15 failure mode)
dist = final_nav_dist
if dist < 0.55:
    log("GRASP", f"Very close ({dist:.2f}m) — skip re-find, use reset_torso only")
elif dist < 0.7:
    log("GRASP", "Close range — torso_rotate")
    try:
        find_object_torso_rotate(found_name)
    except:
        pass
else:
    log("GRASP", "Far — base_rotate")
    try:
        find_object_base_rotate(found_name)
    except:
        pass

# CRITICAL: reset_torso before sample_grasp_pose (depth projection accuracy)
try:
    reset_torso()
except:
    pass
rgb, depth = get_env_observation()
save_current_observation("pre_grasp")

# Update radio position after re-find
refind_result = safe_get_pose(found_name, retries=2)
if refind_result[0] is not None:
    refind_pos = refind_result[0]
    refind_dist = robot_dist_to(refind_pos)
    log("GRASP", f"Re-found at [{refind_pos[0]:.2f}, {refind_pos[1]:.2f}, z={refind_pos[2]:.2f}], dist={refind_dist:.2f}m")

    # Validate: don't trust detection that drifted >1.0m from initial
    if initial_radio_pos is not None:
        drift = get_dist(initial_radio_pos, refind_pos)
        log("GRASP", f"Drift from initial: {drift:.2f}m")
        if drift > 1.0:
            log("GRASP", "REJECT — drifted too far, using initial pos for reference")
        else:
            radio_pos = refind_pos

# v3: Close the gap if too far (>0.70m → cuRobo hang risk from long reach)
_pre_grasp_dist = robot_dist_to(radio_pos) if radio_pos is not None else 999
if _pre_grasp_dist > 0.70 and radio_pos is not None:
    log("GRASP", f"Too far ({_pre_grasp_dist:.2f}m > 0.70m). Approaching closer...")
    _rp_gap, _, _ = get_robot_position()
    _dir = np.array(radio_pos[:2]) - np.array(_rp_gap[:2])
    _d = float(np.linalg.norm(_dir))
    if _d > 0.01:
        _unit = _dir / _d
        _target = np.array(radio_pos[:2]) - _unit * 0.55  # aim for 0.55m
        _tyaw = float(np.arctan2(_dir[1], _dir[0]))
        moved_closer, _ = safe_navigate([float(_target[0]), float(_target[1]), _tyaw], "close_gap")
        _new_dist = robot_dist_to(radio_pos)
        log("GRASP", f"After close_gap: dist={_new_dist:.2f}m (moved={moved_closer})")

# Grasp loop: try arm=1 first (better for table objects), then arm=0
# v3: skip arm=1 if repositioning failed (same-position cuRobo hang pattern)
grasped = False
MAX_GRASP_ROUNDS = 5

# Pre-compute approach angles for between-round navigation
_rp_pre, _, _ = get_robot_position()
_curr_angle = float(np.arctan2(radio_pos[1] - _rp_pre[1], radio_pos[0] - _rp_pre[0])) if radio_pos is not None else 0.0
ROUND_ANGLE_OFFSETS = [0, np.pi/2, -np.pi/2, np.pi/3, -np.pi/3, np.pi]

# v3: Track arm=1 safety — if we're at the same position, arm=1 triggers cuRobo hang
_arm1_safe = True  # True for first round (new position from get_navigation_pose)

for grasp_round in range(MAX_GRASP_ROUNDS):
    if grasped or remaining() < 120:
        break

    log("GRASP", f"--- Round {grasp_round+1}/{MAX_GRASP_ROUNDS} ({remaining():.0f}s left) ---")

    # Sample grasp poses
    grasp_result = None
    try:
        grasp_result = sample_grasp_pose(found_name)
    except Exception as e:
        log("GRASP", f"sample_grasp_pose error: {e}")

    if grasp_result is None or grasp_result[0] is None:
        _cur_dist = robot_dist_to(radio_pos) if radio_pos is not None else 999
        log("GRASP", f"No grasp poses (dist={_cur_dist:.2f}m). Trying recovery...")

        # v4: DON'T call find_object_base_rotate when close — it wastes 200s in
        # PLANNING_ERRORs when nav-stuck. Instead, navigate to a different side.
        if _cur_dist < 0.8 and radio_pos is not None:
            # Navigate to a different position around the radio
            rx, ry = float(radio_pos[0]), float(radio_pos[1])
            _rp_nf, _, _ = get_robot_position()
            _nf_angle = float(np.arctan2(ry - _rp_nf[1], rx - _rp_nf[0]))
            # Try perpendicular directions first (most likely to clear occluding furniture)
            _no_pose_offsets = [np.pi/2, -np.pi/2, np.pi, np.pi/4, -np.pi/4]
            _nf_moved = False
            for _nf_off in _no_pose_offsets:
                if remaining() < 200:
                    break
                _nf_a = _nf_angle + _nf_off + np.pi  # approach FROM this direction
                _nf_x = rx + 0.55 * np.cos(_nf_a)
                _nf_y = ry + 0.55 * np.sin(_nf_a)
                _nf_yaw = float(np.arctan2(ry - _nf_y, rx - _nf_x))
                _nf_moved, _ = safe_navigate([_nf_x, _nf_y, _nf_yaw], f"no_pose_R{grasp_round+1}_{np.degrees(_nf_off):.0f}")
                if _nf_moved:
                    _arm1_safe = True  # new position
                    break
            if not _nf_moved:
                log("GRASP", "  All positions blocked. Trying torso_rotate...")
                try:
                    find_object_torso_rotate(found_name)
                except:
                    pass
            try:
                reset_torso()
            except:
                pass
        else:
            # Far away — use base_rotate (traditional approach)
            log("GRASP", "  Far — base_rotate re-find")
            try:
                find_object_base_rotate(found_name)
                reset_torso()
            except:
                pass
        get_env_observation()
        continue

    pregrasp_poses, grasp_poses = grasp_result
    n_poses = len(pregrasp_poses)
    log("GRASP", f"Got {n_poses} grasp candidates")

    # v3: arm order depends on whether we moved since last round
    # arm=1 from same position → cuRobo hang (confirmed on seeds 11, 13, 14)
    if _arm1_safe:
        arm_order = [1, 0]
        log("GRASP", f"  arm order: [1, 0] (new position)")
    else:
        arm_order = [0]
        log("GRASP", f"  arm order: [0] only (same position — skip arm=1 to avoid cuRobo hang)")

    for arm in arm_order:
        if grasped or remaining() < 120:
            break

        pg, gr = pregrasp_poses[0], grasp_poses[0]
        log("GRASP", f"  arm={arm}, pose 1/{n_poses}")

        open_gripper(arm=arm)
        gs = time.time()
        try:
            grasp_object(pg, gr, found_name, arm=arm)
            dur = time.time() - gs
            log("GRASP", f"  grasp_object completed in {dur:.0f}s")
        except Exception as e:
            dur = time.time() - gs
            log("GRASP", f"  grasp_object error after {dur:.0f}s: {e}")

        # Always check — even after error/timeout
        lift_arm(arm=arm)
        in_hand = check_object_in_hand(arm=arm)
        log("GRASP", f"  in_hand={in_hand}")

        if in_hand:
            grasped = True
            log("GRASP", f"SUCCESS! arm={arm}")
            break
        else:
            open_gripper(arm=arm)

    # If failed this round, try to reposition before next round
    # v3: improved repositioning — back up first, then approach from different angle
    if not grasped and grasp_round < MAX_GRASP_ROUNDS - 1 and remaining() > 150:
        offset_idx = (grasp_round + 1) % len(ROUND_ANGLE_OFFSETS)
        offset = ROUND_ANGLE_OFFSETS[offset_idx]
        log("GRASP", f"Round {grasp_round+1} failed. Repositioning (offset {np.degrees(offset):.0f}deg)...")

        try:
            reset_torso()
        except:
            pass

        _reposition_moved = False
        if radio_pos is not None:
            rx, ry = float(radio_pos[0]), float(radio_pos[1])

            # v3 strategy: try 0.70m offset first (wider than v2's 0.50m to clear furniture)
            ax = rx + 0.70 * np.cos(_curr_angle + offset)
            ay = ry + 0.70 * np.sin(_curr_angle + offset)
            fy = float(np.arctan2(ry - ay, rx - ax))
            moved, _ = safe_navigate([ax, ay, fy], f"round_{grasp_round+1}_reposition")

            if not moved:
                # v3 fallback: back up to 1.2m from radio (clear table), then try approach
                log("GRASP", f"  0.70m offset failed. Backing up to 1.2m...")
                rp_now, _, _ = get_robot_position()
                away_angle = float(np.arctan2(rp_now[1] - ry, rp_now[0] - rx))
                bx = rx + 1.2 * np.cos(away_angle)
                by = ry + 1.2 * np.sin(away_angle)
                byaw = float(np.arctan2(ry - by, rx - bx))
                moved, _ = safe_navigate([bx, by, byaw], f"round_{grasp_round+1}_backup")

                if moved:
                    # Now approach from the new angle
                    ax2 = rx + 0.60 * np.cos(_curr_angle + offset)
                    ay2 = ry + 0.60 * np.sin(_curr_angle + offset)
                    fy2 = float(np.arctan2(ry - ay2, rx - ax2))
                    moved2, _ = safe_navigate([ax2, ay2, fy2], f"round_{grasp_round+1}_reapproach")
                    _reposition_moved = moved2
                else:
                    _reposition_moved = False
            else:
                _reposition_moved = True

            # Check distance after reposition
            nd = robot_dist_to(radio_pos)
            log("GRASP", f"  After reposition: dist={nd:.2f}m, moved={_reposition_moved}")

            # Back up if too close
            if nd < 0.40:
                rp2, _, _ = get_robot_position()
                direction = np.array(radio_pos[:2]) - np.array(rp2[:2])
                d_norm = float(np.linalg.norm(direction))
                if d_norm > 0.01:
                    unit = direction / d_norm
                    backup = np.array(rp2[:2]) - unit * 0.15
                    safe_navigate([float(backup[0]), float(backup[1]), fy], "backup")

        # v3: Update arm=1 safety based on whether we actually moved
        _arm1_safe = _reposition_moved
        if not _reposition_moved:
            log("GRASP", "  Reposition FAILED — next round arm=0 only (cuRobo hang avoidance)")
        else:
            log("GRASP", "  Reposition OK — next round arm=1 allowed")

        # Re-find from position
        try:
            find_object_torso_rotate(found_name)
        except:
            try:
                find_object_base_rotate(found_name)
            except:
                pass
        try:
            reset_torso()
        except:
            pass
        get_env_observation()
        save_current_observation(f"round_{grasp_round+1}_retry")

# =========================================================================
# BLOCK 4: Re-navigate via get_navigation_pose + retry
# (Multi-angle is now in Block 3's between-round logic)
# =========================================================================
if not grasped and remaining() > 200 and radio_pos is not None:
    log("RECOVERY", "=== BLOCK 4: Re-navigate + Retry ===")

    try:
        reset_torso()
    except:
        pass

    # Re-detect everything fresh for a new navigation attempt
    try:
        find_object_base_rotate(found_name)
    except:
        pass
    try:
        reset_torso()
    except:
        pass

    radio_result4 = safe_get_pose(found_name, retries=2)
    P_table4 = None
    for tname in TABLE_NAMES:
        t_result4 = safe_get_pose(tname, retries=1)
        if t_result4[0] is not None and t_result4[3] is not None:
            t_pts4 = np.asarray(t_result4[3])
            if len(t_pts4) > 50:
                P_table4 = t_pts4
                break

    if radio_result4[0] is not None and radio_result4[3] is not None and P_table4 is not None:
        try:
            P_radio4 = np.asarray(radio_result4[3])
            nav_goal4 = get_navigation_pose(P_table4, P_radio4)
            log("RECOVERY", f"Re-nav goal: [{nav_goal4[0]:.2f}, {nav_goal4[1]:.2f}]")
            safe_navigate(nav_goal4, "recovery_nav")
        except Exception as e:
            log("RECOVERY", f"Re-nav error: {e}")

    # Re-find and grasp (1 pose × 2 arms)
    try:
        find_object_torso_rotate(found_name)
    except:
        pass
    try:
        reset_torso()
    except:
        pass
    get_env_observation()
    save_current_observation("recovery")

    grasp_result4 = None
    try:
        grasp_result4 = sample_grasp_pose(found_name)
    except:
        pass

    if grasp_result4 is not None and grasp_result4[0] is not None:
        pregrasp_poses, grasp_poses = grasp_result4
        # v3: arm=0 first in recovery (arm=1 more likely to trigger cuRobo hang)
        for arm in [0, 1]:
            if grasped or remaining() < 120:
                break
            open_gripper(arm=arm)
            try:
                grasp_object(pregrasp_poses[0], grasp_poses[0], found_name, arm=arm)
            except Exception as e:
                log("RECOVERY", f"  Error: {e}")
            lift_arm(arm=arm)
            in_hand = check_object_in_hand(arm=arm)
            log("RECOVERY", f"  arm={arm}: in_hand={in_hand}")
            if in_hand:
                grasped = True
                log("RECOVERY", "SUCCESS from re-navigation!")
                break
            else:
                open_gripper(arm=arm)

# =========================================================================
# BLOCK 5: Last resort — direct approach from opposite side
# =========================================================================
if not grasped and remaining() > 200 and radio_pos is not None:
    log("LAST", "=== BLOCK 5: Last Resort — Opposite Approach ===")
    try:
        reset_torso()
    except:
        pass

    # Navigate to the opposite side of the radio
    rp, _, _ = get_robot_position()
    rx, ry = float(radio_pos[0]), float(radio_pos[1])
    curr_ang = float(np.arctan2(ry - rp[1], rx - rp[0]))
    # Go to opposite side
    opp_x = rx + 0.70 * np.cos(curr_ang + np.pi)
    opp_y = ry + 0.70 * np.sin(curr_ang + np.pi)
    opp_yaw = float(np.arctan2(ry - opp_y, rx - opp_x))
    log("LAST", f"Approaching from opposite: [{opp_x:.2f}, {opp_y:.2f}]")
    safe_navigate([opp_x, opp_y, opp_yaw], "last_opposite")

    # Re-find and grasp
    try:
        find_object_torso_rotate(found_name)
    except:
        try:
            find_object_base_rotate(found_name)
        except:
            pass
    try:
        reset_torso()
    except:
        pass
    get_env_observation()
    save_current_observation("last_resort")

    grasp_result5 = None
    try:
        grasp_result5 = sample_grasp_pose(found_name)
    except:
        pass

    if grasp_result5 is not None and grasp_result5[0] is not None:
        pregrasp_poses, grasp_poses = grasp_result5
        # v3: arm=0 first in last resort (safer against cuRobo hangs)
        for arm in [0, 1]:
            if grasped or remaining() < 120:
                break
            open_gripper(arm=arm)
            try:
                grasp_object(pregrasp_poses[0], grasp_poses[0], found_name, arm=arm)
            except:
                pass
            lift_arm(arm=arm)
            in_hand = check_object_in_hand(arm=arm)
            log("LAST", f"  arm={arm}: in_hand={in_hand}")
            if in_hand:
                grasped = True
                break
            else:
                open_gripper(arm=arm)

# =========================================================================
# FINAL
# =========================================================================
rgb, depth = get_env_observation()
save_current_observation("final")

if grasped:
    log("DONE", "SUCCESS!")
    for arm in [0, 1]:
        ih = check_object_in_hand(arm=arm)
        if ih:
            log("DONE", f"Confirmed in hand: arm={arm}")
            break
else:
    log("DONE", "FAILED — could not grasp radio")

log("DONE", f"Total time: {elapsed():.0f}s")
save_current_observation("end")