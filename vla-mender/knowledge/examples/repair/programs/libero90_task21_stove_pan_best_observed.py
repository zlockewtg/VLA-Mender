
import math
import numpy as np

RAW_GRASP_EXPERIMENT = False
WRIST_HANDLE_SEARCH = True
MIN_CONFIDENT_HANDLE_GRIP_WIDTH = 0.140


def quat_from_matrix(R):
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1e-9)


def make_topdown_quat(yaw_deg=90.0):
    yaw = math.radians(float(yaw_deg))
    cz = math.cos(yaw)
    sz = math.sin(yaw)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    return quat_from_matrix(rz @ down)


def fresh_agentview():
    obs = get_observation()
    cam = obs["agentview"]
    depth = cam["images"]["depth"]
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    return obs, cam["images"]["rgb"], depth, cam["intrinsics"], cam["pose_mat"]


def fresh_camera(camera_name):
    obs = get_observation()
    cam = obs[camera_name]
    depth = cam["images"]["depth"]
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    return obs, cam["images"]["rgb"], depth, cam["intrinsics"], cam["pose_mat"]


def world_points_from_candidate(candidate, depth, K, E):
    mask = np.asarray(candidate["mask"], dtype=np.uint8)
    pts = mask_to_world_points(mask, depth, K, E)
    if pts is None:
        return None
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    return pts[np.all(np.isfinite(pts), axis=1)]


def qvalue(values, quantile):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    arr = np.sort(arr)
    idx = int(round((arr.size - 1) * float(quantile)))
    idx = max(0, min(idx, arr.size - 1))
    return float(arr[idx])


def qcenter(points):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.array(
        [qvalue(pts[:, 0], 0.5), qvalue(pts[:, 1], 0.5), qvalue(pts[:, 2], 0.5)],
        dtype=np.float64,
    )


def localize_pan_and_handle(rgb, depth, K, E):
    pan_pts = None
    pan_commit = None
    pan_mask = None
    for prompt in ["frying pan", "pan", "skillet"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for cand in sorted(masks or [], key=lambda item: float(item.get("score", 0.0)), reverse=True)[:5]:
            if float(cand.get("score", 0.0)) < 0.20:
                continue
            pts = world_points_from_candidate(cand, depth, K, E)
            if pts is None or len(pts) < 250:
                continue
            center = qcenter(pts)
            if 0.35 < center[0] < 0.85 and -0.45 < center[1] < 0.35 and -0.04 < center[2] < 0.12:
                pan_pts = pts
                pan_mask = np.asarray(cand["mask"], dtype=np.uint8)
                pan_commit = commit_target_mask(rgb, cand, "frying_pan_object")
                print("pan prompt", prompt, "score", float(cand.get("score", 0.0)), "center", center, flush=True)
                break
        if pan_pts is not None:
            break
    if pan_pts is None:
        raise RuntimeError("frying pan not localized")

    med_y = qvalue(pan_pts[:, 1], 0.5)
    body_pts = pan_pts[pan_pts[:, 1] < med_y]
    if len(body_pts) < 50:
        body_pts = pan_pts
    body_center = qcenter(body_pts)

    # Follow the KS3 pan examples: the usable handle is the high-Y, low-Z tail.
    # High-Z "handle" masks in this scene often land on the moka pot or stove parts.
    y_cut = qvalue(pan_pts[:, 1], 0.72)
    z_cut = min(0.050, qvalue(pan_pts[:, 2], 0.82))
    best_handle_pts = pan_pts[(pan_pts[:, 1] >= y_cut) & (pan_pts[:, 2] <= z_cut)]
    if len(best_handle_pts) < 30:
        y_cut = qvalue(pan_pts[:, 1], 0.64)
        best_handle_pts = pan_pts[(pan_pts[:, 1] >= y_cut) & (pan_pts[:, 2] <= 0.052)]
    if len(best_handle_pts) < 30:
        best_handle_pts = pan_pts[pan_pts[:, 1] >= qvalue(pan_pts[:, 1], 0.70)]

    best_prompt_handle = None
    best_prompt_quality = -1.0e9
    pan_x0 = qvalue(pan_pts[:, 0], 0.0)
    pan_x1 = qvalue(pan_pts[:, 0], 1.0)
    pan_y0 = qvalue(pan_pts[:, 1], 0.0)
    pan_y1 = qvalue(pan_pts[:, 1], 1.0)
    for hprompt in ["frying pan handle", "pan handle"]:
        masks = segment_sam3_text_prompt(rgb, hprompt)
        for cand in sorted(masks or [], key=lambda item: float(item.get("score", 0.0)), reverse=True)[:6]:
            if float(cand.get("score", 0.0)) < 0.20:
                continue
            pts = world_points_from_candidate(cand, depth, K, E)
            if pts is None or len(pts) < 30:
                continue
            low_pts = pts[pts[:, 2] < 0.050]
            if len(low_pts) < 100:
                continue
            center = qcenter(low_pts)
            if not (pan_x0 - 0.02 < center[0] < pan_x1 + 0.02 and pan_y0 - 0.02 < center[1] < pan_y1 + 0.05):
                continue
            x_ext = qvalue(low_pts[:, 0], 1.0) - qvalue(low_pts[:, 0], 0.0)
            y_ext = qvalue(low_pts[:, 1], 1.0) - qvalue(low_pts[:, 1], 0.0)
            top_z = qvalue(low_pts[:, 2], 0.95)
            if x_ext > 0.10 or top_z > 0.052:
                continue
            quality = float(len(low_pts)) - 1000.0 * x_ext + 100.0 * y_ext - 500.0 * max(0.0, top_z - 0.045)
            if quality > best_prompt_quality:
                best_prompt_quality = quality
                best_prompt_handle = low_pts
                print("prompt handle candidate", hprompt, "score", float(cand.get("score", 0.0)), "center", center, flush=True)
    if best_prompt_handle is not None:
        best_handle_pts = best_prompt_handle

    handle_center = qcenter(best_handle_pts)
    handle_top_z = qvalue(best_handle_pts[:, 2], 1.0)
    print("handle center", handle_center, "top_z", handle_top_z, "body", body_center, flush=True)
    return pan_commit, pan_mask, pan_pts, body_center, best_handle_pts, handle_center, handle_top_z


def localize_burner(rgb, depth, K, E):
    anchor = np.array([0.62, 0.20], dtype=np.float64)
    best = None
    support = None
    for prompt in ["red stove burner", "stove burner", "burner", "stove top"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for cand in sorted(masks or [], key=lambda item: float(item.get("score", 0.0)), reverse=True)[:8]:
            pts = world_points_from_candidate(cand, depth, K, E)
            if pts is None or len(pts) < 80:
                continue
            center = qcenter(pts)
            if not (0.45 < center[0] < 0.78 and 0.03 < center[1] < 0.34 and -0.05 < center[2] < 0.12):
                continue
            top_z = qvalue(pts[:, 2], 0.90)
            top_pts = pts[pts[:, 2] > top_z - 0.02]
            if len(top_pts) >= 20:
                xy = np.array([
                    0.5 * (qvalue(top_pts[:, 0], 0.05) + qvalue(top_pts[:, 0], 0.95)),
                    0.5 * (qvalue(top_pts[:, 1], 0.05) + qvalue(top_pts[:, 1], 0.95)),
                ], dtype=np.float64)
            else:
                xy = center[:2].copy()
            distance = float(np.linalg.norm(xy - anchor))
            prompt_bonus = 0.0 if "burner" in prompt else 0.05
            score = distance + prompt_bonus
            if best is None or score < best[0]:
                best = (score, prompt, cand, xy, top_z)
            x_ext = qvalue(pts[:, 0], 0.95) - qvalue(pts[:, 0], 0.05)
            y_ext = qvalue(pts[:, 1], 0.95) - qvalue(pts[:, 1], 0.05)
            radius = 0.5 * max(x_ext, y_ext)
            if distance < 0.11 and 0.060 <= radius <= 0.180:
                support_score = radius - 0.5 * distance + (0.03 if "stove" in prompt else 0.0)
                if support is None or support_score > support[0]:
                    support = (support_score, prompt, cand, radius)
    if best is None:
        raise RuntimeError("active stove burner not localized")
    _, prompt, cand, xy, top_z = best
    support_prompt = prompt
    support_cand = cand
    support_radius = 0.0
    if support is not None:
        _, support_prompt, support_cand, support_radius = support
    committed = commit_target_mask(rgb, support_cand, "active_stove_burner")
    target = ground_placement_target(get_observation(), "agentview", committed)
    target_center = np.asarray(target.get("center", [xy[0], xy[1], top_z]), dtype=np.float64)
    print("burner prompt", prompt, "xy", xy, "guard_xy", target_center[:2], "top_z", top_z, "support", support_prompt, support_radius, flush=True)
    return target, xy, target_center[:2].copy(), top_z


def localize_live_pan_body_center(reference_xy):
    obs, rgb, depth, K, E = fresh_agentview()
    reference_xy = np.asarray(reference_xy, dtype=np.float64).reshape(2)
    best = None
    for prompt in ["pan body", "frying pan"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for cand in sorted(masks or [], key=lambda item: float(item.get("score", 0.0)), reverse=True)[:6]:
            pts = world_points_from_candidate(cand, depth, K, E)
            if pts is None or len(pts) < 80:
                continue
            center = qcenter(pts)
            if not (0.35 < center[0] < 0.85 and -0.10 < center[1] < 0.38 and 0.035 < center[2] < 0.35):
                continue
            x_ext = qvalue(pts[:, 0], 0.95) - qvalue(pts[:, 0], 0.05)
            y_ext = qvalue(pts[:, 1], 0.95) - qvalue(pts[:, 1], 0.05)
            radius = 0.5 * max(x_ext, y_ext)
            if radius < 0.045:
                continue
            score = radius - 0.25 * float(np.linalg.norm(center[:2] - reference_xy))
            if best is None or score > best[0]:
                best = (score, prompt, center, radius)
    if best is None:
        print("live pan body center unavailable", flush=True)
        return None
    _, prompt, center, radius = best
    print("live pan body", prompt, center, "radius", radius, flush=True)
    return center


def step_to(position, quat, max_steps=220, required=False):
    try:
        goto_pose_osc(np.asarray(position, dtype=np.float64), np.asarray(quat, dtype=np.float64), max_steps=max_steps)
        return True
    except Exception as exc:
        print("step_to warning", str(exc), flush=True)
        if required:
            raise
        return False


def result_gripper_width(result):
    after = result.get("grasp_state_after") if isinstance(result, dict) else None
    if not isinstance(after, dict):
        return None
    try:
        return float(after.get("gripper_width_normalized"))
    except Exception:
        return None


def localize_stove_knob_from_agentview(fallback_xy):
    _, rgb, depth, K, E = fresh_agentview()
    best = None
    for prompt in ["black stove knob", "stove knob", "knob", "stove switch"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for cand in sorted(masks or [], key=lambda item: float(item.get("score", 0.0)), reverse=True)[:8]:
            if float(cand.get("score", 0.0)) < 0.05:
                continue
            pts = world_points_from_candidate(cand, depth, K, E)
            if pts is None or len(pts) < 40:
                continue
            center = qcenter(pts)
            if not (0.39 <= center[0] <= 0.56 and 0.10 <= center[1] <= 0.30):
                continue
            z_top = qvalue(pts[:, 2], 1.0)
            if not (0.015 <= z_top <= 0.10):
                continue
            ext = np.array(
                [
                    qvalue(pts[:, 0], 0.95) - qvalue(pts[:, 0], 0.05),
                    qvalue(pts[:, 1], 0.95) - qvalue(pts[:, 1], 0.05),
                ],
                dtype=np.float64,
            )
            if float(np.max(ext)) > 0.16:
                continue
            fallback_dist = float(np.linalg.norm(center[:2] - fallback_xy))
            score = float(cand.get("score", 0.0)) - 0.5 * fallback_dist
            if best is None or score > best[0]:
                best = (score, center, float(z_top), prompt)
        if best is not None:
            break
    if best is None:
        print("stove_knob_localize_fallback", fallback_xy, flush=True)
        return np.array([fallback_xy[0], fallback_xy[1], 0.048], dtype=np.float64), 0.048
    _, center, z_top, prompt = best
    print("stove_knob_localized", prompt, "center", center, "top_z", z_top, flush=True)
    return center, z_top


def turn_stove_knob_from_burner(burner_xy, burner_top_z):
    burner_xy = np.asarray(burner_xy, dtype=np.float64).reshape(2)
    fallback_xy = burner_xy + np.array([-0.150, 0.000], dtype=np.float64)
    knob_center, knob_top_z = localize_stove_knob_from_agentview(fallback_xy)
    knob_xy = np.asarray(knob_center[:2], dtype=np.float64)
    turn_yaw = 0.0
    turn_quat = make_topdown_quat(turn_yaw)
    contact_z = float(np.clip(float(knob_top_z) - 0.012, 0.035, 0.060))
    print("stove_knob_turn_start", "knob_xy", knob_xy, "top_z", knob_top_z, flush=True)
    open_gripper()
    step_to([knob_xy[0], knob_xy[1], 0.150], turn_quat, max_steps=120)
    step_to([knob_xy[0], knob_xy[1], contact_z + 0.025], turn_quat, max_steps=80)
    step_to([knob_xy[0], knob_xy[1], contact_z], turn_quat, max_steps=80)
    close_gripper()
    for _ in range(6):
        obs = get_observation()
    width = float(np.asarray(obs.get("robot_cartesian_pos", [0.0] * 8), dtype=np.float64).reshape(-1)[-1])
    print("stove_knob_grip_width", width, flush=True)
    if width > 0.05:
        for yaw in [35.0, 70.0, 105.0, 140.0, 175.0]:
            print("stove_knob_wrist_yaw", yaw, flush=True)
            step_to([knob_xy[0], knob_xy[1], contact_z], make_topdown_quat(yaw), max_steps=45)
    else:
        print("stove_knob_rotation_skipped_no_grip", flush=True)
    open_gripper()
    step_to([knob_xy[0], knob_xy[1], 0.170], make_topdown_quat(175.0), max_steps=80)
    for _ in range(8):
        get_observation()


def wrist_refine_handle(body_center, rough_handle_center, pan_points, preferred_yaw, prefer_short_scan=False):
    body_center = np.asarray(body_center, dtype=np.float64).reshape(3)
    rough_handle_center = np.asarray(rough_handle_center, dtype=np.float64).reshape(3)
    pan_points = np.asarray(pan_points, dtype=np.float64).reshape(-1, 3)
    pan_x0 = qvalue(pan_points[:, 0], 0.02)
    pan_x1 = qvalue(pan_points[:, 0], 0.98)
    pan_y0 = qvalue(pan_points[:, 1], 0.02)
    pan_y1 = qvalue(pan_points[:, 1], 0.98)
    handle_dir = rough_handle_center[:2] - body_center[:2]
    norm = float(np.linalg.norm(handle_dir))
    if norm > 1.0e-6:
        handle_dir = handle_dir / norm
    else:
        handle_dir = np.array([0.0, 1.0], dtype=np.float64)

    scan_centers = [
        rough_handle_center[:2],
        rough_handle_center[:2] + 0.035 * handle_dir,
        rough_handle_center[:2] - 0.030 * handle_dir,
        0.50 * (rough_handle_center[:2] + body_center[:2]),
    ]
    best = None
    for scan_idx, xy in enumerate(scan_centers):
        xy = np.asarray(xy, dtype=np.float64).reshape(2)
        xy[0] = float(np.clip(xy[0], 0.36, 0.74))
        xy[1] = float(np.clip(xy[1], -0.38, 0.24))
        scan_z = 0.335 if prefer_short_scan else 0.255
        if prefer_short_scan or rough_handle_center[2] < 0.032:
            scan_yaws = [0.0]
            if abs(float(preferred_yaw)) > 1.0e-6:
                scan_yaws.append(float(preferred_yaw))
        else:
            scan_yaws = [preferred_yaw]
        moved_for_scan = False
        for scan_yaw in scan_yaws:
            scan_quat = make_topdown_quat(scan_yaw)
            print("wrist_handle_scan_pose", scan_idx, xy, "z", scan_z, "yaw", scan_yaw, flush=True)
            moved_for_scan = step_to([xy[0], xy[1], scan_z], scan_quat, max_steps=170)
            if moved_for_scan:
                break
        if not moved_for_scan:
            print("wrist_handle_scan_unreachable", scan_idx, xy, flush=True)
            continue
        for _ in range(3):
            get_observation()
        obs_w, rgb_w, depth_w, K_w, E_w = fresh_camera("robot0_eye_in_hand")
        for prompt in ["frying pan handle", "pan handle", "black pan handle"]:
            masks = segment_sam3_text_prompt(rgb_w, prompt)
            for cand in sorted(masks or [], key=lambda item: float(item.get("score", 0.0)), reverse=True)[:8]:
                score_sam = float(cand.get("score", 0.0))
                if score_sam < 0.04:
                    continue
                pts = world_points_from_candidate(cand, depth_w, K_w, E_w)
                if pts is None or len(pts) < 18:
                    continue
                pts = pts[(pts[:, 2] > -0.030) & (pts[:, 2] < 0.090)]
                if len(pts) < 18:
                    continue
                center = qcenter(pts)
                if not (pan_x0 - 0.08 < center[0] < pan_x1 + 0.08 and pan_y0 - 0.08 < center[1] < pan_y1 + 0.10):
                    continue
                if float(np.linalg.norm(center[:2] - rough_handle_center[:2])) > 0.145:
                    continue
                x_ext = qvalue(pts[:, 0], 0.95) - qvalue(pts[:, 0], 0.05)
                y_ext = qvalue(pts[:, 1], 0.95) - qvalue(pts[:, 1], 0.05)
                z_top = qvalue(pts[:, 2], 0.95)
                slender_bonus = max(x_ext, y_ext) - 0.75 * min(x_ext, y_ext)
                toward_tip = float(np.dot(center[:2] - body_center[:2], handle_dir))
                dist_penalty = float(np.linalg.norm(center[:2] - rough_handle_center[:2]))
                quality = score_sam + 1.8 * slender_bonus + 0.45 * toward_tip - 1.2 * dist_penalty - 0.8 * max(0.0, z_top - 0.055)
                print(
                    "wrist_handle_candidate",
                    scan_idx,
                    prompt,
                    "sam",
                    score_sam,
                    "center",
                    center,
                    "ext",
                    [x_ext, y_ext],
                    "z_top",
                    z_top,
                    "quality",
                    quality,
                    flush=True,
                )
                if best is None or quality > best[0]:
                    best = (quality, pts, center, z_top, prompt, scan_idx)
        if prefer_short_scan and best is not None:
            print("wrist_handle_short_scan_stop", scan_idx, flush=True)
            break
    if best is None:
        print("wrist_handle_refine_unavailable", flush=True)
        return None, None, None
    _, pts, center, z_top, prompt, scan_idx = best
    print("wrist_handle_selected", prompt, "scan", scan_idx, "center", center, "top_z", z_top, flush=True)
    return pts, center, z_top


def raw_grasp_if_unheld(obs, prompts, pregrasp, grasp, quat):
    before = estimate_grasp_state(obs, prompts)
    print("raw_grasp_state_before", before, flush=True)
    if before.get("state") == "held":
        return {"status": "already_held", "executed": False, "grasp_state_before": before}
    if before.get("state") != "not_held":
        return {"status": "ambiguous_hold", "executed": False, "grasp_state_before": before}
    open_gripper()
    pregrasp = np.asarray(pregrasp, dtype=np.float64)
    grasp = np.asarray(grasp, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)
    step_to(pregrasp, quat, max_steps=170, required=True)
    approach = grasp.copy()
    approach[2] = float(max(approach[2] + 0.030, 0.080))
    step_to(approach, quat, max_steps=120, required=False)
    step_to(grasp, quat, max_steps=170, required=True)
    close_gripper()
    for _ in range(12):
        get_observation()
    after = estimate_grasp_state(get_observation(), prompts)
    print("raw_grasp_state_after", after, flush=True)
    return {
        "status": "grasped" if after.get("state") == "held" else "grasp_unverified",
        "executed": True,
        "grasp_state_before": before,
        "grasp_state_after": after,
    }


def plan_pan_grasp_with_graspnet(pan_mask, depth, K, E, pan_pts, desired_xy=None):
    if pan_mask is None:
        return None, None
    grasps, scores = plan_grasp(
        depth=depth,
        intrinsics=K,
        segmentation=np.asarray(pan_mask, dtype=np.int32),
    )
    if grasps is None or len(grasps) == 0:
        print("GraspNet returned no pan grasps", flush=True)
        return None, None
    best_world = None
    best_score = None
    if desired_xy is not None:
        desired_xy = np.asarray(desired_xy, dtype=np.float64).reshape(2)
        best_metric = 1.0e9
        for i, score in enumerate(scores):
            world = E @ grasps[i]
            pos, _ = decompose_transform(world)
            pos = np.asarray(pos, dtype=np.float64).reshape(3)
            if not (0.42 < pos[0] < 0.74 and -0.34 < pos[1] < -0.14 and 0.015 < pos[2] < 0.12):
                continue
            metric = float(np.linalg.norm(pos[:2] - desired_xy)) - 0.010 * float(score)
            print("GraspNet candidate", i, pos, "score", float(score), "metric", metric, flush=True)
            if metric < best_metric:
                best_metric = metric
                best_world = world
                best_score = float(score)
    if best_world is None:
        best_world, best_score = select_top_down_grasp(grasps, scores, E, vertical_threshold=0.35)
    if best_world is None:
        best_i = 0
        best_s = float(scores[0])
        for i, score in enumerate(scores):
            s = float(score)
            if s > best_s:
                best_i = i
                best_s = s
        best_world = E @ grasps[best_i]
        best_score = best_s
    gpos, gquat = decompose_transform(best_world)
    gpos = np.asarray(gpos, dtype=np.float64).reshape(3)
    gquat = np.asarray(gquat, dtype=np.float64).reshape(4)
    pan_center = qcenter(pan_pts)
    if float(np.linalg.norm(gpos[:2] - pan_center[:2])) > 0.18:
        print("GraspNet grasp rejected, xy too far", gpos, "pan", pan_center, flush=True)
        return None, None
    pan_top = qvalue(pan_pts[:, 2], 0.95)
    gpos[2] = float(np.clip(gpos[2] + 0.012, max(pan_top - 0.005, 0.020), pan_top + 0.025))
    print("GraspNet pan grasp", gpos, "score", float(best_score), flush=True)
    return gpos, gquat


print("=== FM-01 stove-pan suffix repair ===", flush=True)
obs0, rgb0, depth0, K0, E0 = fresh_agentview()
pan_commit, pan_mask, pan_pts, body_center, handle_pts, handle_center, handle_top_z = localize_pan_and_handle(rgb0, depth0, K0, E0)
target_commit, burner_xy, burner_guard_xy, burner_top_z = localize_burner(rgb0, depth0, K0, E0)
agent_handle_pts = handle_pts.copy()
agent_handle_center = handle_center.copy()
agent_handle_top_z = float(handle_top_z)
agent_handle_after_wrist = False
wrist_far_front_handle = False

x_extent = qvalue(handle_pts[:, 0], 1.0) - qvalue(handle_pts[:, 0], 0.0)
y_extent = qvalue(handle_pts[:, 1], 1.0) - qvalue(handle_pts[:, 1], 0.0)
yaw = 90.0 if y_extent >= x_extent else 0.0
if abs(x_extent - y_extent) < 0.012:
    yaw = 90.0

if WRIST_HANDLE_SEARCH:
    refined_handle_pts, refined_handle_center, refined_handle_top_z = wrist_refine_handle(
        body_center, handle_center, pan_pts, yaw, prefer_short_scan=(agent_handle_center[1] > -0.130)
    )
    if refined_handle_pts is not None:
        handle_pts = refined_handle_pts
        handle_center = refined_handle_center
        handle_top_z = refined_handle_top_z
        x_extent = qvalue(handle_pts[:, 0], 1.0) - qvalue(handle_pts[:, 0], 0.0)
        y_extent = qvalue(handle_pts[:, 1], 1.0) - qvalue(handle_pts[:, 1], 0.0)
        yaw = 90.0 if y_extent >= x_extent else 0.0
        if abs(x_extent - y_extent) < 0.012:
            yaw = 90.0
        print("using_wrist_handle", "center", handle_center, "top_z", handle_top_z, "yaw", yaw, flush=True)
        handle_view_delta = float(np.linalg.norm(agent_handle_center[:2] - handle_center[:2]))
        wrist_far_front_handle = (
            agent_handle_center[0] < 0.620
            and agent_handle_center[1] > -0.130
            and agent_handle_top_z >= 0.032
            and handle_view_delta > 0.040
        )
        if agent_handle_center[1] > -0.130 and agent_handle_top_z >= 0.032 and not wrist_far_front_handle:
            handle_pts = agent_handle_pts
            handle_center = agent_handle_center
            handle_top_z = agent_handle_top_z
            x_extent = qvalue(handle_pts[:, 0], 1.0) - qvalue(handle_pts[:, 0], 0.0)
            y_extent = qvalue(handle_pts[:, 1], 1.0) - qvalue(handle_pts[:, 1], 0.0)
            yaw = 90.0 if y_extent >= x_extent else 0.0
            if abs(x_extent - y_extent) < 0.012:
                yaw = 90.0
            agent_handle_after_wrist = True
            print("using_agentview_handle_after_wrist", "center", handle_center, "top_z", handle_top_z, "yaw", yaw, flush=True)
quat = make_topdown_quat(yaw)
grasp_quat = quat

toward_body = body_center[:2] - handle_center[:2]
norm = float(np.linalg.norm(toward_body))
if norm > 1.0e-6:
    toward_body = toward_body / norm
else:
    toward_body = np.array([0.0, -1.0], dtype=np.float64)
graspnet_pos, graspnet_quat = None, None
if handle_top_z < 0.032:
    desired_body_grasp = np.array([body_center[0] + 0.025, body_center[1] + 0.035], dtype=np.float64)
    graspnet_pos, graspnet_quat = plan_pan_grasp_with_graspnet(
        pan_mask, depth0, K0, E0, pan_pts, desired_xy=desired_body_grasp
    )
    graspnet_pos = None
if graspnet_pos is not None:
    grasp = graspnet_pos.copy()
    grasp_xy = grasp[:2].copy()
    grasp_quat = graspnet_quat.copy()
else:
    if handle_top_z < 0.032:
        grasp_xy = np.array([handle_center[0] - 0.014, handle_center[1] + 0.012], dtype=np.float64)
        grasp_z = 0.045
        if graspnet_quat is not None:
            grasp_quat = graspnet_quat.copy()
    else:
        grasp_xy = handle_center[:2] + 0.035 * toward_body
        grasp_z = float(max(handle_top_z - 0.020, 0.005))
    grasp = np.array([grasp_xy[0], grasp_xy[1], grasp_z], dtype=np.float64)
grasp_z = float(grasp[2])
pregrasp = np.array([grasp_xy[0], grasp_xy[1], max(0.16, float(grasp[2]) + 0.12)], dtype=np.float64)
print("grasp", grasp, "yaw", yaw, flush=True)

grasp_prompts = ["frying pan", "frying pan handle", "pan handle"]
obs_for_grasp = get_observation()
if RAW_GRASP_EXPERIMENT:
    grasp_result = raw_grasp_if_unheld(obs_for_grasp, grasp_prompts, pregrasp, grasp, grasp_quat)
else:
    grasp_result = grasp_if_unheld(obs_for_grasp, grasp_prompts, pregrasp, grasp, grasp_quat)
print("grasp_result", grasp_result, flush=True)
if grasp_result.get("status") == "motion_failed":
    grasp_retry = grasp.copy()
    grasp_retry[2] = float(grasp_retry[2] + (0.010 if handle_top_z < 0.032 else 0.015))
    pregrasp_retry = np.array(
        [grasp_xy[0], grasp_xy[1], max(0.17, float(grasp_retry[2]) + 0.12)],
        dtype=np.float64,
    )
    if RAW_GRASP_EXPERIMENT:
        grasp_result = raw_grasp_if_unheld(get_observation(), grasp_prompts, pregrasp_retry, grasp_retry, grasp_quat)
    else:
        grasp_result = grasp_if_unheld(
            get_observation(),
            grasp_prompts,
            pregrasp_retry,
            grasp_retry,
            grasp_quat,
        )
    print("grasp_retry_result", grasp_result, flush=True)
    if grasp_result.get("status") in ("grasped", "grasp_unverified", "already_held"):
        grasp = grasp_retry
        grasp_xy = grasp[:2].copy()
        grasp_z = float(grasp[2])
if WRIST_HANDLE_SEARCH and grasp_result.get("status") == "motion_failed":
    handle_axis = handle_center[:2] - body_center[:2]
    axis_norm = float(np.linalg.norm(handle_axis))
    if axis_norm > 1.0e-6:
        handle_axis = handle_axis / axis_norm
    else:
        handle_axis = np.array([0.0, 1.0], dtype=np.float64)
    wrist_retry_specs = [
        (-0.010, min(handle_top_z + 0.006, 0.055)),
        (0.012, min(handle_top_z + 0.010, 0.058)),
    ]
    tried = []
    retry_yaws = [yaw]
    for axis_offset, retry_z in wrist_retry_specs:
        grasp_retry_xy = handle_center[:2] + float(axis_offset) * handle_axis
        grasp_retry_xy[0] = float(np.clip(grasp_retry_xy[0], 0.36, 0.74))
        grasp_retry_xy[1] = float(np.clip(grasp_retry_xy[1], -0.38, 0.28))
        grasp_retry = np.array([grasp_retry_xy[0], grasp_retry_xy[1], float(retry_z)], dtype=np.float64)
        pregrasp_retry = np.array(
            [grasp_retry_xy[0], grasp_retry_xy[1], max(0.17, float(retry_z) + 0.12)],
            dtype=np.float64,
        )
        key = (round(float(grasp_retry_xy[0]), 3), round(float(grasp_retry_xy[1]), 3), round(float(retry_z), 3))
        if key in tried:
            continue
        tried.append(key)
        for retry_yaw in retry_yaws:
            retry_quat = make_topdown_quat(retry_yaw)
            print("wrist_handle_grasp_retry", "axis_offset", axis_offset, "grasp", grasp_retry, "yaw", retry_yaw, flush=True)
            grasp_result = grasp_if_unheld(
                get_observation(),
                grasp_prompts,
                pregrasp_retry,
                grasp_retry,
                retry_quat,
            )
            print("wrist_handle_grasp_retry_result", grasp_result, flush=True)
            if grasp_result.get("status") in ("grasped", "grasp_unverified", "already_held"):
                grasp = grasp_retry
                grasp_xy = grasp_retry_xy.copy()
                grasp_z = float(grasp[2])
                grasp_quat = retry_quat
                quat = retry_quat
                break
        if grasp_result.get("status") in ("grasped", "grasp_unverified", "already_held"):
            break
weak_grasp_width = result_gripper_width(grasp_result)
if (
    agent_handle_after_wrist
    and handle_top_z >= 0.032
    and grasp_result.get("status") in ("grasped", "grasp_unverified")
    and weak_grasp_width is not None
    and weak_grasp_width < MIN_CONFIDENT_HANDLE_GRIP_WIDTH
):
    print("weak_handle_grasp_width", weak_grasp_width, "retrying_centerline_handle_grasp", flush=True)
    handle_axis = handle_center[:2] - body_center[:2]
    axis_norm = float(np.linalg.norm(handle_axis))
    if axis_norm > 1.0e-6:
        handle_axis = handle_axis / axis_norm
    else:
        handle_axis = np.array([0.0, 1.0], dtype=np.float64)
    step_to([grasp_xy[0], grasp_xy[1], max(0.16, grasp_z + 0.12)], grasp_quat, max_steps=100)
    open_gripper()
    for _ in range(4):
        get_observation()
    centerline_specs = [
        (-0.006, max(handle_top_z - 0.008, 0.026), yaw),
        (0.012, max(handle_top_z - 0.004, 0.030), yaw),
        (0.018, max(handle_top_z - 0.002, 0.032), 0.0),
    ]
    weak_retry_accepted = False
    for axis_offset, retry_z, retry_yaw in centerline_specs:
        retry_xy = handle_center[:2] + float(axis_offset) * handle_axis
        retry_xy[0] = float(np.clip(retry_xy[0], 0.36, 0.74))
        retry_xy[1] = float(np.clip(retry_xy[1], -0.38, 0.28))
        retry_grasp = np.array([retry_xy[0], retry_xy[1], float(retry_z)], dtype=np.float64)
        retry_pregrasp = np.array([retry_xy[0], retry_xy[1], max(0.17, float(retry_z) + 0.12)], dtype=np.float64)
        retry_quat = make_topdown_quat(retry_yaw)
        print("centerline_handle_grasp_retry", "axis_offset", axis_offset, "grasp", retry_grasp, "yaw", retry_yaw, flush=True)
        grasp_result = grasp_if_unheld(get_observation(), grasp_prompts, retry_pregrasp, retry_grasp, retry_quat)
        retry_width = result_gripper_width(grasp_result)
        print("centerline_handle_grasp_retry_result", grasp_result, "width", retry_width, flush=True)
        if grasp_result.get("status") in ("grasped", "grasp_unverified", "already_held") and (
            retry_width is None or retry_width >= MIN_CONFIDENT_HANDLE_GRIP_WIDTH
        ):
            grasp = retry_grasp
            grasp_xy = retry_xy.copy()
            grasp_z = float(grasp[2])
            grasp_quat = retry_quat
            quat = retry_quat
            weak_retry_accepted = True
            break
        if grasp_result.get("status") in ("grasped", "grasp_unverified"):
            step_to([retry_xy[0], retry_xy[1], max(0.17, float(retry_z) + 0.12)], retry_quat, max_steps=80)
            open_gripper()
            for _ in range(3):
                get_observation()
    if not weak_retry_accepted:
        grasp_result = {
            "status": "weak_grasp_rejected",
            "executed": True,
            "failure_reason": "all centerline handle retries kept a weak gripper width",
            "last_gripper_width": result_gripper_width(grasp_result),
        }
if grasp_result.get("status") not in ("grasped", "grasp_unverified", "already_held"):
    raise RuntimeError("pan grasp failed: " + str(grasp_result))
if handle_top_z < 0.032 and graspnet_quat is not None:
    quat = grasp_quat.copy()

lift_z = 0.46
for z in [max(grasp_z + 0.08, 0.14), 0.26, 0.36, lift_z]:
    step_to([grasp_xy[0], grasp_xy[1], z], quat)

body_offset = grasp_xy - body_center[:2]
target_xy = burner_xy + body_offset
if wrist_far_front_handle:
    target_xy = target_xy + np.array([-0.045, -0.020], dtype=np.float64)
target_xy[0] = float(np.clip(target_xy[0], 0.36, 0.74))
normal_y_max = 0.30 if body_offset[1] > 0.171 else 0.27
target_xy[1] = float(np.clip(target_xy[1], -0.45, normal_y_max if handle_top_z >= 0.032 else 0.30))
print("target_xy", target_xy, "body_offset", body_offset, "burner", burner_xy, flush=True)

current = np.asarray(get_robot_state(get_observation())["motion_target_position"], dtype=np.float64)
for frac in [0.25, 0.50, 0.75, 1.0]:
    xy_wp = current[:2] * (1.0 - frac) + target_xy * frac
    step_to([xy_wp[0], xy_wp[1], lift_z], quat)

if handle_top_z >= 0.032:
    live_body = localize_live_pan_body_center(burner_xy)
    if live_body is not None:
        correction = burner_xy - live_body[:2]
        corr_norm = float(np.linalg.norm(correction))
        if np.isfinite(corr_norm) and corr_norm <= 0.12:
            correction = np.clip(correction, -0.060, 0.060)
            state = get_robot_state(get_observation())
            pos = np.asarray(state["motion_target_position"], dtype=np.float64)
            print("high_align_correction", correction, "from", live_body[:2], "to", burner_xy, flush=True)
            step_to([pos[0] + correction[0], pos[1] + correction[1], lift_z], quat, max_steps=140)

release_offset = 0.085 if WRIST_HANDLE_SEARCH else 0.130
release_z = float(np.clip(burner_top_z + release_offset, 0.100 if WRIST_HANDLE_SEARCH else 0.145, 0.170))
for z in [0.34, 0.24, release_z]:
    if z >= release_z:
        step_to([target_xy[0], target_xy[1], z], quat)

obs_release = get_observation()
release = guarded_open_gripper(
    obs_release,
    ["frying pan", "pan body"],
    ["red stove burner", "stove burner", "burner"],
    "guarded_release",
    target_commit=target_commit,
)
print("release_result", release, flush=True)
for release_retry_idx in range(2):
    if not str(release.get("status")).startswith("blocked"):
        break
    geom = release.get("placement_geometry", {})
    obj = np.asarray(geom.get("object_center", [target_xy[0], target_xy[1], release_z]), dtype=np.float64)
    tgt = np.asarray(geom.get("target_prompt", target_xy), dtype=object)
    del tgt
    correction = burner_guard_xy - obj[:2]
    corr_norm = float(np.linalg.norm(correction))
    if np.isfinite(corr_norm) and corr_norm <= 0.095:
        correction = np.clip(correction, -0.045, 0.045)
        state = get_robot_state(get_observation())
        pos = np.asarray(state["motion_target_position"], dtype=np.float64)
        corrected_z = pos[2]
        z_clearance = float(geom.get("z_clearance", 0.0) or 0.0)
        if WRIST_HANDLE_SEARCH and z_clearance > 0.010:
            corrected_z = max(float(burner_top_z + 0.040), float(pos[2] - min(0.075, z_clearance - 0.004)))
        step_to([pos[0] + correction[0], pos[1] + correction[1], corrected_z], quat)
        release = guarded_open_gripper(
            get_observation(),
            ["frying pan", "pan body"],
            ["red stove burner", "stove burner", "burner"],
            "guarded_release",
            target_commit=target_commit,
        )
        print("release_retry", release_retry_idx + 1, release, flush=True)
    else:
        break
if release.get("status") not in ("opened", "already_released"):
    print("guarded_release_blocked_finish_for_task_predicate", release, flush=True)

if release.get("status") in ("opened", "already_released"):
    turn_stove_knob_from_burner(burner_xy, burner_top_z)

for _ in range(20):
    get_observation()
print("=== FM-01 suffix done ===", flush=True)
