"""Pick up a can using only the left arm."""

from __future__ import annotations

import os
import time
from typing import Any

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    capture_scene,
    current_run_dir,
    generate_side_grasp_candidates,
    rank_motion_candidates,
    staged_close_with_contact,
    verify_lift,
    write_json,
)


TASK_RESULT: dict[str, Any] = {
    "success": False,
    "reward": 0.0,
    "method": "left_arm_can_pickup",
    "why_stopped": None,
    "selected_detection": None,
    "selected_plan": None,
    "execution": None,
    "verification": None,
}


def get_task_info() -> dict[str, Any]:
    return TASK_RESULT


PROMPTS = ["can", "soda can", "aluminum can", "food can"]
TASK_NAME = "pick_can_left"
ARM = "left"


def _xyz(det: dict[str, Any] | None) -> list[float] | None:
    if not det:
        return None
    xyz = det.get("position_3d") or det.get("position")
    if not xyz or len(xyz) < 3:
        return None
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])]


def _bbox_ratio(det: dict[str, Any]) -> float | None:
    box = det.get("box_2d") or []
    if len(box) < 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    if w < 8.0 or h < 8.0:
        return None
    return h / max(w, 1.0)


def _workspace_ok(xyz: list[float]) -> bool:
    x, y, z = xyz
    return 0.18 <= x <= 1.05 and -0.55 <= y <= 0.55 and 0.74 <= z <= 1.10


def _select_can_detection(scene: dict[str, Any]) -> dict[str, Any] | None:
    scored: list[tuple[float, dict[str, Any]]] = []
    for det in scene.get("all_detections") or []:
        xyz = _xyz(det)
        if xyz is None or not _workspace_ok(xyz):
            continue
        ratio = _bbox_ratio(det)
        ratio_bonus = 0.0
        if ratio is not None:
            if 0.45 <= ratio <= 2.4:
                ratio_bonus = 0.15
            elif ratio > 3.2:
                continue
        prompt = str(det.get("prompt") or det.get("label") or "").lower()
        prompt_bonus = 0.2 if "can" in prompt else 0.0
        y_reach_bonus = 0.06 if xyz[1] >= -0.10 else 0.0
        score = float(det.get("score") or 0.0) + ratio_bonus + prompt_bonus + y_reach_bonus
        scored.append((score, det))
    if not scored:
        return None
    return sorted(scored, key=lambda item: item[0], reverse=True)[0][1]


def _extent_values(det: dict[str, Any]) -> tuple[float, float]:
    half = det.get("half_extents") or []
    if len(half) >= 3:
        radius = max(0.025, min(0.055, min(abs(float(half[0])), abs(float(half[1])))))
        half_height = max(0.035, min(0.085, abs(float(half[2]))))
        return radius, half_height
    return 0.034, 0.055


def _grasp_body_z(det: dict[str, Any]) -> float:
    xyz = _xyz(det)
    if xyz is None:
        raise ValueError("selected detection is missing position_3d")
    half = det.get("half_extents") or []
    if len(half) >= 3 and abs(float(half[2])) > 0.0:
        z_center = float(xyz[2])
        half_h = max(0.035, min(0.085, abs(float(half[2]))))
        return z_center - half_h + 2.0 * half_h * 0.65
    return float(xyz[2]) + 0.02


def _make_topdown_candidates(det: dict[str, Any]) -> list[dict[str, Any]]:
    xyz = _xyz(det)
    if xyz is None:
        return []
    radius, _half_height = _extent_values(det)
    z = _grasp_body_z(det)
    base = [float(xyz[0]), float(xyz[1]), z]
    hover_dz = 0.16
    lift_dz = 0.13
    candidates: list[dict[str, Any]] = []
    for idx, yaw in enumerate([90.0, 45.0, 135.0, 0.0, 180.0]):
        rpy = [0.0, 180.0, yaw]
        pre = [base[0], base[1], base[2] + hover_dz]
        lift = [base[0], base[1], base[2] + lift_dz]
        candidates.append(
            {
                "label": f"can_topdown_{idx}",
                "arm": ARM,
                "position": base,
                "rpy": rpy,
                "score": 1.0 - 0.05 * idx,
                "width": 2.0 * radius + 0.025,
                "pregrasp_pose": {"position": pre, "rpy": rpy},
                "grasp_pose": {"position": base, "rpy": rpy},
                "lift_pose": {"position": lift, "rpy": rpy},
                "source_detection": det,
                "strategy": "sim_can_topdown",
                "estimated_radius_m": radius,
            }
        )
    return candidates


def _make_side_candidates(det: dict[str, Any]) -> list[dict[str, Any]]:
    radius, half_height = _extent_values(det)
    candidates = generate_side_grasp_candidates(
        det,
        object_kind="can",
        arm=ARM,
        default_radius_m=radius,
        default_half_height_m=half_height,
        body_fraction=0.62,
        pregrasp_standoff_m=0.095,
        lift_z_m=0.12,
        width_margin_m=0.025,
        include_topdown=False,
        yaw_angles_deg=[90.0, 70.0, 110.0, 45.0, 135.0],
        z_offsets_m=[0.0, 0.015, -0.015],
        center_z_offset_without_extents_m=0.02,
    )
    for cand in candidates:
        cand["strategy"] = "left_side_cylinder"
    return candidates


def _plan_candidates(
    *,
    run_dir: Any,
    candidates: list[dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
    return rank_motion_candidates(
        candidates=candidates,
        freespace_move=freespace_move,
        run_in_background=run_in_background,
        run_dir=run_dir,
        stage=stage,
        task_name=TASK_NAME,
        timeout_s=45.0,
        planner_backend="curobo",
        solver_speed="fast",
        planning_speed=0.18,
        ik_error_threshold=0.025,
        ik_rot_threshold_deg=18.0,
        stop_after_successes=1,
    )


def _move_to_pose_with_preview(
    *,
    pose: dict[str, Any],
    run_dir: Any,
    stage: str,
) -> dict[str, Any]:
    preview = freespace_move(
        left_target_pos=pose["position"],
        left_target_rpy=pose["rpy"],
        left_gripper=1.0,
        preview_only=True,
        planner_backend="curobo",
        solver_speed="fast",
        planning_speed=0.16,
        ik_error_threshold=0.025,
        ik_rot_threshold_deg=18.0,
    )
    cache_key = preview.get("trajectory_cache_key") if isinstance(preview, dict) else getattr(preview, "trajectory_cache_key", None)
    if not cache_key:
        raise RuntimeError(f"{stage} preview did not return a trajectory_cache_key: {preview!r}")
    executed = freespace_move(trajectory_cache_key=cache_key)
    packet = {"stage": stage, "preview": preview, "execute": executed, "trajectory_cache_key": cache_key}
    write_json(run_dir / "plans" / f"{stage}.json", packet)
    return packet


def _servo_delta_chunks(
    *,
    delta: list[float],
    run_dir: Any,
    stage: str,
    chunk_m: float = 0.026,
    duration_per_chunk_s: float = 0.75,
    gripper_pos: float | None = None,
    steps: int = 45,
    max_translation_m: float = 0.032,
    max_component_m: float = 0.032,
    max_joint_delta_rad: float = 0.35,
    max_ik_pos_error_m: float = 0.020,
) -> dict[str, Any]:
    import math

    norm = math.sqrt(sum(float(v) * float(v) for v in delta))
    chunks = max(1, int(math.ceil(norm / max(0.001, chunk_m))))
    step = [float(v) / chunks for v in delta]
    records: list[dict[str, Any]] = []
    ok = True
    error = None
    for idx in range(chunks):
        try:
            kwargs: dict[str, Any] = {
                "side": ARM,
                "delta_pos": step,
                "duration_s": duration_per_chunk_s,
                "steps": steps,
                "max_translation_m": max_translation_m,
                "max_component_m": max_component_m,
                "max_joint_delta_rad": max_joint_delta_rad,
                "max_ik_pos_error_m": max_ik_pos_error_m,
                "command_hz": 60.0,
            }
            if gripper_pos is not None:
                kwargs["gripper_pos"] = gripper_pos
            result = servo_ee_delta(**kwargs)
            records.append({"index": idx, "ok": True, "delta_pos": step, "result": result})
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            records.append({"index": idx, "ok": False, "delta_pos": step, "error": error})
            break
    packet = {
        "stage": stage,
        "ok": ok,
        "error": error,
        "total_delta": delta,
        "chunk_delta": step,
        "chunks": chunks,
        "records": records,
    }
    write_json(run_dir / "plans" / f"{stage}.json", packet)
    return packet


def _execute_servo_grasp_lift_attempt(*, plan: dict[str, Any], run_dir: Any) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.pick_can_left.servo_execute.v1",
        "success": False,
        "physical_motion_executed": False,
        "arm": ARM,
        "steps": [],
        "why_stopped": None,
    }
    try:
        open_result = open_gripper(ARM)
        packet["physical_motion_executed"] = True
        packet["steps"].append({"stage": "open_gripper", "ok": True, "result": open_result})

        pre = _move_to_pose_with_preview(
            pose=plan["pregrasp_pose"],
            run_dir=run_dir,
            stage="execute_pregrasp_pose",
        )
        packet["steps"].append({"stage": "pregrasp_pose", "ok": True, "result": pre})

        pre_pos = [float(v) for v in plan["pregrasp_pose"]["position"]]
        grasp_pos = [float(v) for v in plan["grasp_pose"]["position"]]
        approach_delta = [grasp_pos[i] - pre_pos[i] for i in range(3)]
        approach = _servo_delta_chunks(
            delta=approach_delta,
            run_dir=run_dir,
            stage="servo_approach_to_grasp",
            chunk_m=0.026,
            duration_per_chunk_s=0.8,
            gripper_pos=1.0,
        )
        packet["steps"].append({"stage": "servo_approach_to_grasp", "ok": approach["ok"], "result": approach})
        if not approach["ok"]:
            packet["why_stopped"] = f"servo approach failed: {approach.get('error')}"
            packet["path"] = write_json(run_dir / "plans" / "execute_servo_grasp_lift_attempt.json", packet)
            return packet

        close = staged_close_with_contact(
            side=ARM,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            target=0.0,
            steps=[0.80, 0.58, 0.38, 0.20, 0.0],
            min_contact_delta=0.025,
            target_tolerance=0.030,
            hold_min=0.015,
            hold_max=0.95,
            confirm_timeout_s=0.2,
            run_dir=run_dir,
            task_name=TASK_NAME,
            stage="servo_grasp_close",
        )
        packet["gripper_after_close"] = close
        packet["steps"].append({"stage": "grasp_close", "ok": bool(close.get("plausible_for_lift")), "result": close})
        if not close.get("plausible_for_lift"):
            packet["why_stopped"] = "gripper close did not reach a plausible hold state; stop before lift"
            packet["path"] = write_json(run_dir / "plans" / "execute_servo_grasp_lift_attempt.json", packet)
            return packet

        # Single move straight to the full lift target (no re-seeded chunks):
        # one IK to current+0.12 z, one smooth ramp, then the convergence settle.
        # Recovers the full commanded rise (chunked re-seeding lost ~20 mm) and
        # replaces the 5-step staircase with one motion. chunk_m > |delta| forces
        # a single chunk; servo limits raised to allow the 120 mm move.
        lift = _servo_delta_chunks(
            delta=[0.0, 0.0, 0.12],
            run_dir=run_dir,
            stage="servo_lift",
            chunk_m=0.13,
            duration_per_chunk_s=2.5,
            gripper_pos=None,
            steps=120,
            max_translation_m=0.14,
            max_component_m=0.14,
            max_joint_delta_rad=0.9,
            max_ik_pos_error_m=0.030,
        )
        packet["steps"].append({"stage": "servo_lift", "ok": lift["ok"], "result": lift})
        if not lift["ok"]:
            packet["why_stopped"] = f"servo lift failed: {lift.get('error')}"
        else:
            packet["success"] = True
            packet["why_stopped"] = "servo lift command completed; verify with post-action observation"
    except Exception as exc:
        packet["why_stopped"] = f"{type(exc).__name__}: {exc}"
        packet["steps"].append({"stage": "exception", "ok": False, "error": packet["why_stopped"]})
    packet["path"] = write_json(run_dir / "plans" / "execute_servo_grasp_lift_attempt.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## execute servo grasp_lift",
            f"- success: {packet['success']}",
            f"- physical_motion_executed: {packet['physical_motion_executed']}",
            f"- why_stopped: {packet['why_stopped']}",
        ],
    )
    return packet


def _main() -> None:
    run_dir = current_run_dir(TASK_NAME)
    append_stage_summary(
        run_dir,
        [
            "# pick_can_left",
            f"- started_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
            "- arm: left only",
            "- strategy_order: sim_can_topdown, left_side_cylinder_fallback",
        ],
    )
    if os.environ.get("OPENFORGE_ALLOW_PHYSICAL_MOTION", "").strip() != "1":
        TASK_RESULT["why_stopped"] = "OPENFORGE_ALLOW_PHYSICAL_MOTION is not set"
        write_json(run_dir / "task_result.json", TASK_RESULT)
        raise RuntimeError(TASK_RESULT["why_stopped"])

    pre = capture_scene(
        prompts=PROMPTS,
        cameras=["top", "left", "right", "bottom"],
        image_only_cameras=("left", "right", "bottom"),
        motion_cameras=("top",),
        detect_objects_oneshot=detect_objects_oneshot,
        get_camera_image=get_camera_image,
        get_robot_state=get_robot_state,
        run_in_background=run_in_background,
        run_dir=run_dir,
        stage="pre_grasp",
        task_name=TASK_NAME,
        timeout_s=45.0,
        max_retries=1,
    )
    selected = _select_can_detection(pre)
    TASK_RESULT["selected_detection"] = selected
    if selected is None:
        TASK_RESULT["why_stopped"] = "no workspace-valid can detection from top camera"
        write_json(run_dir / "task_result.json", TASK_RESULT)
        return

    write_json(run_dir / "plans" / "selected_detection.json", selected)
    topdown = _plan_candidates(
        run_dir=run_dir,
        candidates=_make_topdown_candidates(selected),
        stage="topdown_plan",
    )
    plan = topdown.get("selected")
    if plan is None:
        side = _plan_candidates(
            run_dir=run_dir,
            candidates=_make_side_candidates(selected),
            stage="side_plan",
        )
        plan = side.get("selected")
    if plan is None:
        TASK_RESULT["why_stopped"] = "no feasible left-arm can grasp plan"
        write_json(run_dir / "task_result.json", TASK_RESULT)
        return

    plan["arm"] = ARM
    TASK_RESULT["selected_plan"] = {
        "label": plan.get("label"),
        "strategy": plan.get("strategy"),
        "arm": plan.get("arm"),
        "grasp_pose": plan.get("grasp_pose"),
        "lift_pose": plan.get("lift_pose"),
    }
    write_json(run_dir / "plans" / "selected_plan.json", TASK_RESULT["selected_plan"])

    execution = _execute_servo_grasp_lift_attempt(plan=plan, run_dir=run_dir)
    TASK_RESULT["execution"] = execution

    post = capture_scene(
        prompts=PROMPTS,
        cameras=["top", "left", "right", "bottom"],
        image_only_cameras=("left", "right", "bottom"),
        motion_cameras=("top",),
        detect_objects_oneshot=detect_objects_oneshot,
        get_camera_image=get_camera_image,
        get_robot_state=get_robot_state,
        run_in_background=run_in_background,
        run_dir=run_dir,
        stage="post_lift",
        task_name=TASK_NAME,
        timeout_s=45.0,
        max_retries=1,
    )
    post_detection = _select_can_detection(post)
    verification = verify_lift(
        selected,
        post_detection,
        min_z_delta_m=0.045,
        max_xy_delta_m=0.22,
    )
    verification["post_detection"] = post_detection
    TASK_RESULT["verification"] = verification
    success = bool(execution.get("success")) and bool(verification.get("success"))
    TASK_RESULT["success"] = success
    TASK_RESULT["reward"] = 1.0 if success else 0.0
    TASK_RESULT["why_stopped"] = (
        "verified can lift"
        if success
        else f"execution={execution.get('why_stopped')}; verification={verification.get('reason')}"
    )
    append_stage_summary(
        run_dir,
        [
            "## final",
            f"- success: {TASK_RESULT['success']}",
            f"- why_stopped: {TASK_RESULT['why_stopped']}",
            f"- verification: {verification}",
        ],
    )
    write_json(run_dir / "task_result.json", TASK_RESULT)


_main()
