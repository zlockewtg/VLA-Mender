"""Pick the orange bowl and place it on the blue plate.

This script is intended to run under run_script.py with Real-YAM injected tools.
It plans from fresh detections only and records enough artifacts for the next
debug pass.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    capture_scene,
    current_run_dir,
    json_safe,
    write_json,
)


TASK_NAME = "orange_bowl_on_blue_plate"
PROMPTS = ["orange bowl", "blue plate"]
CAMERAS = ["top", "left", "right", "bottom"]

TABLE_Z_M = 0.760
BOWL_HEIGHT_M = 0.075
BOWL_RADIUS_M = 0.095
BOWL_TOP_DOWN_TCP_ABOVE_RIM_M = 0.010
BOWL_RIM_INSET_M = 0.010
PREGRASP_ABOVE_RIM_M = 0.090
LIFT_ABOVE_GRASP_M = 0.085
PLACE_PRE_ABOVE_M = 0.090
RETREAT_ABOVE_PLACE_M = 0.115
PLACE_PRE_ABOVE_CANDIDATES_M = (0.060, 0.075, 0.090)
RETREAT_ABOVE_PLACE_CANDIDATES_M = (0.085, 0.100)
PLATE_TOP_DEFAULT_M = TABLE_Z_M + 0.010

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = 0.04
PLANNING_SPEED = 0.20
IK_ERROR_M = 0.025
IK_ROT_DEG = 15.0
PLACE_CENTER_TOL_M = 0.090

TASK_RESULT: dict[str, Any] = {
    "success": False,
    "reward": 0.0,
    "method": TASK_NAME,
    "why_stopped": "not_started",
    "physical_motion_executed": False,
    "selected_plan": None,
    "motion_steps": [],
    "observations": {},
    "checks": {},
}


def get_task_info() -> dict[str, Any]:
    return TASK_RESULT


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _xyz(det: dict[str, Any]) -> list[float]:
    value = det.get("position_3d") or det.get("position")
    if not value or len(value) < 3:
        raise ValueError(f"detection has no usable position: {det!r}")
    return [float(value[0]), float(value[1]), float(value[2])]


def _norm_angle_deg(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _topdown_yaw_for_radial(theta_deg: float) -> float:
    # For display RPY [0, 180, yaw], local X/opening axis equals the radial
    # rim-straddle direction when yaw = -theta - 90.
    return _norm_angle_deg(-float(theta_deg) - 90.0)


def _unit_from_angle(theta_deg: float) -> list[float]:
    rad = math.radians(theta_deg)
    return [math.cos(rad), math.sin(rad), 0.0]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _best_detection(obs: dict[str, Any], prompt: str) -> dict[str, Any] | None:
    target = prompt.strip().lower()
    detections = [
        det
        for det in obs.get("all_detections", [])
        if str(det.get("prompt", "")).strip().lower() == target
    ]
    if not detections:
        return None

    def key(det: dict[str, Any]) -> tuple[int, float]:
        camera = str(det.get("source_camera") or det.get("camera") or "")
        score = float(det.get("score") or 0.0)
        return (1 if camera == "top" else 0, score)

    return sorted(detections, key=key, reverse=True)[0]


def _merge_obs(primary: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    merged["all_detections"] = list(primary.get("all_detections", [])) + list(
        fallback.get("all_detections", [])
    )
    merged["fallback_observation"] = fallback.get("packet_path")
    return merged


def _radius_from_detection(det: dict[str, Any]) -> float:
    half_extents = det.get("half_extents") or []
    if len(half_extents) >= 2:
        xy_vals = sorted(abs(float(v)) for v in half_extents[:2])
        if 0.060 <= xy_vals[0] <= 0.130:
            return float(xy_vals[0])
    return BOWL_RADIUS_M


def _rim_z_from_detection(det: dict[str, Any]) -> float:
    xyz = _xyz(det)
    half_extents = det.get("half_extents") or []
    rim_z = TABLE_Z_M + BOWL_HEIGHT_M
    z = float(xyz[2])
    if TABLE_Z_M + 0.040 <= z <= TABLE_Z_M + 0.130:
        rim_z = max(rim_z, z)
    elif len(half_extents) >= 3:
        he_z = abs(float(half_extents[2]))
        if 0.020 <= he_z <= 0.070:
            rim_z = max(rim_z, z + he_z)
    return _clamp(rim_z, TABLE_Z_M + 0.065, TABLE_Z_M + 0.105)


def _plate_top_from_detection(det: dict[str, Any]) -> float:
    xyz = _xyz(det)
    half_extents = det.get("half_extents") or []
    he_z = 0.005
    if len(half_extents) >= 3 and 0.001 <= abs(float(half_extents[2])) <= 0.030:
        he_z = abs(float(half_extents[2]))
    candidate = float(xyz[2]) + he_z
    if TABLE_Z_M - 0.010 <= candidate <= TABLE_Z_M + 0.040:
        return _clamp(candidate, TABLE_Z_M + 0.003, TABLE_Z_M + 0.035)
    return PLATE_TOP_DEFAULT_M


def _arm_for_xy(xy: list[float]) -> str:
    return "left" if float(xy[1]) > 0.04 else "right"


def _pose(position: list[float], rpy: list[float]) -> dict[str, Any]:
    return {
        "position": [round(float(v), 5) for v in position],
        "rpy": [round(float(v), 4) for v in rpy],
    }


def _make_pick_candidates(bowl_det: dict[str, Any], plate_det: dict[str, Any]) -> list[dict[str, Any]]:
    bowl_xyz = _xyz(bowl_det)
    plate_xyz = _xyz(plate_det)
    arm = _arm_for_xy(bowl_xyz)
    radius = _radius_from_detection(bowl_det)
    rim_offset = _clamp(radius - BOWL_RIM_INSET_M, 0.065, 0.090)
    rim_tcp_z = _rim_z_from_detection(bowl_det) + BOWL_TOP_DOWN_TCP_ABOVE_RIM_M
    plate_top_z = _plate_top_from_detection(plate_det)
    place_tcp_z = plate_top_z + BOWL_HEIGHT_M + BOWL_TOP_DOWN_TCP_ABOVE_RIM_M

    preferred_thetas = [0.0, -90.0, 90.0, 180.0, -45.0, 45.0, -135.0, 135.0]
    if arm == "left":
        preferred_thetas = [0.0, 90.0, -90.0, 180.0, 45.0, -45.0, 135.0, -135.0]

    candidates: list[dict[str, Any]] = []
    for idx, theta in enumerate(preferred_thetas):
        radial = _unit_from_angle(theta)
        yaw = _topdown_yaw_for_radial(theta)
        rpy = [0.0, 180.0, yaw]
        grasp_xy = [
            bowl_xyz[0] + radial[0] * rim_offset,
            bowl_xyz[1] + radial[1] * rim_offset,
        ]
        place_xy = [
            plate_xyz[0] + radial[0] * rim_offset,
            plate_xyz[1] + radial[1] * rim_offset,
        ]
        grasp = [grasp_xy[0], grasp_xy[1], rim_tcp_z]
        candidate = {
            "label": f"top_down_rim_straddle_{idx}",
            "arm": arm,
            "theta_deg": theta,
            "radial": [round(radial[0], 5), round(radial[1], 5), 0.0],
            "yaw_deg": yaw,
            "rpy": rpy,
            "rim_offset_m": rim_offset,
            "bowl_center": [round(v, 5) for v in bowl_xyz],
            "plate_center": [round(v, 5) for v in plate_xyz],
            "radius_m": radius,
            "rim_tcp_z_m": rim_tcp_z,
            "plate_top_z_m": plate_top_z,
            "place_tcp_z_m": place_tcp_z,
            "pregrasp_pose": _pose(
                [grasp_xy[0], grasp_xy[1], rim_tcp_z + PREGRASP_ABOVE_RIM_M],
                rpy,
            ),
            "grasp_pose": _pose(grasp, rpy),
            "lift_pose": _pose(
                [grasp_xy[0], grasp_xy[1], rim_tcp_z + LIFT_ABOVE_GRASP_M],
                rpy,
            ),
            "nominal_place_xy": [round(float(place_xy[0]), 5), round(float(place_xy[1]), 5)],
        }
        candidates.append(candidate)
    return candidates


def _make_place_candidates(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Create plate placement targets after a verified pickup.

    Keep the same rim radial relation used for the grasp, but bias the bowl
    center toward the active arm side so the left arm is not asked to hold an
    exact top-down pose at the plate center. The offsets are small enough to
    keep visible bowl overlap with the plate while improving IK reachability.
    """
    arm = str(plan["arm"])
    arm_y_sign = 1.0 if arm == "left" else -1.0
    plate_center = [float(v) for v in plan["plate_center"]]
    radial = [float(v) for v in plan["radial"][:3]]
    rim_offset = float(plan["rim_offset_m"])
    place_tcp_z = float(plan["place_tcp_z_m"])
    base_rpy = [float(v) for v in plan["rpy"]]
    center_offsets = [
        [0.000, 0.000, "center"],
        [0.000, arm_y_sign * 0.035, "arm_side_35mm"],
        [-0.020, arm_y_sign * 0.040, "back_arm_side_40mm"],
        [0.020, arm_y_sign * 0.040, "front_arm_side_40mm"],
        [0.000, arm_y_sign * 0.055, "arm_side_55mm"],
    ]
    pitch_offsets = [0.0, -5.0, -10.0]
    candidates: list[dict[str, Any]] = []
    idx = 0
    for center_dx, center_dy, center_label in center_offsets:
        bowl_center_xy = [plate_center[0] + center_dx, plate_center[1] + center_dy]
        place_xy = [
            bowl_center_xy[0] + radial[0] * rim_offset,
            bowl_center_xy[1] + radial[1] * rim_offset,
        ]
        for pre_above in PLACE_PRE_ABOVE_CANDIDATES_M:
            for retreat_above in RETREAT_ABOVE_PLACE_CANDIDATES_M:
                for pitch_offset in pitch_offsets:
                    rpy = [base_rpy[0], base_rpy[1] + pitch_offset, base_rpy[2]]
                    candidates.append(
                        {
                            "label": f"place_{idx}_{center_label}",
                            "arm": arm,
                            "rpy": [round(float(v), 4) for v in rpy],
                            "bowl_center_target": [
                                round(float(bowl_center_xy[0]), 5),
                                round(float(bowl_center_xy[1]), 5),
                            ],
                            "place_xy": [
                                round(float(place_xy[0]), 5),
                                round(float(place_xy[1]), 5),
                            ],
                            "pre_above_m": float(pre_above),
                            "retreat_above_m": float(retreat_above),
                            "pitch_offset_deg": float(pitch_offset),
                            "place_pre_pose": _pose(
                                [place_xy[0], place_xy[1], place_tcp_z + float(pre_above)],
                                rpy,
                            ),
                            "place_pose": _pose([place_xy[0], place_xy[1], place_tcp_z], rpy),
                            "retreat_pose": _pose(
                                [place_xy[0], place_xy[1], place_tcp_z + float(retreat_above)],
                                rpy,
                            ),
                        }
                    )
                    idx += 1
    return candidates


def _move_kwargs(arm: str, pose: dict[str, Any], *, preview_only: bool, gripper: float | None) -> dict[str, Any]:
    prefix = "left" if arm == "left" else "right"
    kwargs = {
        f"{prefix}_target_pos": pose["position"],
        f"{prefix}_target_rpy": pose["rpy"],
        "preview_only": bool(preview_only),
        "planner_backend": "curobo",
        "solver_speed": "fast",
        "planning_speed": PLANNING_SPEED,
        "ik_error_threshold": IK_ERROR_M,
        "ik_rot_threshold_deg": IK_ROT_DEG,
    }
    if gripper is not None:
        kwargs[f"{prefix}_gripper"] = float(gripper)
    return kwargs


def _cache_key(preview: Any) -> str | None:
    key = _field(preview, "trajectory_cache_key", None)
    if key is None and isinstance(preview, dict):
        key = preview.get("trajectory_cache_key")
    return None if key is None else str(key)


def _preview_pose(arm: str, pose: dict[str, Any], *, gripper: float | None) -> Any:
    return freespace_move(**_move_kwargs(arm, pose, preview_only=True, gripper=gripper))


def _execute_pose(
    arm: str,
    pose: dict[str, Any],
    *,
    label: str,
    gripper: float | None,
    run_dir: Path,
) -> Any:
    preview = _preview_pose(arm, pose, gripper=gripper)
    key = _cache_key(preview)
    if not key:
        raise RuntimeError(f"{label} preview did not return a trajectory cache key: {preview!r}")
    executed = freespace_move(trajectory_cache_key=key)
    TASK_RESULT["physical_motion_executed"] = True
    TASK_RESULT["motion_steps"].append(
        {
            "label": label,
            "pose": pose,
            "preview": json_safe(preview),
            "execute": json_safe(executed),
            "trajectory_cache_key": key,
        }
    )
    write_json(run_dir / "plans" / "motion_steps_latest.json", TASK_RESULT["motion_steps"])
    return executed


def _pick_candidate_previews(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    arm = candidate["arm"]
    checks = []
    for label, gripper in (
        ("pregrasp_pose", GRIPPER_OPEN),
        ("grasp_pose", GRIPPER_OPEN),
        ("lift_pose", GRIPPER_CLOSE),
    ):
        preview = _preview_pose(arm, candidate[label], gripper=gripper)
        checks.append({"label": label, "preview": json_safe(preview)})
    return checks


def _select_pick_candidate(candidates: list[dict[str, Any]], run_dir: Path) -> dict[str, Any] | None:
    attempts = []
    for candidate in candidates:
        try:
            checks = _pick_candidate_previews(candidate)
            selected = dict(candidate)
            selected["preview_checks"] = checks
            attempts.append({"label": candidate["label"], "ok": True, "checks": checks})
            packet = {"selected": selected, "attempts": attempts}
            write_json(run_dir / "plans" / "pick_candidate_selection.json", packet)
            return selected
        except Exception as exc:
            attempts.append(
                {
                    "label": candidate["label"],
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "candidate": candidate,
                }
            )
            write_json(run_dir / "plans" / "pick_candidate_selection.json", {"attempts": attempts})
    return None


def _place_candidate_previews(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    arm = candidate["arm"]
    checks = []
    for label, gripper in (
        ("place_pre_pose", GRIPPER_CLOSE),
        ("place_pose", GRIPPER_CLOSE),
        ("retreat_pose", GRIPPER_OPEN),
    ):
        preview = _preview_pose(arm, candidate[label], gripper=gripper)
        checks.append({"label": label, "preview": json_safe(preview)})
    return checks


def _select_place_candidate(
    candidates: list[dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any] | None:
    attempts = []
    for candidate in candidates:
        try:
            checks = _place_candidate_previews(candidate)
            selected = dict(candidate)
            selected["preview_checks"] = checks
            attempts.append({"label": candidate["label"], "ok": True, "checks": checks})
            write_json(
                run_dir / "plans" / "place_candidate_selection.json",
                {"selected": selected, "attempts": attempts},
            )
            return selected
        except Exception as exc:
            attempts.append(
                {
                    "label": candidate["label"],
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "candidate": candidate,
                }
            )
            write_json(run_dir / "plans" / "place_candidate_selection.json", {"attempts": attempts})
    return None


def _capture_initial(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    primary = capture_scene(
        prompts=PROMPTS,
        cameras=CAMERAS,
        detect_objects_oneshot=detect_objects_oneshot,
        get_camera_image=get_camera_image,
        get_robot_state=get_robot_state,
        run_in_background=run_in_background,
        run_dir=run_dir,
        stage="observe_initial_top",
        task_name=TASK_NAME,
        timeout_s=45.0,
        max_retries=1,
        motion_cameras=("top",),
        image_only_cameras=("left", "right", "bottom"),
    )
    obs = primary
    bowl = _best_detection(obs, "orange bowl")
    plate = _best_detection(obs, "blue plate")
    if bowl is None or plate is None:
        fallback = capture_scene(
            prompts=PROMPTS,
            cameras=["left", "right", "bottom"],
            detect_objects_oneshot=detect_objects_oneshot,
            get_camera_image=get_camera_image,
            get_robot_state=get_robot_state,
            run_in_background=run_in_background,
            run_dir=run_dir,
            stage="observe_initial_side_fallback",
            task_name=TASK_NAME,
            timeout_s=45.0,
            max_retries=1,
            motion_cameras=("left", "right"),
            image_only_cameras=("bottom",),
        )
        obs = _merge_obs(primary, fallback)
        bowl = bowl or _best_detection(fallback, "orange bowl")
        plate = plate or _best_detection(fallback, "blue plate")
    if bowl is None:
        raise RuntimeError("orange bowl was not detected in top or side cameras")
    if plate is None:
        raise RuntimeError("blue plate was not detected in top or side cameras")
    TASK_RESULT["observations"]["initial"] = {
        "primary_packet": primary.get("packet_path"),
        "selected_bowl": bowl,
        "selected_plate": plate,
    }
    return obs, bowl, plate


def _xy_distance(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _final_check(run_dir: Path, plate_center: list[float]) -> dict[str, Any]:
    final_obs = capture_scene(
        prompts=PROMPTS,
        cameras=CAMERAS,
        detect_objects_oneshot=detect_objects_oneshot,
        get_camera_image=get_camera_image,
        get_robot_state=get_robot_state,
        run_in_background=run_in_background,
        run_dir=run_dir,
        stage="observe_final",
        task_name=TASK_NAME,
        timeout_s=45.0,
        max_retries=1,
        motion_cameras=("top",),
        image_only_cameras=("left", "right", "bottom"),
    )
    final_bowl = _best_detection(final_obs, "orange bowl")
    check: dict[str, Any] = {
        "packet": final_obs.get("packet_path"),
        "final_bowl": final_bowl,
        "plate_center": plate_center,
        "success_estimate": False,
        "reason": "missing final orange bowl detection",
    }
    if final_bowl is not None:
        dist = _xy_distance(_xyz(final_bowl), plate_center)
        check.update(
            {
                "xy_distance_m": round(dist, 5),
                "success_estimate": dist <= PLACE_CENTER_TOL_M,
                "reason": "final bowl center is near plate center"
                if dist <= PLACE_CENTER_TOL_M
                else "final bowl center is not near plate center",
            }
        )
    TASK_RESULT["observations"]["final"] = check
    return check


def _finish(run_dir: Path, *, success: bool, why: str) -> None:
    TASK_RESULT["success"] = bool(success)
    TASK_RESULT["reward"] = 1.0 if success else 0.0
    TASK_RESULT["why_stopped"] = why
    TASK_RESULT["task_result_path"] = write_json(run_dir / "task_result.json", TASK_RESULT)
    append_stage_summary(
        run_dir,
        [
            "## task result",
            f"- success: {TASK_RESULT['success']}",
            f"- reward: {TASK_RESULT['reward']}",
            f"- why_stopped: {why}",
            f"- physical_motion_executed: {TASK_RESULT['physical_motion_executed']}",
        ],
    )


def main() -> None:
    run_dir = current_run_dir(TASK_NAME)
    required = [
        "detect_objects_oneshot",
        "get_camera_image",
        "get_robot_state",
        "run_in_background",
        "freespace_move",
        "open_gripper",
        "set_gripper",
    ]
    missing = [name for name in required if name not in globals() or not callable(globals()[name])]
    if missing:
        raise RuntimeError(f"missing run_script injected tools: {missing}")
    if os.environ.get("OPENFORGE_ALLOW_PHYSICAL_MOTION", "").strip() != "1":
        raise RuntimeError("OPENFORGE_ALLOW_PHYSICAL_MOTION=1 is required for this physical task")

    _, bowl, plate = _capture_initial(run_dir)
    candidates = _make_pick_candidates(bowl, plate)
    plan = _select_pick_candidate(candidates, run_dir)
    if plan is None:
        _finish(run_dir, success=False, why="no pickup candidate previewed successfully; no motion executed")
        return

    TASK_RESULT["selected_plan"] = plan
    write_json(run_dir / "plans" / "selected_pick_plan.json", plan)
    append_stage_summary(
        run_dir,
        [
            "## selected plan",
            f"- label: {plan['label']}",
            f"- arm: {plan['arm']}",
            f"- theta_deg: {plan['theta_deg']}",
            f"- bowl_center: {plan['bowl_center']}",
            f"- plate_center: {plan['plate_center']}",
        ],
    )

    arm = plan["arm"]
    open_gripper(arm)
    TASK_RESULT["physical_motion_executed"] = True
    TASK_RESULT["motion_steps"].append({"label": "open_gripper_before_pick", "arm": arm})

    _execute_pose(arm, plan["pregrasp_pose"], label="pregrasp", gripper=GRIPPER_OPEN, run_dir=run_dir)
    _execute_pose(arm, plan["grasp_pose"], label="descend_to_rim_straddle", gripper=GRIPPER_OPEN, run_dir=run_dir)

    set_gripper(arm, GRIPPER_CLOSE)
    time.sleep(0.7)
    close_state = get_robot_state()
    TASK_RESULT["motion_steps"].append(
        {
            "label": "close_gripper_on_rim",
            "arm": arm,
            "target": GRIPPER_CLOSE,
            "robot_state_after": json_safe(close_state),
        }
    )
    write_json(run_dir / "plans" / "motion_steps_latest.json", TASK_RESULT["motion_steps"])

    _execute_pose(arm, plan["lift_pose"], label="lift_bowl", gripper=GRIPPER_CLOSE, run_dir=run_dir)

    post_lift = capture_scene(
        prompts=["orange bowl"],
        cameras=CAMERAS,
        detect_objects_oneshot=detect_objects_oneshot,
        get_camera_image=get_camera_image,
        get_robot_state=get_robot_state,
        run_in_background=run_in_background,
        run_dir=run_dir,
        stage="observe_post_lift",
        task_name=TASK_NAME,
        timeout_s=45.0,
        max_retries=1,
        motion_cameras=("top",),
        image_only_cameras=("left", "right", "bottom"),
    )
    TASK_RESULT["observations"]["post_lift"] = {"packet": post_lift.get("packet_path")}

    place_candidates = _make_place_candidates(plan)
    place_plan = _select_place_candidate(place_candidates, run_dir)
    if place_plan is None:
        TASK_RESULT["selected_place_plan"] = None
        TASK_RESULT["why_stopped"] = "no plate placement candidate previewed after lift; returning bowl to start"
        write_json(run_dir / "plans" / "selected_place_plan.json", TASK_RESULT["selected_place_plan"])
        _execute_pose(arm, plan["grasp_pose"], label="return_to_original_rim_pose", gripper=GRIPPER_CLOSE, run_dir=run_dir)
        open_gripper(arm)
        time.sleep(0.7)
        TASK_RESULT["motion_steps"].append({"label": "open_gripper_return_release", "arm": arm})
        _execute_pose(arm, plan["pregrasp_pose"], label="retreat_after_return_release", gripper=GRIPPER_OPEN, run_dir=run_dir)
        _finish(run_dir, success=False, why="no plate placement candidate previewed after lift; bowl returned")
        return
    TASK_RESULT["selected_place_plan"] = place_plan
    write_json(run_dir / "plans" / "selected_place_plan.json", place_plan)
    append_stage_summary(
        run_dir,
        [
            "## selected place plan",
            f"- label: {place_plan['label']}",
            f"- bowl_center_target: {place_plan['bowl_center_target']}",
            f"- place_xy: {place_plan['place_xy']}",
            f"- rpy: {place_plan['rpy']}",
        ],
    )

    _execute_pose(arm, place_plan["place_pre_pose"], label="transport_above_plate", gripper=GRIPPER_CLOSE, run_dir=run_dir)
    _execute_pose(arm, place_plan["place_pose"], label="lower_to_plate", gripper=GRIPPER_CLOSE, run_dir=run_dir)

    open_gripper(arm)
    time.sleep(0.7)
    TASK_RESULT["motion_steps"].append({"label": "open_gripper_release", "arm": arm})
    _execute_pose(arm, place_plan["retreat_pose"], label="vertical_retreat_after_release", gripper=GRIPPER_OPEN, run_dir=run_dir)

    final_check = _final_check(run_dir, list(plan["plate_center"]))
    success = bool(final_check.get("success_estimate"))
    why = str(final_check.get("reason") or "completed placement sequence")
    _finish(run_dir, success=success, why=why)


try:
    main()
except Exception as exc:
    run_dir = current_run_dir(TASK_NAME)
    message = f"{type(exc).__name__}: {exc}"
    TASK_RESULT["error"] = message
    print(f"[{TASK_NAME}] ERROR: {message}")
    _finish(run_dir, success=False, why=message)
