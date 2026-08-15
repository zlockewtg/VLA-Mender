"""Agent handoff, failure windows, prefix replay and reset-bank materialization."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..libero_runtime import LiberoRuntime, PUBLIC_REPLAY_TOLERANCE
from ..parameters import ExperimentSettings
from ..trajectory_protocol import (
    diagnosis_evidence_metadata,
    validate_episode,
    validate_rollout_contract,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _episode_path(rollout_dir: Path, episode_index: int) -> Path:
    return rollout_dir / "episodes" / f"episode_{episode_index:06d}.json"


def _render_prompt(
    settings: ExperimentSettings, *, run_root: Path, settings_path: Path
) -> str:
    """Render the checked-in task-level prompt with the resolved contract."""

    template_candidates = (
        Path(__file__).resolve().parents[2] / "run" / "pre_repair" / "prompt.md",
        Path(__file__).with_name("prompt.md"),
    )
    template_path = next((path for path in template_candidates if path.is_file()), None)
    if template_path is None:
        raise FileNotFoundError("pre-repair prompt template not found")
    template = template_path.read_text(encoding="utf-8")
    values = {
        "{{SUITE}}": settings.task.suite,
        "{{TASK_ID}}": str(settings.task.task_id),
        "{{TASK_DESCRIPTION}}": settings.task.task_description
        or "(use the task instruction from the rollout)",
        "{{CHECKPOINT}}": str(settings.task.checkpoint),
        "{{RUNTIME_BACKEND}}": settings.backend.name,
        "{{OPENPI_COMMIT}}": settings.backend.openpi_commit or "(vendored pin)",
        "{{OPENPI_ENVIRONMENT}}": str(
            settings.backend.openpi_environment or Path(sys.prefix).resolve()
        ),
        "{{LIBERO_ROOT}}": str(settings.backend.libero_root or "(LIBERO default)"),
        "{{TRAJECTORY_PROTOCOL}}": "vla-mender.libero.openpi/v2",
        "{{STATE_PROVIDER}}": settings.initial_states.provider,
        "{{STATE_COUNT}}": str(settings.initial_states.count),
        "{{STATE_MANIFEST}}": str(settings.initial_states.state_manifest or "(none)"),
        "{{CONTROL_FREQUENCY_HZ}}": str(settings.rollout.control_frequency_hz),
        "{{MAX_STEPS}}": str(settings.rollout.max_steps),
        "{{POLICY_SEED}}": str(settings.rollout.policy_seed),
        "{{GPUS}}": ", ".join(str(gpu) for gpu in settings.rollout.gpus),
        "{{WORKERS_PER_GPU}}": str(settings.rollout.workers_per_gpu),
        "{{ACTION_CHUNK}}": str(settings.rollout.action_chunk),
        "{{INFERENCE_STEPS}}": str(settings.rollout.inference_steps),
        "{{NUM_STEPS_WAIT}}": str(settings.rollout.num_steps_wait),
        "{{BINARY_GRIPPER}}": str(settings.rollout.binary_gripper).lower(),
        "{{GRIPPER_HYSTERESIS_THRESHOLD}}": str(
            settings.rollout.gripper_hysteresis_threshold
        ),
        "{{SOURCE_CONTROL_SPACE}}": settings.controller.source_control_space,
        "{{TARGET_CONTROL_SPACE}}": settings.controller.target_control_space,
        "{{RESET_DYNAMICS}}": settings.reset.dynamics,
        "{{FRAMES_PER_FAILURE}}": str(settings.reset.frames_per_failure),
        "{{FRAME_STRIDE}}": str(settings.reset.frame_stride),
        "{{OUTPUT_DIR}}": str(run_root),
        "{{SETTINGS_PATH}}": str(settings_path),
    }
    for placeholder, value in values.items():
        template = template.replace(placeholder, value)
    unresolved = sorted(set(re.findall(r"\{\{[^{}]+\}\}", template)))
    if unresolved:
        raise ValueError(f"unresolved prompt placeholders: {unresolved}")
    return template


def write_task_prompt(
    settings: ExperimentSettings,
    output_dir: str | Path,
    filename: str = "prompt.md",
    *,
    run_root: str | Path | None = None,
    settings_path: str | Path | None = None,
) -> Path:
    """Write the complete config-rendered pre-repair prompt without a rollout."""

    output = Path(output_dir).resolve()
    root = Path(run_root).resolve() if run_root is not None else output
    source = (
        Path(settings_path).resolve()
        if settings_path is not None
        else root / "experiment.resolved.yaml"
    )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / filename
    destination.write_text(
        _render_prompt(settings, run_root=root, settings_path=source),
        encoding="utf-8",
    )
    return destination


def build_agent_prompt(
    rollout_dir: str | Path,
    output_dir: str | Path,
    settings: ExperimentSettings | None = None,
) -> dict[str, Any]:
    """Export only policy-visible evidence and a vendor-neutral agent prompt."""

    rollout = Path(rollout_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = _read_json(rollout / "summary.json")
    if settings is None:
        from ..parameters import load_settings

        resolved = rollout.parent / "experiment.resolved.yaml"
        if not resolved.is_file():
            resolved = rollout.parent.parent / "experiment.resolved.yaml"
        if not resolved.is_file():
            raise ValueError(
                "settings are required unless experiment.resolved.yaml exists"
            )
        settings = load_settings(resolved)
    validate_rollout_contract(summary, settings.fingerprint())
    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    for record in summary.get("episodes", []):
        index = int(record["episode_index"])
        episode = _read_json(_episode_path(rollout, index))
        validate_episode(episode)
        states = np.asarray(episode["states"], dtype=np.float64)
        actions = np.asarray(episode["actions"], dtype=np.float64)
        count = len(states)
        selected = sorted(
            set(np.linspace(0, max(0, count - 1), min(24, count), dtype=int).tolist())
        )
        item = {
            "episode_index": index,
            "outcome": "success" if bool(record.get("success")) else "failure",
            "num_frames": count,
            "wide_video": str(rollout / "videos" / f"episode_{index:06d}_wide.mp4"),
            "wrist_video": str(rollout / "videos" / f"episode_{index:06d}_wrist.mp4"),
            "timeline": [
                {
                    "frame_index": frame,
                    "state": states[frame].tolist(),
                    "action": actions[frame].tolist(),
                }
                for frame in selected
            ],
        }
        (successes if bool(record.get("success")) else failures).append(item)
    evidence = {
        "schema_version": 2,
        "observation_only": True,
        **diagnosis_evidence_metadata(summary),
        "successful_episode_count": len(successes),
        "failure_episode_count": len(failures),
        "successes": successes,
        "failures": failures,
    }
    LiberoRuntime.write_json(output / "agent_input.json", evidence)
    assert settings is not None
    run_root = rollout.parent
    write_task_prompt(
        settings,
        output,
        run_root=run_root,
        settings_path=run_root / "experiment.resolved.yaml",
    )
    return evidence


def validate_diagnosis(
    rollout_dir: str | Path, diagnosis_path: str | Path
) -> dict[str, Any]:
    rollout = Path(rollout_dir).resolve()
    summary = _read_json(rollout / "summary.json")
    if "trajectory_protocol" in summary:
        validate_rollout_contract(summary)
    expected = {
        int(item["episode_index"]): int(item["num_steps"])
        for item in summary.get("episodes", [])
        if not bool(item.get("success"))
    }
    diagnosis = _read_json(Path(diagnosis_path).resolve())
    if int(diagnosis.get("schema_version", 0)) != 1 or not isinstance(
        diagnosis.get("episodes"), list
    ):
        raise ValueError("diagnosis must use schema_version=1 and contain episodes[]")
    actual = [int(item.get("episode_index", -1)) for item in diagnosis["episodes"]]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError(
            f"diagnosis episode coverage mismatch: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    mode_records = diagnosis.get("failure_modes")
    if not isinstance(mode_records, list) or not mode_records:
        raise ValueError("diagnosis must contain failure_modes[]")
    mode_ids = {str(mode.get("failure_mode_id", "")) for mode in mode_records}
    if "" in mode_ids:
        raise ValueError("failure_modes[] contains an empty failure_mode_id")
    declared_mode_episodes: dict[str, set[int]] = {}
    for mode in mode_records:
        mode_id = str(mode.get("failure_mode_id", ""))
        if (
            not str(mode.get("label", "")).strip()
            or not str(mode.get("category", "")).strip()
        ):
            raise ValueError(f"failure mode {mode_id} must have label and category")
        declared = mode.get("episode_indices")
        if not isinstance(declared, list):
            raise ValueError(f"failure mode {mode_id} must contain episode_indices[]")
        declared_mode_episodes[mode_id] = {int(index) for index in declared}
        if not declared_mode_episodes[mode_id].issubset(expected):
            raise ValueError(f"failure mode {mode_id} references an unknown episode")
    actual_mode_episodes: dict[str, set[int]] = {}
    for item in diagnosis["episodes"]:
        index = int(item["episode_index"])
        causal = int(item["first_causal_frame_index"])
        start = int(item["recoverable_window_start_frame_index"])
        stop = int(item["recoverable_window_stop_frame_index"])
        if not str(item.get("failure_phase", "")).strip():
            raise ValueError(f"episode {index} has empty failure_phase")
        mode_id = str(item.get("failure_mode_id", ""))
        if mode_id not in mode_ids:
            raise ValueError(
                f"episode {index} references unknown failure_mode_id: {mode_id!r}"
            )
        if not str(item.get("failure_category", "")).strip():
            raise ValueError(f"episode {index} has empty failure_category")
        if not str(item.get("failure_mode", "")).strip():
            raise ValueError(f"episode {index} has empty failure_mode")
        actual_mode_episodes.setdefault(mode_id, set()).add(index)
        if not (0 <= start <= causal <= stop < expected[index]):
            raise ValueError(f"episode {index} has invalid causal/window indices")
        confidence = float(item.get("confidence", -1.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"episode {index} confidence must be in [0, 1]")
        if not isinstance(item.get("evidence"), list) or not item["evidence"]:
            raise ValueError(f"episode {index} must cite public evidence")
    if declared_mode_episodes != actual_mode_episodes:
        raise ValueError(
            f"failure mode episode membership mismatch: declared={declared_mode_episodes}, "
            f"actual={actual_mode_episodes}"
        )
    return diagnosis


def select_reset_candidates(
    settings: ExperimentSettings, diagnosis: dict[str, Any]
) -> dict[str, Any]:
    """Take ascending stride-filtered frames inside each diagnosed window."""

    selected: list[dict[str, Any]] = []
    for item in sorted(
        diagnosis["episodes"], key=lambda value: int(value["episode_index"])
    ):
        start = int(item["recoverable_window_start_frame_index"])
        stop = int(item["recoverable_window_stop_frame_index"])
        frames = list(range(start, stop + 1, settings.reset.frame_stride))
        if len(frames) < settings.reset.frames_per_failure:
            raise ValueError(
                f"episode {item['episode_index']} window [{start}, {stop}] yields only {len(frames)} "
                f"frames at stride {settings.reset.frame_stride}; need {settings.reset.frames_per_failure}"
            )
        for rank, frame in enumerate(frames[: settings.reset.frames_per_failure]):
            selected.append(
                {
                    "episode_index": int(item["episode_index"]),
                    "candidate_rank": rank,
                    "requested_frame_index": frame,
                    "failure_phase": item["failure_phase"],
                    "failure_mode_id": item.get("failure_mode_id", ""),
                    "failure_category": item.get(
                        "failure_category", item.get("failure_phase", "")
                    ),
                    "failure_mode": item.get(
                        "failure_mode", item.get("failure_phase", "")
                    ),
                    "window_start": start,
                    "window_stop": stop,
                }
            )
    return {
        "schema_version": 1,
        "selection": "ascending inclusive window stride",
        "frame_stride": settings.reset.frame_stride,
        "frames_per_failure": settings.reset.frames_per_failure,
        "candidates": selected,
    }


def _restore_gripper(env: Any, state: np.ndarray | None) -> None:
    if state is None:
        return
    gripper = env.env.robots[0].gripper
    desired = np.asarray(state, dtype=np.float64)
    current = np.asarray(gripper.current_action)
    if current.shape != desired.shape:
        if current.size == 1 or desired.size == 1:
            current[...] = desired.reshape(-1)[0]
            return
        raise RuntimeError(
            f"gripper controller shape mismatch: {current.shape} != {desired.shape}"
        )
    current[...] = desired


def _replay_one(
    settings: ExperimentSettings,
    rollout: Path,
    initial_states: np.ndarray,
    candidate: dict[str, Any],
    private_path: Path,
    agent_view_path: Path,
) -> dict[str, Any]:
    index = int(candidate["episode_index"])
    requested = int(candidate["requested_frame_index"])
    episode = _read_json(_episode_path(rollout, index))
    actions = np.asarray(episode["actions"], dtype=np.float32)
    recorded = np.asarray(episode["states"], dtype=np.float64)
    source = LiberoRuntime(
        settings.task.suite,
        settings.task.task_id,
        settings.controller.source_control_space,
        settings.rollout.control_frequency_hz,
        libero_root=settings.backend.libero_root,
    )
    scene_seed = int(episode["scene_model_seed"])
    env = source.new_env(scene_seed)
    try:
        obs = env.set_init_state(initial_states[int(episode["initial_state_index"])])
        for _ in range(10):
            obs, _, done, _ = env.step(source.neutral_action(env).tolist())
            if done:
                raise RuntimeError(
                    f"episode {index} terminated during replay stabilization"
                )
        initial_error = float(np.max(np.abs(source.public_state(obs) - recorded[0])))
        if initial_error > PUBLIC_REPLAY_TOLERANCE:
            raise RuntimeError(
                f"episode {index} initial replay error {initial_error} exceeds tolerance"
            )
        max_error = initial_error
        for action_index in range(requested):
            obs, _, done, _ = env.step(actions[action_index].tolist())
            if done:
                raise RuntimeError(
                    f"episode {index} ended before requested frame {requested}"
                )
            error = float(
                np.max(np.abs(source.public_state(obs) - recorded[action_index + 1]))
            )
            max_error = max(max_error, error)
            if error > PUBLIC_REPLAY_TOLERANCE:
                raise RuntimeError(
                    f"episode {index} diverged at frame {action_index + 1}: {error}"
                )
        import imageio.v2 as imageio

        agent_view, _ = source.observation_images(obs)
        agent_view_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(agent_view_path, agent_view)
        agent_view_sha256 = hashlib.sha256(
            np.ascontiguousarray(agent_view).view(np.uint8)
        ).hexdigest()
        sim_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        gripper_state = source.capture_gripper_controller(env)
    finally:
        env.close()

    target = LiberoRuntime(
        settings.task.suite,
        settings.task.task_id,
        settings.controller.target_control_space,
        settings.rollout.control_frequency_hz,
        libero_root=settings.backend.libero_root,
    )
    target_env = target.new_env(scene_seed)
    try:
        target_env.set_init_state(initial_states[int(episode["initial_state_index"])])
        target.set_sim_state(target_env, sim_state)
        _restore_gripper(target_env, gripper_state)
        dynamics = target.apply_reset_dynamics(target_env, settings.reset.dynamics)
        final_state = np.asarray(target_env.get_sim_state(), dtype=np.float64).copy()
        final_gripper = target.capture_gripper_controller(target_env)
        target.write_private_state(
            private_path, sim_state=final_state, gripper_state=final_gripper
        )
    finally:
        target_env.close()
    return {
        **candidate,
        "verified": True,
        "replayed_action_count": requested,
        "max_public_state_error": max_error,
        "public_tolerance": PUBLIC_REPLAY_TOLERANCE,
        "source_control_space": settings.controller.source_control_space,
        "target_control_space": settings.controller.target_control_space,
        "reset_dynamics": settings.reset.dynamics,
        "dynamics_audit": dynamics,
        "private_state": str(private_path.name),
        "private_state_sha256": target.state_hash(final_state),
        "agent_view": str(agent_view_path.name),
        "agent_view_sha256": agent_view_sha256,
    }


def materialize_reset_bank(
    settings: ExperimentSettings,
    rollout_dir: str | Path,
    diagnosis_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate all failures and atomically publish reset bank plus repair jobs."""

    rollout = Path(rollout_dir).resolve()
    output = Path(output_dir).resolve()
    summary = _read_json(rollout / "summary.json")
    validate_rollout_contract(summary, settings.fingerprint())
    diagnosis = validate_diagnosis(rollout, diagnosis_path)
    candidates = select_reset_candidates(settings, diagnosis)
    initial_states = np.load(rollout / "initial_states.npy", allow_pickle=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        reports: list[dict[str, Any]] = []
        for candidate in candidates["candidates"]:
            state_name = (
                f"episode_{candidate['episode_index']:06d}_"
                f"frame_{candidate['requested_frame_index']:06d}.npz"
            )
            reports.append(
                _replay_one(
                    settings,
                    rollout,
                    initial_states,
                    candidate,
                    temporary / "private_reset_states" / state_name,
                    temporary / "agent_views" / state_name.replace(".npz", ".png"),
                )
            )
        public = {
            "schema_version": 1,
            "settings_fingerprint": settings.fingerprint(),
            "count": len(reports),
            "resets": reports,
        }
        jobs = {
            "schema_version": 1,
            "jobs": [
                {
                    "job_id": f"e{item['episode_index']:06d}-f{item['requested_frame_index']:06d}",
                    "episode_index": item["episode_index"],
                    "reset_frame_index": item["requested_frame_index"],
                    "reset_state": item["private_state"],
                    "agent_view": item["agent_view"],
                    "target_control_space": item["target_control_space"],
                    "reset_dynamics": item["reset_dynamics"],
                    "failure_phase": item["failure_phase"],
                    "failure_mode_id": item["failure_mode_id"],
                    "failure_category": item["failure_category"],
                    "failure_mode": item["failure_mode"],
                }
                for item in reports
            ],
        }
        LiberoRuntime.write_json(temporary / "reset_candidates.json", candidates)
        LiberoRuntime.write_json(
            temporary / "replay_verification.json",
            {"schema_version": 1, "reports": reports},
        )
        LiberoRuntime.write_json(temporary / "public_reset_bank.json", public)
        LiberoRuntime.write_json(temporary / "repair_jobs.json", jobs)
        if output.exists():
            if not output.is_dir():
                raise FileExistsError(
                    f"refusing to replace existing reset bank: {output}"
                )
            children = list(temporary.iterdir())
            conflicts = [
                output / child.name
                for child in children
                if (output / child.name).exists()
            ]
            if conflicts:
                raise FileExistsError(
                    f"refusing to replace existing reset artifacts: {conflicts}"
                )
            for child in children:
                child.replace(output / child.name)
            temporary.rmdir()
        else:
            temporary.replace(output)
        return public
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
