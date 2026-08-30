"""Shared LIBERO initial-state loading, validation, and generation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..libero_runtime import LiberoRuntime, libero_imports
from ..parameters import ExperimentSettings


SUPPORTED_STATE_MANIFEST_KINDS = {
    "custom_bddl_sampler_initial_states",
    "robot_randomized_initial_states",
}


@dataclasses.dataclass(frozen=True)
class InitialStateBundle:
    """Validated simulator states and the provenance needed by evaluation."""

    states: np.ndarray
    entries: dict[int, dict[str, Any]] | None
    kind: str
    array_sha256: str
    manifest: dict[str, Any] | None = None
    manifest_path: Path | None = None


@dataclasses.dataclass(frozen=True)
class RandomizedStateBundle:
    """In-memory randomized states plus their cache documents."""

    states: np.ndarray
    manifest: dict[str, Any]
    validation: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class RobotInitialStateRandomization:
    """Deterministic, bounded OSC motion baked into an initial state.

    Scene placement and robot pose intentionally use different seeds.  The
    bounds are offsets from the reset EEF position, in metres.
    """

    seed_start: int = 2_000_000
    ee_offset_low: tuple[float, float, float] = (-0.04, -0.04, -0.02)
    ee_offset_high: tuple[float, float, float] = (0.04, 0.04, 0.04)
    minimum_offset_norm: float = 0.015
    position_tolerance: float = 0.003
    maximum_final_position_error: float = 0.005
    maximum_motion_steps: int = 60
    required_stable_steps: int = 3
    settle_steps: int = 10
    maximum_restore_ee_observation_error: float = 0.0001
    maximum_restore_ee_drift: float = 0.005

    def validate(self) -> None:
        low = np.asarray(self.ee_offset_low, dtype=np.float64)
        high = np.asarray(self.ee_offset_high, dtype=np.float64)
        if low.shape != (3,) or high.shape != (3,) or np.any(low >= high):
            raise ValueError("robot EEF offset bounds must be three ordered intervals")
        if self.minimum_offset_norm <= 0:
            raise ValueError("minimum robot EEF offset norm must be positive")
        if self.minimum_offset_norm > float(np.linalg.norm(np.maximum(abs(low), abs(high)))):
            raise ValueError("minimum robot EEF offset norm cannot be reached by the bounds")
        if (
            self.position_tolerance <= 0
            or self.maximum_final_position_error < self.position_tolerance
            or self.maximum_restore_ee_observation_error <= 0
            or self.maximum_restore_ee_drift <= 0
        ):
            raise ValueError("robot EEF tolerances must be positive")
        if (
            self.maximum_motion_steps <= 0
            or self.required_stable_steps <= 0
            or self.settle_steps < 0
        ):
            raise ValueError("robot motion steps must be positive and settle steps non-negative")


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def load_custom_initial_states(
    manifest_path: Path,
) -> tuple[dict[str, Any], np.ndarray, dict[int, dict[str, Any]], str]:
    """Load a fail-closed schema-v1 custom-state bundle."""

    path = manifest_path.expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(manifest.get("schema_version", 0)) != 1
        or manifest.get("kind") not in SUPPORTED_STATE_MANIFEST_KINDS
        or not isinstance(manifest.get("suite"), str)
        or not manifest["suite"]
        or int(manifest.get("task_id", -1)) < 0
        or not isinstance(manifest.get("state_file"), str)
        or not isinstance(manifest.get("entries"), list)
    ):
        raise ValueError(f"unsupported custom initial-state manifest: {path}")
    states_path = (path.parent / manifest["state_file"]).resolve()
    states = np.load(states_path, allow_pickle=False)
    expected_shape = tuple(int(item) for item in manifest.get("state_shape", ()))
    if (
        states.ndim != 2
        or states.shape != expected_shape
        or len(states) != int(manifest["count"])
    ):
        raise ValueError(f"custom initial-state array does not match manifest: {path}")
    if not np.isfinite(states).all():
        raise ValueError(
            f"custom initial-state array contains non-finite values: {path}"
        )

    entries: dict[int, dict[str, Any]] = {}
    for value in manifest["entries"]:
        if not isinstance(value, dict):
            raise ValueError(f"custom manifest entry is not a mapping: {path}")
        index = int(value.get("custom_initial_state_index", -1))
        if index in entries:
            raise ValueError(f"duplicate custom initial-state index {index}: {path}")
        entries[index] = value
    if sorted(entries) != list(range(len(states))):
        raise ValueError(f"custom initial-state indices are not contiguous: {path}")
    for index, state in enumerate(states):
        entry = entries[index]
        if int(entry.get("state_vector_index", -1)) != index:
            raise ValueError(f"custom state-vector index mismatch at {index}: {path}")
        if entry.get("simulator_state_sha256") != array_hash(state):
            raise ValueError(f"custom state hash mismatch at index {index}: {path}")
        if manifest["kind"] == "custom_bddl_sampler_initial_states":
            required_integer_fields = ("scene_model_seed", "placement_seed")
        else:
            required_integer_fields = ("official_initial_state_index", "robot_seed")
        for key in required_integer_fields:
            if not isinstance(entry.get(key), int):
                raise ValueError(f"custom manifest entry {index} has invalid {key}: {path}")
        if (
            entry.get("suite") != manifest["suite"]
            or int(entry.get("task_id", -1)) != int(manifest["task_id"])
            or entry.get("task") != manifest.get("task")
        ):
            raise ValueError(
                f"custom manifest entry {index} has a task mismatch: {path}"
            )

    validation_path = path.parent / "validation_report.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    state_array_hash = array_hash(states)
    if (
        int(validation.get("schema_version", 0)) != 1
        or not validation.get("valid")
        or int(validation.get("count", -1)) != len(states)
        or validation.get("state_array_sha256") != state_array_hash
    ):
        raise ValueError(f"custom initial-state validation failed: {path}")
    return manifest, states, entries, state_array_hash


def resolve_evaluation_initial_states(
    *,
    runtime: LiberoRuntime,
    manifest_path: Path | None,
    suite: str,
    task_id: int,
    task_description: str,
    control_frequency: int,
) -> InitialStateBundle:
    """Resolve official or strict manifest states for any evaluation caller."""

    if manifest_path is None:
        states = np.asarray(runtime.official_initial_states(), dtype=np.float64)
        if states.ndim != 2 or not len(states) or not np.isfinite(states).all():
            raise ValueError(f"invalid official initial-state array: {states.shape}")
        return InitialStateBundle(
            states=states,
            entries=None,
            kind="official",
            array_sha256=array_hash(states),
        )

    path = manifest_path.expanduser().resolve()
    manifest, states, entries, state_array_hash = load_custom_initial_states(path)
    if manifest["suite"] != suite or int(manifest["task_id"]) != task_id:
        raise ValueError(
            "custom initial-state manifest task mismatch: "
            f"manifest={manifest['suite']}:{manifest['task_id']}, "
            f"requested={suite}:{task_id}"
        )
    if str(manifest.get("task")) != task_description:
        raise ValueError(
            "custom initial-state task text does not match LIBERO benchmark"
        )
    if int(manifest.get("control_frequency", -1)) != control_frequency:
        raise ValueError(
            "custom initial-state control frequency does not match evaluation"
        )
    return InitialStateBundle(
        states=states,
        entries=entries,
        kind=str(manifest["kind"]),
        array_sha256=state_array_hash,
        manifest=manifest,
        manifest_path=path,
    )


def _load_workflow_state_manifest(
    path: Path,
    *,
    suite: str,
    task_id: int,
    task_description: str,
    control_frequency: int,
    count: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("kind") in SUPPORTED_STATE_MANIFEST_KINDS:
        manifest, states, entries, _ = load_custom_initial_states(path)
        if manifest["suite"] != suite or int(manifest["task_id"]) != task_id:
            raise ValueError(
                "state manifest task identity does not match experiment settings"
            )
        if str(manifest.get("task")) != task_description:
            raise ValueError(
                "state manifest task text does not match experiment settings"
            )
        if int(manifest.get("control_frequency", -1)) != control_frequency:
            raise ValueError(
                "state manifest control frequency does not match experiment settings"
            )
        if len(states) < count:
            raise ValueError(
                f"state manifest has {len(states)} states but {count} were requested"
            )
        provenance = [
            {
                **entries[index],
                "initial_state_index": index,
                "provider": "state_manifest",
            }
            for index in range(count)
        ]
        return np.asarray(states[:count], dtype=np.float64), provenance

    if int(document.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported state manifest schema: {path}")
    if document.get("suite") != suite or int(document.get("task_id", -1)) != task_id:
        raise ValueError(
            "state manifest task identity does not match experiment settings"
        )
    state_path = (path.parent / str(document["state_file"])).resolve()
    states = np.load(state_path, allow_pickle=False)
    if states.ndim != 2 or not np.isfinite(states).all():
        raise ValueError(f"state manifest contains an invalid state array: {path}")
    if len(states) < count:
        raise ValueError(
            f"state manifest has {len(states)} states but {count} were requested"
        )
    entries = document.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"state manifest entries must be a list: {path}")
    return np.asarray(states[:count], dtype=np.float64), entries[:count]


def _sample_workflow_randomized_states(
    settings: ExperimentSettings,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Use the same validated sampler as standalone randomized evaluation."""

    generated = _build_randomized_state_bundle(
        suite_name=settings.task.suite,
        task_id=settings.task.task_id,
        count=settings.initial_states.count,
        seed_start=settings.initial_states.seed_start,
        wait_steps=10,
        validation_hold_steps=5,
        control_freq=settings.rollout.control_frequency_hz,
        maximum_stabilization_drift=0.05,
        max_attempts=settings.initial_states.count * 20,
        libero_root=settings.backend.libero_root,
    )
    entries = [
        {
            **entry,
            "initial_state_index": index,
            "provider": "randomized_bddl",
        }
        for index, entry in enumerate(generated.manifest["entries"])
    ]
    return generated.states, entries


def generate_initial_states(
    settings: ExperimentSettings,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Resolve the initial states used by the pre-repair workflow."""

    provider = settings.initial_states.provider
    count = settings.initial_states.count
    if provider == "state_manifest":
        assert settings.initial_states.state_manifest is not None
        runtime = LiberoRuntime(
            settings.task.suite,
            settings.task.task_id,
            settings.controller.source_control_space,
            settings.rollout.control_frequency_hz,
            libero_root=settings.backend.libero_root,
        )
        return _load_workflow_state_manifest(
            settings.initial_states.state_manifest,
            suite=settings.task.suite,
            task_id=settings.task.task_id,
            task_description=(
                settings.task.task_description or runtime.task_description()
            ),
            control_frequency=settings.rollout.control_frequency_hz,
            count=count,
        )
    if provider == "randomized_bddl":
        return _sample_workflow_randomized_states(settings)
    runtime = LiberoRuntime(
        settings.task.suite,
        settings.task.task_id,
        settings.controller.source_control_space,
        settings.rollout.control_frequency_hz,
        libero_root=settings.backend.libero_root,
    )
    states = runtime.official_initial_states()[:count].copy()
    if len(states) < count:
        raise ValueError(
            f"official LIBERO provides {len(states)} states but {count} were requested"
        )
    entries = [
        {
            "initial_state_index": index,
            "provider": provider,
            "scene_model_seed": settings.rollout.policy_seed,
        }
        for index in range(count)
    ]
    return states, entries


RANDOMIZED_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


class RejectedSceneError(RuntimeError):
    """A deterministic placement seed is unsuitable for evaluation."""


def _generator_imports(
    libero_root: Path | None = None,
) -> tuple[Any, Any, Any]:
    benchmark, env_class, get_libero_path = libero_imports(libero_root)
    return benchmark, get_libero_path, env_class


def _body_pose(env: Any, body_name: str) -> np.ndarray:
    body_id = env.sim.model.body_name2id(body_name)
    return np.concatenate(
        (
            np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64),
            np.asarray(env.sim.data.body_xquat[body_id], dtype=np.float64),
        )
    )


def _model_body_pose(env: Any, body_name: str) -> np.ndarray:
    body_id = env.sim.model.body_name2id(body_name)
    return np.concatenate(
        (
            np.asarray(env.sim.model.body_pos[body_id], dtype=np.float64),
            np.asarray(env.sim.model.body_quat[body_id], dtype=np.float64),
        )
    )


def _poses(env: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    objects = {obj.name: _body_pose(env, obj.root_body) for obj in env.env.objects}
    fixtures = {
        fixture.name: _model_body_pose(env, fixture.root_body)
        for fixture in env.env.fixtures
    }
    return objects, fixtures


def _signature(
    objects: dict[str, np.ndarray], fixtures: dict[str, np.ndarray]
) -> np.ndarray:
    poses = {**objects, **fixtures}
    if not poses:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate([np.asarray(poses[name]) for name in sorted(poses)])


def _max_object_drift(
    before: dict[str, np.ndarray], after: dict[str, np.ndarray]
) -> float:
    if not before:
        return 0.0
    return max(
        float(
            np.linalg.norm(np.asarray(before[name][:3]) - np.asarray(after[name][:3]))
        )
        for name in before
    )


def _hold(env: Any, obs: dict[str, Any], steps: int) -> tuple[dict[str, Any], bool]:
    done = False
    for _ in range(steps):
        obs, _, done, _ = env.step(RANDOMIZED_DUMMY_ACTION.tolist())
        if done:
            break
    return obs, bool(done)


def _current_observation(env: Any) -> dict[str, Any]:
    """Read observations after a state restore without advancing simulation."""

    if hasattr(env, "regenerate_obs_from_state"):
        state = np.asarray(env.get_sim_state(), dtype=np.float64)
        return env.regenerate_obs_from_state(state)
    return env.env._get_observations()


def _sample_ee_offset(
    robot_seed: int, spec: RobotInitialStateRandomization
) -> np.ndarray:
    rng = np.random.default_rng(robot_seed)
    low = np.asarray(spec.ee_offset_low, dtype=np.float64)
    high = np.asarray(spec.ee_offset_high, dtype=np.float64)
    for _ in range(10_000):
        offset = rng.uniform(low, high)
        if float(np.linalg.norm(offset)) >= spec.minimum_offset_norm:
            return offset
    raise RuntimeError("failed to sample a robot EEF offset satisfying the norm bound")


def _move_ee_to_randomized_initial_pose(
    *,
    env: Any,
    obs: dict[str, Any],
    robot_seed: int,
    spec: RobotInitialStateRandomization,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    """Move the EEF with actual OSC commands and return auditable provenance."""

    if "robot0_eef_pos" not in obs or "robot0_joint_pos" not in obs:
        raise RejectedSceneError("OSC robot observations are unavailable")
    start_ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
    start_joints = np.asarray(obs["robot0_joint_pos"], dtype=np.float64).copy()
    requested_offset = _sample_ee_offset(robot_seed, spec)
    target_ee = start_ee + requested_offset
    low, high = env.env.action_spec
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    if low.shape != (7,) or high.shape != (7,):
        raise RejectedSceneError(
            f"robot initial-state randomization requires 7-D OSC actions, got {low.shape}"
        )

    done = False
    motion_steps = 0
    stable_steps = 0
    previous_ee = start_ee.copy()
    command_hasher = hashlib.sha256()
    for motion_steps in range(1, spec.maximum_motion_steps + 1):
        current = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        error = target_ee - current
        position_error = float(np.linalg.norm(error))
        motion_since_previous = float(np.linalg.norm(current - previous_ee))
        if position_error <= spec.position_tolerance and motion_since_previous <= 0.0005:
            stable_steps += 1
        else:
            stable_steps = 0
        if stable_steps >= spec.required_stable_steps:
            motion_steps -= 1
            break
        previous_ee = current.copy()
        action = np.zeros(7, dtype=np.float32)
        # Robosuite OSC_POSE maps a unit translation command to 5 cm.
        if position_error > spec.position_tolerance:
            action[:3] = np.clip(error / 0.05, low[:3], high[:3])
        action[-1] = np.clip(-1.0, low[-1], high[-1])
        command_hasher.update(np.ascontiguousarray(action).view(np.uint8))
        obs, _, done, _ = env.step(action.tolist())
        if done or bool(env.env._check_success()):
            break
    else:
        motion_steps = spec.maximum_motion_steps

    reached_ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
    reached_error = float(np.linalg.norm(target_ee - reached_ee))
    if done or bool(env.env._check_success()):
        raise RejectedSceneError("task terminated or succeeded during robot initialization")
    if reached_error > spec.position_tolerance:
        raise RejectedSceneError(
            "robot EEF target was not reached: "
            f"error={reached_error}, tolerance={spec.position_tolerance}"
        )

    obs, settle_done = _hold(env, obs, spec.settle_steps)
    final_ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
    final_joints = np.asarray(obs["robot0_joint_pos"], dtype=np.float64).copy()
    final_error = float(np.linalg.norm(target_ee - final_ee))
    if settle_done or bool(env.env._check_success()):
        raise RejectedSceneError("task terminated or succeeded while robot state settled")
    if final_error > spec.maximum_final_position_error:
        raise RejectedSceneError(
            "robot EEF drifted outside target tolerance while settling: "
            f"error={final_error}, tolerance={spec.maximum_final_position_error}"
        )
    return (
        obs,
        False,
        {
            "robot_seed": int(robot_seed),
            "robot_initial_eef_xyz": start_ee.tolist(),
            "robot_requested_eef_offset_xyz": requested_offset.tolist(),
            "robot_target_eef_xyz": target_ee.tolist(),
            "robot_achieved_eef_xyz": final_ee.tolist(),
            "robot_achieved_eef_offset_xyz": (final_ee - start_ee).tolist(),
            "robot_target_error_m": final_error,
            "robot_initial_joint_positions": start_joints.tolist(),
            "robot_achieved_joint_positions": final_joints.tolist(),
            "robot_osc_motion_steps": int(motion_steps),
            "robot_required_stable_steps": int(spec.required_stable_steps),
            "robot_osc_settle_steps": int(spec.settle_steps),
            "robot_osc_command_sha256": command_hasher.hexdigest(),
        },
    )


def existing_randomized_cache(
    output: Path, expected: dict[str, Any]
) -> dict[str, Any] | None:
    if not output.exists():
        return None
    manifest_path = output / "manifest.json"
    validation_path = output / "validation_report.json"
    state_path = output / "states.npy"
    if not all(path.is_file() for path in (manifest_path, validation_path, state_path)):
        raise FileExistsError(f"incomplete randomized-scene cache: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"cached": manifest.get(key), "requested": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise FileExistsError(f"randomized-scene cache contract mismatch: {mismatches}")
    states = np.load(state_path, allow_pickle=False)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    required_audits = (
        "all_distinct_from_official_states",
        "all_initial_predicates_false",
        "all_sampler_replays_exact",
        "all_explicit_restores_exact",
    )
    if expected.get("robot_initial_state_randomization") == "actual_osc_command":
        required_audits += ("all_robot_initial_states_use_actual_osc_commands",)
    entries = manifest.get("entries")
    entries_valid = isinstance(entries, list) and len(entries) == int(expected["count"])
    if entries_valid:
        try:
            entries_valid = all(
                int(entry["custom_initial_state_index"]) == index
                and int(entry["state_vector_index"]) == index
                and entry["simulator_state_sha256"] == array_hash(states[index])
                for index, entry in enumerate(entries)
            )
        except (KeyError, TypeError, ValueError):
            entries_valid = False
    if (
        int(validation.get("schema_version", 0)) != 1
        or not validation.get("valid")
        or int(validation.get("count", -1)) != int(expected["count"])
        or int(validation.get("unique_state_hashes", -1)) != int(expected["count"])
        or not all(validation.get(name) for name in required_audits)
        or validation.get("allowed_maximum_stabilization_drift")
        != expected["allowed_maximum_stabilization_drift"]
        or float(validation.get("maximum_stabilization_drift", math.inf))
        > float(expected["allowed_maximum_stabilization_drift"])
        or (
            expected.get("robot_initial_state_randomization") == "actual_osc_command"
            and (
                int(validation.get("unique_robot_seeds", -1)) != int(expected["count"])
                or int(validation.get("unique_robot_achieved_eef_xyz", -1))
                != int(expected["count"])
                or float(validation.get("maximum_restore_ee_drift_m", math.inf))
                > float(expected["allowed_maximum_restore_ee_drift_m"])
                or float(
                    validation.get("maximum_restore_ee_observation_error_m", math.inf)
                )
                > float(expected["allowed_maximum_restore_ee_observation_error_m"])
            )
        )
        or states.ndim != 2
        or len(states) != int(expected["count"])
        or list(states.shape) != manifest.get("state_shape")
        or not entries_valid
        or array_hash(states) != validation.get("state_array_sha256")
    ):
        raise ValueError(f"randomized-scene cache failed validation: {output}")
    return manifest


def _sample_scene(
    *,
    env: Any,
    placement_seed: int,
    robot_seed: int,
    robot_randomization: RobotInitialStateRandomization,
    expected_shape: tuple[int, ...],
    wait_steps: int,
    validation_hold_steps: int,
    maximum_stabilization_drift: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    robot_randomization.validate()
    env.seed(placement_seed)
    try:
        obs = env.reset()
    except Exception as exc:
        raise RejectedSceneError(f"native sampler reset failed: {exc}") from exc
    initial_success = bool(env.env._check_success())
    raw_objects, _ = _poses(env)
    obs, stabilization_done = _hold(env, obs, wait_steps)
    stabilized_objects, _ = _poses(env)
    settling_drift = _max_object_drift(raw_objects, stabilized_objects)
    stabilized_success = bool(env.env._check_success())
    if initial_success or stabilization_done or stabilized_success:
        raise RejectedSceneError("task predicate is true during initialization")
    obs, _, robot_audit = _move_ee_to_randomized_initial_pose(
        env=env,
        obs=obs,
        robot_seed=robot_seed,
        spec=robot_randomization,
    )
    state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    objects, fixtures = _poses(env)
    robot_motion_object_drift = _max_object_drift(stabilized_objects, objects)
    if state.shape != expected_shape or not np.isfinite(state).all():
        raise RejectedSceneError(f"invalid state shape/value: {state.shape}")

    env.seed(placement_seed)
    replay_obs = env.reset()
    replay_obs, replay_done = _hold(env, replay_obs, wait_steps)
    replay_obs, _, replay_robot_audit = _move_ee_to_randomized_initial_pose(
        env=env,
        obs=replay_obs,
        robot_seed=robot_seed,
        spec=robot_randomization,
    )
    replay_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    replay_objects, replay_fixtures = _poses(env)
    state_error = float(np.max(np.abs(replay_state - state)))
    pose_delta = _signature(replay_objects, replay_fixtures) - _signature(
        objects, fixtures
    )
    pose_error = float(np.max(np.abs(pose_delta))) if pose_delta.size else 0.0
    replay_robot_error = float(
        np.max(
            np.abs(
                np.asarray(replay_robot_audit["robot_achieved_eef_xyz"])
                - np.asarray(robot_audit["robot_achieved_eef_xyz"])
            )
        )
    )
    if (
        replay_done
        or state_error > 1e-12
        or pose_error > 1e-12
        or replay_robot_error > 1e-12
    ):
        raise RejectedSceneError(
            f"sampler replay mismatch: done={replay_done}, "
            f"state={state_error}, pose={pose_error}, robot={replay_robot_error}"
        )

    env.seed(placement_seed)
    env.reset()
    LiberoRuntime.set_sim_state(env, state)
    restored_obs = _current_observation(env)
    restore_error = float(
        np.max(np.abs(np.asarray(env.get_sim_state(), dtype=np.float64) - state))
    )
    restored_ee = np.asarray(restored_obs["robot0_eef_pos"], dtype=np.float64).copy()
    saved_ee = np.asarray(robot_audit["robot_achieved_eef_xyz"], dtype=np.float64)
    restore_ee_error = float(np.linalg.norm(restored_ee - saved_ee))
    restored_objects, _ = _poses(env)
    restored_obs, validation_done = _hold(env, restored_obs, validation_hold_steps)
    validation_objects, _ = _poses(env)
    validation_drift = _max_object_drift(restored_objects, validation_objects)
    validation_ee = np.asarray(restored_obs["robot0_eef_pos"], dtype=np.float64)
    validation_ee_drift = float(np.linalg.norm(validation_ee - restored_ee))
    if (
        restore_error > 1e-12
        or restore_ee_error
        > robot_randomization.maximum_restore_ee_observation_error
        or validation_done
        or bool(env.env._check_success())
        or validation_drift > maximum_stabilization_drift
        or validation_ee_drift > robot_randomization.maximum_restore_ee_drift
    ):
        raise RejectedSceneError(
            "restore/hold validation failed: "
            f"restore={restore_error}, restore_ee={restore_ee_error}, "
            f"done={validation_done}, object_drift={validation_drift}, "
            f"ee_drift={validation_ee_drift}"
        )
    return (
        state,
        _signature(objects, fixtures),
        {
            "initial_predicate_success": initial_success,
            "post_stabilization_predicate_success": stabilized_success,
            "sampler_replay_state_max_abs_error": state_error,
            "sampler_replay_pose_max_abs_error": pose_error,
            "sampler_replay_robot_eef_max_abs_error": replay_robot_error,
            "explicit_restore_state_max_abs_error": restore_error,
            "explicit_restore_robot_eef_error_m": restore_ee_error,
            "object_position_drift_during_initial_settling": settling_drift,
            "maximum_object_position_drift_during_robot_motion": robot_motion_object_drift,
            "maximum_object_position_drift_during_validation_hold": validation_drift,
            "robot_eef_drift_during_validation_hold_m": validation_ee_drift,
            **robot_audit,
        },
    )


def _sample_official_robot_state(
    *,
    env: Any,
    official_state: np.ndarray,
    official_index: int,
    scene_model_seed: int,
    robot_seed: int,
    robot_randomization: RobotInitialStateRandomization,
    wait_steps: int,
    validation_hold_steps: int,
    maximum_stabilization_drift: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bake a real OSC EEF perturbation into one official LIBERO state."""

    def initialize() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
        env.seed(scene_model_seed)
        env.reset()
        LiberoRuntime.set_sim_state(env, official_state)
        base_restore_error = float(
            np.max(
                np.abs(
                    np.asarray(env.get_sim_state(), dtype=np.float64)
                    - np.asarray(official_state, dtype=np.float64)
                )
            )
        )
        obs = _current_observation(env)
        initial_success = bool(env.env._check_success())
        raw_objects, _ = _poses(env)
        obs, stabilization_done = _hold(env, obs, wait_steps)
        stabilized_objects, _ = _poses(env)
        stabilized_success = bool(env.env._check_success())
        if initial_success or stabilization_done or stabilized_success:
            raise RejectedSceneError("official task predicate is true during initialization")
        obs, _, robot_audit = _move_ee_to_randomized_initial_pose(
            env=env,
            obs=obs,
            robot_seed=robot_seed,
            spec=robot_randomization,
        )
        state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        final_objects, _ = _poses(env)
        audit = {
            "official_source_restore_state_max_abs_error": base_restore_error,
            "initial_predicate_success": initial_success,
            "post_stabilization_predicate_success": stabilized_success,
            "object_position_drift_during_initial_settling": _max_object_drift(
                raw_objects, stabilized_objects
            ),
            "maximum_object_position_drift_during_robot_motion": _max_object_drift(
                stabilized_objects, final_objects
            ),
            **robot_audit,
        }
        return state, final_objects, audit

    state, objects, robot_audit = initialize()
    if state.shape != official_state.shape or not np.isfinite(state).all():
        raise RejectedSceneError(f"invalid official robot state: {state.shape}")
    replay_state, replay_objects, replay_audit = initialize()
    state_error = float(np.max(np.abs(replay_state - state)))
    object_error_values = _signature(replay_objects, {}) - _signature(objects, {})
    object_error = (
        float(np.max(np.abs(object_error_values))) if object_error_values.size else 0.0
    )
    replay_robot_error = float(
        np.max(
            np.abs(
                np.asarray(replay_audit["robot_achieved_eef_xyz"])
                - np.asarray(robot_audit["robot_achieved_eef_xyz"])
            )
        )
    )
    if state_error > 1e-12 or object_error > 1e-12 or replay_robot_error > 1e-12:
        raise RejectedSceneError(
            "official robot-state replay mismatch: "
            f"state={state_error}, object={object_error}, robot={replay_robot_error}"
        )

    env.seed(scene_model_seed)
    env.reset()
    LiberoRuntime.set_sim_state(env, state)
    restored_obs = _current_observation(env)
    restore_error = float(
        np.max(np.abs(np.asarray(env.get_sim_state(), dtype=np.float64) - state))
    )
    saved_ee = np.asarray(robot_audit["robot_achieved_eef_xyz"], dtype=np.float64)
    restored_ee = np.asarray(restored_obs["robot0_eef_pos"], dtype=np.float64)
    restore_ee_error = float(np.linalg.norm(restored_ee - saved_ee))
    restored_objects, _ = _poses(env)
    restored_obs, validation_done = _hold(env, restored_obs, validation_hold_steps)
    validation_objects, _ = _poses(env)
    validation_object_drift = _max_object_drift(restored_objects, validation_objects)
    validation_ee = np.asarray(restored_obs["robot0_eef_pos"], dtype=np.float64)
    validation_ee_drift = float(np.linalg.norm(validation_ee - restored_ee))
    if (
        restore_error > 1e-12
        or restore_ee_error > robot_randomization.maximum_restore_ee_observation_error
        or validation_done
        or bool(env.env._check_success())
        or validation_object_drift > maximum_stabilization_drift
        or validation_ee_drift > robot_randomization.maximum_restore_ee_drift
    ):
        raise RejectedSceneError(
            "official restore/hold validation failed: "
            f"restore={restore_error}, restore_ee={restore_ee_error}, "
            f"done={validation_done}, object_drift={validation_object_drift}, "
            f"ee_drift={validation_ee_drift}"
        )
    return state, {
        **robot_audit,
        "official_initial_state_index": int(official_index),
        "official_source_state_sha256": array_hash(official_state),
        "sampler_replay_state_max_abs_error": state_error,
        "sampler_replay_pose_max_abs_error": object_error,
        "sampler_replay_robot_eef_max_abs_error": replay_robot_error,
        "explicit_restore_state_max_abs_error": restore_error,
        "explicit_restore_robot_eef_error_m": restore_ee_error,
        "maximum_object_position_drift_during_validation_hold": validation_object_drift,
        "robot_eef_drift_during_validation_hold_m": validation_ee_drift,
    }


def generate_official_robot_states(
    *,
    output: Path,
    suite_name: str,
    task_id: int,
    count: int,
    control_freq: int,
    wait_steps: int,
    validation_hold_steps: int,
    maximum_stabilization_drift: float,
    scene_model_seed: int = 7,
    libero_root: Path | None = None,
    robot_randomization: RobotInitialStateRandomization | None = None,
) -> dict[str, Any]:
    """Materialize official scenes with independently randomized robot poses."""

    robot_randomization = robot_randomization or RobotInitialStateRandomization()
    robot_randomization.validate()
    expected_contract = {
        "schema_version": 1,
        "kind": "robot_randomized_initial_states",
        "scene_source": "official",
        "suite": suite_name,
        "task_id": task_id,
        "count": count,
        "control_frequency": control_freq,
        "scene_model_seed": scene_model_seed,
        "official_state_index_start": 0,
        "non_training_stabilization_steps_baked_into_state": wait_steps,
        "validation_hold_steps": validation_hold_steps,
        "allowed_maximum_stabilization_drift": maximum_stabilization_drift,
        "robot_initial_state_randomization": "actual_osc_command",
        "robot_seed_start": robot_randomization.seed_start,
        "robot_ee_offset_low_m": list(robot_randomization.ee_offset_low),
        "robot_ee_offset_high_m": list(robot_randomization.ee_offset_high),
        "robot_minimum_ee_offset_norm_m": robot_randomization.minimum_offset_norm,
        "robot_position_tolerance_m": robot_randomization.position_tolerance,
        "robot_maximum_final_position_error_m": (
            robot_randomization.maximum_final_position_error
        ),
        "robot_maximum_motion_steps": robot_randomization.maximum_motion_steps,
        "robot_required_stable_steps": robot_randomization.required_stable_steps,
        "robot_settle_steps_baked_into_state": robot_randomization.settle_steps,
        "allowed_maximum_restore_ee_observation_error_m": (
            robot_randomization.maximum_restore_ee_observation_error
        ),
        "allowed_maximum_restore_ee_drift_m": robot_randomization.maximum_restore_ee_drift,
    }
    if output.exists():
        manifest, states, entries, _ = load_custom_initial_states(output / "manifest.json")
        mismatches = {
            key: {"cached": manifest.get(key), "requested": value}
            for key, value in expected_contract.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise FileExistsError(f"official robot-state cache contract mismatch: {mismatches}")
        if len(states) != count or len(entries) != count:
            raise ValueError(f"official robot-state cache has an invalid count: {output}")
        return manifest

    benchmark, get_libero_path, env_class = _generator_imports(libero_root)
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"unknown LIBERO suite: {suite_name}")
    suite = benchmark_dict[suite_name]()
    task = suite.get_task(task_id)
    official = np.asarray(suite.get_task_init_states(task_id), dtype=np.float64)
    if official.ndim != 2 or len(official) < count:
        raise ValueError(f"official LIBERO has {len(official)} states; requested {count}")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    env = env_class(
        bddl_file_name=str(bddl),
        camera_heights=64,
        camera_widths=64,
        control_freq=control_freq,
        controller="OSC_POSE",
    )
    try:
        states: list[np.ndarray] = []
        entries: list[dict[str, Any]] = []
        for index in range(count):
            robot_seed = robot_randomization.seed_start + index
            state, audit = _sample_official_robot_state(
                env=env,
                official_state=official[index],
                official_index=index,
                scene_model_seed=scene_model_seed,
                robot_seed=robot_seed,
                robot_randomization=robot_randomization,
                wait_steps=wait_steps,
                validation_hold_steps=validation_hold_steps,
                maximum_stabilization_drift=maximum_stabilization_drift,
            )
            states.append(state)
            entries.append(
                {
                    "custom_initial_state_index": index,
                    "state_vector_index": index,
                    "scene_source": "official",
                    "scene_model_seed": scene_model_seed,
                    "suite": suite_name,
                    "task_id": task_id,
                    "task": str(task.language),
                    "state_dimension": int(state.size),
                    "simulator_state_sha256": array_hash(state),
                    **audit,
                }
            )
            print(
                f"Accepted official robot state {index + 1}/{count}: "
                f"official_index={index}, robot_seed={robot_seed}",
                flush=True,
            )
        state_array = np.stack(states)
        np.save(temporary / "states.npy", state_array)
        manifest = {
            **expected_contract,
            "generated_at": datetime.now(UTC).isoformat(),
            "task_name": str(task.name),
            "task": str(task.language),
            "official": True,
            "state_file": "states.npy",
            "state_shape": list(state_array.shape),
            "entries": entries,
        }
        validation = {
            "schema_version": 1,
            "valid": True,
            "count": count,
            "unique_state_hashes": len({item["simulator_state_sha256"] for item in entries}),
            "unique_robot_seeds": len({item["robot_seed"] for item in entries}),
            "unique_robot_achieved_eef_xyz": len(
                {tuple(item["robot_achieved_eef_xyz"]) for item in entries}
            ),
            "all_initial_predicates_false": True,
            "all_sampler_replays_exact": True,
            "all_explicit_restores_exact": True,
            "all_robot_initial_states_use_actual_osc_commands": True,
            "maximum_stabilization_drift": max(
                item["maximum_object_position_drift_during_validation_hold"]
                for item in entries
            ),
            "allowed_maximum_stabilization_drift": maximum_stabilization_drift,
            "maximum_restore_ee_observation_error_m": max(
                item["explicit_restore_robot_eef_error_m"] for item in entries
            ),
            "allowed_maximum_restore_ee_observation_error_m": (
                robot_randomization.maximum_restore_ee_observation_error
            ),
            "maximum_restore_ee_drift_m": max(
                item["robot_eef_drift_during_validation_hold_m"] for item in entries
            ),
            "allowed_maximum_restore_ee_drift_m": robot_randomization.maximum_restore_ee_drift,
            "state_array_sha256": array_hash(state_array),
        }
        LiberoRuntime.write_json(temporary / "manifest.json", manifest)
        LiberoRuntime.write_json(temporary / "validation_report.json", validation)
        temporary.replace(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        env.close()


def generate_randomized_states(
    *,
    output: Path,
    suite_name: str,
    task_id: int,
    count: int,
    seed_start: int,
    wait_steps: int,
    validation_hold_steps: int,
    control_freq: int,
    maximum_stabilization_drift: float,
    max_attempts: int,
    libero_root: Path | None = None,
    robot_randomization: RobotInitialStateRandomization | None = None,
) -> dict[str, Any]:
    """Build a deterministic cache varying both BDDL scene and robot pose."""

    robot_randomization = robot_randomization or RobotInitialStateRandomization()
    robot_randomization.validate()

    expected_contract = {
        "schema_version": 1,
        "kind": "custom_bddl_sampler_initial_states",
        "suite": suite_name,
        "task_id": task_id,
        "count": count,
        "placement_seed_start": seed_start,
        "control_frequency": control_freq,
        "non_training_stabilization_steps_baked_into_state": wait_steps,
        "validation_hold_steps": validation_hold_steps,
        "allowed_maximum_stabilization_drift": maximum_stabilization_drift,
        "robot_initial_state_randomization": "actual_osc_command",
        "robot_seed_start": robot_randomization.seed_start,
        "robot_ee_offset_low_m": list(robot_randomization.ee_offset_low),
        "robot_ee_offset_high_m": list(robot_randomization.ee_offset_high),
        "robot_minimum_ee_offset_norm_m": robot_randomization.minimum_offset_norm,
        "robot_position_tolerance_m": robot_randomization.position_tolerance,
        "robot_maximum_final_position_error_m": (
            robot_randomization.maximum_final_position_error
        ),
        "robot_maximum_motion_steps": robot_randomization.maximum_motion_steps,
        "robot_required_stable_steps": robot_randomization.required_stable_steps,
        "robot_settle_steps_baked_into_state": robot_randomization.settle_steps,
        "allowed_maximum_restore_ee_observation_error_m": (
            robot_randomization.maximum_restore_ee_observation_error
        ),
        "allowed_maximum_restore_ee_drift_m": robot_randomization.maximum_restore_ee_drift,
    }
    cached = existing_randomized_cache(output, expected_contract)
    if cached is not None:
        print(f"Reusing randomized scene cache: {output}", flush=True)
        return cached

    benchmark, get_libero_path, env_class = _generator_imports(libero_root)
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"unknown LIBERO suite: {suite_name}")
    suite = benchmark_dict[suite_name]()
    if not 0 <= task_id < suite.n_tasks:
        raise ValueError(
            f"invalid task ID {task_id} for {suite_name}; count={suite.n_tasks}"
        )
    task = suite.get_task(task_id)
    official = np.asarray(suite.get_task_init_states(task_id), dtype=np.float64)
    if official.ndim != 2 or not len(official):
        raise ValueError(f"invalid official state array: {official.shape}")
    expected_shape = tuple(official.shape[1:])
    official_hashes = {array_hash(state) for state in official}
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    env = env_class(
        bddl_file_name=str(bddl),
        camera_heights=64,
        camera_widths=64,
        control_freq=control_freq,
    )
    states: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    try:
        actual_frequency = int(env.env.control_freq)
        if actual_frequency != control_freq:
            raise RuntimeError(
                "control frequency mismatch: "
                f"requested={control_freq}, actual={actual_frequency}"
            )
        for offset in range(max_attempts):
            if len(states) == count:
                break
            placement_seed = seed_start + offset
            robot_seed = robot_randomization.seed_start + offset
            try:
                state, signature, audit = _sample_scene(
                    env=env,
                    placement_seed=placement_seed,
                    robot_seed=robot_seed,
                    robot_randomization=robot_randomization,
                    expected_shape=expected_shape,
                    wait_steps=wait_steps,
                    validation_hold_steps=validation_hold_steps,
                    maximum_stabilization_drift=maximum_stabilization_drift,
                )
                digest = array_hash(state)
                if digest in official_hashes:
                    raise RejectedSceneError("state duplicates an official state")
                if any(digest == item["simulator_state_sha256"] for item in entries):
                    raise RejectedSceneError(
                        "state duplicates an earlier randomized state"
                    )
                if any(np.array_equal(signature, previous) for previous in signatures):
                    raise RejectedSceneError(
                        "placement signature duplicates an earlier scene"
                    )
            except RejectedSceneError as exc:
                rejections.append(
                    {
                        "placement_seed": placement_seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"Rejected scene seed {placement_seed}: {exc}", flush=True)
                continue
            index = len(states)
            states.append(state)
            signatures.append(signature)
            entries.append(
                {
                    "custom_initial_state_index": index,
                    "placement_seed": placement_seed,
                    "scene_model_seed": placement_seed,
                    "suite": suite_name,
                    "task_id": task_id,
                    "task": str(task.language),
                    "state_vector_index": index,
                    "state_dimension": int(state.size),
                    "simulator_state_sha256": digest,
                    "official_state_duplicate": False,
                    **audit,
                }
            )
            print(
                f"Accepted randomized scene {index + 1}/{count}: "
                f"placement_seed={placement_seed}, robot_seed={robot_seed}",
                flush=True,
            )
        if len(states) != count:
            raise RuntimeError(
                f"generated only {len(states)}/{count} scenes "
                f"after {max_attempts} attempts"
            )

        state_array = np.stack(states)
        np.save(temporary / "states.npy", state_array)
        signature_array = np.stack(signatures)
        if count > 1:
            distances = np.linalg.norm(
                signature_array[:, None, :] - signature_array[None, :, :], axis=-1
            )
            distances[np.eye(count, dtype=bool)] = math.inf
            minimum_distance = float(np.min(distances))
        else:
            minimum_distance = None
        manifest = {
            **expected_contract,
            "generated_at": datetime.now(UTC).isoformat(),
            "task_name": str(task.name),
            "task": str(task.language),
            "official": False,
            "state_file": "states.npy",
            "state_shape": list(state_array.shape),
            "placement_seed_stop_exclusive": int(entries[-1]["placement_seed"]) + 1,
            "accepted_placement_seeds": [item["placement_seed"] for item in entries],
            "entries": entries,
            "rejections": rejections,
        }
        validation = {
            "schema_version": 1,
            "valid": True,
            "count": count,
            "unique_state_hashes": len(
                {item["simulator_state_sha256"] for item in entries}
            ),
            "all_distinct_from_official_states": True,
            "all_initial_predicates_false": True,
            "all_sampler_replays_exact": True,
            "all_explicit_restores_exact": True,
            "all_robot_initial_states_use_actual_osc_commands": True,
            "unique_robot_seeds": len({item["robot_seed"] for item in entries}),
            "unique_robot_achieved_eef_xyz": len(
                {
                    tuple(float(value) for value in item["robot_achieved_eef_xyz"])
                    for item in entries
                }
            ),
            "maximum_stabilization_drift": max(
                item["maximum_object_position_drift_during_validation_hold"]
                for item in entries
            ),
            "allowed_maximum_stabilization_drift": maximum_stabilization_drift,
            "maximum_restore_ee_drift_m": max(
                item["robot_eef_drift_during_validation_hold_m"]
                for item in entries
            ),
            "maximum_restore_ee_observation_error_m": max(
                item["explicit_restore_robot_eef_error_m"] for item in entries
            ),
            "allowed_maximum_restore_ee_observation_error_m": (
                robot_randomization.maximum_restore_ee_observation_error
            ),
            "allowed_maximum_restore_ee_drift_m": robot_randomization.maximum_restore_ee_drift,
            "minimum_pairwise_placement_signature_l2": minimum_distance,
            "state_array_sha256": array_hash(state_array),
        }
        LiberoRuntime.write_json(temporary / "manifest.json", manifest)
        LiberoRuntime.write_json(temporary / "validation_report.json", validation)
        temporary.replace(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        env.close()


def _build_randomized_state_bundle(
    *,
    suite_name: str,
    task_id: int,
    count: int,
    seed_start: int,
    wait_steps: int,
    validation_hold_steps: int,
    control_freq: int,
    maximum_stabilization_drift: float,
    max_attempts: int,
    libero_root: Path | None = None,
) -> RandomizedStateBundle:
    """Run the cache generator in a temporary materialization for workflows."""

    with tempfile.TemporaryDirectory(prefix="vla-mender-randomized-states.") as root:
        output = Path(root) / "bundle"
        manifest = generate_randomized_states(
            output=output,
            suite_name=suite_name,
            task_id=task_id,
            count=count,
            seed_start=seed_start,
            wait_steps=wait_steps,
            validation_hold_steps=validation_hold_steps,
            control_freq=control_freq,
            maximum_stabilization_drift=maximum_stabilization_drift,
            max_attempts=max_attempts,
            libero_root=libero_root,
        )
        states = np.load(output / "states.npy", allow_pickle=False).copy()
        validation = json.loads(
            (output / "validation_report.json").read_text(encoding="utf-8")
        )
    return RandomizedStateBundle(
        states=states,
        manifest=manifest,
        validation=validation,
    )
