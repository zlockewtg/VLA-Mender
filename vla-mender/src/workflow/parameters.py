"""Experiment contract and the small set of parameters that affect outcomes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml

ControlSpace = Literal["osc", "joint"]
ResetDynamics = Literal["preserve_full_state", "quiescent_osc"]
StateProvider = Literal["official", "randomized_bddl", "state_manifest"]
RuntimeBackend = Literal["openpi"]


@dataclasses.dataclass(frozen=True)
class TaskSettings:
    suite: str
    task_id: int
    checkpoint: Path
    policy_config: str = "pi0_libero"
    task_description: str | None = None


@dataclasses.dataclass(frozen=True)
class InitialStateSettings:
    provider: StateProvider = "official"
    count: int = 50
    seed_start: int = 100_000
    state_manifest: Path | None = None


@dataclasses.dataclass(frozen=True)
class RolloutSettings:
    control_frequency_hz: int = 20
    max_steps: int = 300
    policy_seed: int = 7
    gpus: tuple[int, ...] = (0, 1, 2, 3)
    workers_per_gpu: int = 1
    action_chunk: int = 5
    inference_steps: int = 5
    num_steps_wait: int = 10
    binary_gripper: bool = False
    gripper_hysteresis_threshold: float = 0.2


@dataclasses.dataclass(frozen=True)
class ControllerSettings:
    source_control_space: ControlSpace = "osc"
    target_control_space: ControlSpace = "osc"


@dataclasses.dataclass(frozen=True)
class ResetSettings:
    frames_per_failure: int = 3
    frame_stride: int = 5
    dynamics: ResetDynamics = "preserve_full_state"


@dataclasses.dataclass(frozen=True)
class BackendSettings:
    """Runtime selection for the isolated, pinned OpenPI environment."""

    name: RuntimeBackend = "openpi"
    openpi_environment: Path | None = None
    openpi_source: Path | None = None
    openpi_commit: str | None = None
    openpi_norm_stats: Path | None = None


@dataclasses.dataclass(frozen=True)
class ExperimentSettings:
    """Resolved settings.

    Only fields that materially change an evaluation or reset identity are
    configurable.  Rendering, tolerance, atomic-write and retry defaults are
    deliberately private implementation constants in the runtime modules.
    """

    task: TaskSettings
    initial_states: InitialStateSettings = dataclasses.field(default_factory=InitialStateSettings)
    rollout: RolloutSettings = dataclasses.field(default_factory=RolloutSettings)
    controller: ControllerSettings = dataclasses.field(default_factory=ControllerSettings)
    reset: ResetSettings = dataclasses.field(default_factory=ResetSettings)
    backend: BackendSettings = dataclasses.field(default_factory=BackendSettings)

    def validate(self) -> None:
        if self.task.task_id < 0:
            raise ValueError("task.task_id must be non-negative")
        if not self.task.suite:
            raise ValueError("task.suite must not be empty")
        if self.initial_states.provider not in {"official", "randomized_bddl", "state_manifest"}:
            raise ValueError(f"unsupported initial state provider: {self.initial_states.provider}")
        if self.controller.source_control_space not in {"osc", "joint"}:
            raise ValueError(f"unsupported source control space: {self.controller.source_control_space}")
        if self.controller.target_control_space not in {"osc", "joint"}:
            raise ValueError(f"unsupported target control space: {self.controller.target_control_space}")
        if self.reset.dynamics not in {"preserve_full_state", "quiescent_osc"}:
            raise ValueError(f"unsupported reset dynamics: {self.reset.dynamics}")
        if self.initial_states.count <= 0:
            raise ValueError("initial_states.count must be positive")
        if self.rollout.control_frequency_hz <= 0 or self.rollout.max_steps <= 0:
            raise ValueError("rollout frequency and max_steps must be positive")
        if not self.rollout.gpus or any(gpu < 0 for gpu in self.rollout.gpus):
            raise ValueError("rollout.gpus must contain at least one non-negative GPU id")
        if self.rollout.workers_per_gpu <= 0:
            raise ValueError("rollout.workers_per_gpu must be positive")
        if self.rollout.action_chunk <= 0 or self.rollout.inference_steps <= 0:
            raise ValueError("rollout action_chunk and inference_steps must be positive")
        if self.rollout.num_steps_wait < 0:
            raise ValueError("rollout.num_steps_wait must be non-negative")
        if not 0.0 <= self.rollout.gripper_hysteresis_threshold <= 1.0:
            raise ValueError("rollout.gripper_hysteresis_threshold must be in [0, 1]")
        if self.reset.frames_per_failure <= 0 or self.reset.frame_stride <= 0:
            raise ValueError("reset frames_per_failure and frame_stride must be positive")
        if self.reset.dynamics == "quiescent_osc" and self.controller.target_control_space != "osc":
            raise ValueError("quiescent_osc reset dynamics requires target_control_space=osc")
        if self.backend.name != "openpi":
            raise ValueError(f"unsupported runtime backend: {self.backend.name}")
        if self.backend.openpi_commit is not None:
            commit = self.backend.openpi_commit.strip()
            if len(commit) != 40 or any(char not in "0123456789abcdefABCDEF" for char in commit):
                raise ValueError("backend.openpi_commit must be a 40-character git SHA")
        if self.initial_states.provider == "state_manifest":
            if self.initial_states.state_manifest is None:
                raise ValueError("state_manifest is required when provider=state_manifest")

    def as_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["task"]["checkpoint"] = str(self.task.checkpoint)
        if self.initial_states.state_manifest is not None:
            value["initial_states"]["state_manifest"] = str(self.initial_states.state_manifest)
        value["rollout"]["gpus"] = list(self.rollout.gpus)
        for key in ("openpi_environment", "openpi_source", "openpi_norm_stats"):
            if value["backend"][key] is not None:
                value["backend"][key] = str(value["backend"][key])
        return value

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def _tuple_gpus(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(int(item) for item in value)


def _resolve_optional_path(base: Path, value: Any) -> Path | None:
    if value is None:
        return None
    candidate = Path(str(value)).expanduser()
    return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def load_settings(path: str | Path) -> ExperimentSettings:
    """Load YAML and fail before any simulator or model is initialized."""

    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"experiment settings must be a mapping: {source}")
    task_raw = dict(raw.get("task", {}))
    initial_raw = dict(raw.get("initial_states", {}))
    rollout_raw = dict(raw.get("rollout", {}))
    controller_raw = dict(raw.get("controller", {}))
    reset_raw = dict(raw.get("reset", {}))
    backend_raw = dict(raw.get("backend", {}))
    if "checkpoint" not in task_raw:
        raise ValueError("task.checkpoint is required")
    if initial_raw.get("state_manifest") is not None:
        initial_raw["state_manifest"] = (source.parent / str(initial_raw["state_manifest"])).resolve()
    settings = ExperimentSettings(
        task=TaskSettings(
            suite=str(task_raw.get("suite", "libero_goal")),
            task_id=int(task_raw.get("task_id", 0)),
            checkpoint=(source.parent / str(task_raw["checkpoint"])).resolve()
            if not Path(str(task_raw["checkpoint"])).is_absolute()
            else Path(str(task_raw["checkpoint"])).resolve(),
            policy_config=str(task_raw.get("policy_config", "pi0_libero")),
            task_description=task_raw.get("task_description"),
        ),
        initial_states=InitialStateSettings(
            provider=str(initial_raw.get("provider", "official")),
            count=int(initial_raw.get("count", 50)),
            seed_start=int(initial_raw.get("seed_start", 100_000)),
            state_manifest=initial_raw.get("state_manifest"),
        ),
        rollout=RolloutSettings(
            control_frequency_hz=int(rollout_raw.get("control_frequency_hz", 20)),
            max_steps=int(rollout_raw.get("max_steps", 300)),
            policy_seed=int(rollout_raw.get("policy_seed", 7)),
            gpus=_tuple_gpus(rollout_raw.get("gpus", (0, 1, 2, 3))),
            workers_per_gpu=int(rollout_raw.get("workers_per_gpu", 1)),
            action_chunk=int(rollout_raw.get("action_chunk", 5)),
            inference_steps=int(rollout_raw.get("inference_steps", 5)),
            num_steps_wait=int(rollout_raw.get("num_steps_wait", 10)),
            binary_gripper=bool(rollout_raw.get("binary_gripper", False)),
            gripper_hysteresis_threshold=float(
                rollout_raw.get("gripper_hysteresis_threshold", 0.2)
            ),
        ),
        controller=ControllerSettings(
            source_control_space=str(controller_raw.get("source_control_space", "osc")),
            target_control_space=str(controller_raw.get("target_control_space", "osc")),
        ),
        reset=ResetSettings(
            frames_per_failure=int(reset_raw.get("frames_per_failure", 3)),
            frame_stride=int(reset_raw.get("frame_stride", 5)),
            dynamics=str(reset_raw.get("dynamics", "preserve_full_state")),
        ),
        backend=BackendSettings(
            name=str(backend_raw.get("name", "openpi")),
            openpi_environment=_resolve_optional_path(source.parent, backend_raw.get("openpi_environment")),
            openpi_source=_resolve_optional_path(source.parent, backend_raw.get("openpi_source")),
            openpi_commit=(str(backend_raw["openpi_commit"]).strip() if backend_raw.get("openpi_commit") else None),
            openpi_norm_stats=_resolve_optional_path(source.parent, backend_raw.get("openpi_norm_stats")),
        ),
    )
    settings.validate()
    return settings


def write_resolved_settings(settings: ExperimentSettings, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(settings.as_dict(), sort_keys=False), encoding="utf-8")
