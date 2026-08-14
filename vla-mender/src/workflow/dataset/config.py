"""Configuration contract for task-agnostic prefix-plus-repair builds."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_SIGNATURE_FIELDS = (
    "nq",
    "nv",
    "na",
    "state_width",
    "model_names_sha256",
    "body_joint_names_sha256",
    "geom_names_sha256",
    "model_numeric_sha256",
)


def _unknown(section: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"unknown {section} keys: {sorted(extra)}")


@dataclass(frozen=True)
class ColumnsConfig:
    state: str = "state"
    action: str = "actions"
    timestamp: str = "timestamp"
    frame_index: str = "frame_index"
    episode_index: str = "episode_index"
    global_index: str = "index"
    task_index: str = "task_index"

    @classmethod
    def parse(cls, value: Mapping[str, Any] | None) -> "ColumnsConfig":
        data = dict(value or {})
        _unknown("columns", data, set(cls.__dataclass_fields__))
        return cls(**{key: str(item) for key, item in data.items()})


@dataclass(frozen=True)
class RepairColumnsConfig:
    state: str = "observation.state"
    action: str = "action"
    action_valid: str = "action_is_valid"
    reward: str | None = "next.reward"
    done: str | None = "next.done"

    @classmethod
    def parse(cls, value: Mapping[str, Any] | None) -> "RepairColumnsConfig":
        data = dict(value or {})
        _unknown("repair_columns", data, set(cls.__dataclass_fields__))
        return cls(**data)


@dataclass(frozen=True)
class CameraConfig:
    prefix_column: str
    repair_flip_horizontal: bool = False
    width: int = 256
    height: int = 256

    @classmethod
    def parse(cls, output_column: str, value: Mapping[str, Any]) -> "CameraConfig":
        data = dict(value)
        _unknown(
            f"cameras.{output_column}",
            data,
            {"prefix_column", "repair_flip_horizontal", "width", "height"},
        )
        return cls(
            prefix_column=str(data.get("prefix_column", output_column)),
            repair_flip_horizontal=bool(data.get("repair_flip_horizontal", False)),
            width=int(data.get("width", 256)),
            height=int(data.get("height", 256)),
        )


@dataclass(frozen=True)
class ContinuityConfig:
    require_simulator_evidence: bool = True
    simulator_state_tolerance: float = 1e-12
    splice_state_tolerance: float = 5e-3
    signature_fields: tuple[str, ...] = DEFAULT_SIGNATURE_FIELDS
    max_flow_median_px: float = 0.5
    max_flow_p90_px: float = 1.0

    @classmethod
    def parse(cls, value: Mapping[str, Any] | None) -> "ContinuityConfig":
        data = dict(value or {})
        _unknown("continuity", data, set(cls.__dataclass_fields__))
        if "signature_fields" in data:
            data["signature_fields"] = tuple(str(item) for item in data["signature_fields"])
        result = cls(**data)
        if result.simulator_state_tolerance < 0 or result.splice_state_tolerance < 0:
            raise ValueError("continuity tolerances must be non-negative")
        if result.max_flow_median_px < 0 or result.max_flow_p90_px < 0:
            raise ValueError("flow thresholds must be non-negative")
        return result


@dataclass(frozen=True)
class ActionConfig:
    state_dim: int
    action_dim: int
    gripper_index: int | None = None
    gripper_threshold: float = 0.0
    gripper_low: float = -1.0
    gripper_high: float = 1.0
    maximum_absolute_value: float = 1.000001

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> "ActionConfig":
        data = dict(value)
        _unknown("action", data, set(cls.__dataclass_fields__))
        if "state_dim" not in data or "action_dim" not in data:
            raise ValueError("action.state_dim and action.action_dim are required")
        result = cls(**data)
        if result.state_dim <= 0 or result.action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if result.gripper_index is not None and not (
            -result.action_dim <= result.gripper_index < result.action_dim
        ):
            raise ValueError("gripper_index is outside action_dim")
        return result


@dataclass(frozen=True)
class DatasetBuildConfig:
    config_path: Path
    output: Path
    reference_dataset: Path
    episodes_manifest: Path
    dataset_source: str
    fps: int
    action_horizon: int
    pre_guard_frames: int
    post_guard_frames: int
    columns: ColumnsConfig
    repair_columns: RepairColumnsConfig
    cameras: dict[str, CameraConfig]
    action: ActionConfig
    continuity: ContinuityConfig = field(default_factory=ContinuityConfig)
    task_catalog: Path | None = None
    require_terminal_success: bool = True
    provenance_files: tuple[Path, ...] = ()

    def validate(self) -> None:
        if self.fps <= 0 or self.action_horizon <= 0:
            raise ValueError("fps and action_horizon must be positive")
        if self.pre_guard_frames < 0 or self.post_guard_frames < 0:
            raise ValueError("guard frame counts must be non-negative")
        if not self.cameras:
            raise ValueError("at least one output camera is required")
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError("duplicate output camera columns")


def _resolve(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: str | Path) -> DatasetBuildConfig:
    """Load a strict YAML configuration and resolve relative paths beside it."""

    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("dataset config must be a YAML mapping")
    data = dict(raw)
    allowed = {
        "output",
        "reference_dataset",
        "episodes_manifest",
        "dataset_source",
        "fps",
        "action_horizon",
        "pre_guard_frames",
        "post_guard_frames",
        "columns",
        "repair_columns",
        "cameras",
        "action",
        "continuity",
        "task_catalog",
        "require_terminal_success",
        "provenance_files",
    }
    _unknown("top-level", data, allowed)
    required = {"output", "reference_dataset", "episodes_manifest", "cameras", "action"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"missing dataset config keys: {sorted(missing)}")
    base = config_path.parent
    cameras_raw = data["cameras"]
    if not isinstance(cameras_raw, Mapping):
        raise ValueError("cameras must map output columns to camera settings")
    result = DatasetBuildConfig(
        config_path=config_path,
        output=_resolve(base, data["output"]),  # type: ignore[arg-type]
        reference_dataset=_resolve(base, data["reference_dataset"]),  # type: ignore[arg-type]
        episodes_manifest=_resolve(base, data["episodes_manifest"]),  # type: ignore[arg-type]
        dataset_source=str(data.get("dataset_source", config_path.stem)),
        fps=int(data.get("fps", 20)),
        action_horizon=int(data.get("action_horizon", 50)),
        pre_guard_frames=int(data.get("pre_guard_frames", 20)),
        post_guard_frames=int(data.get("post_guard_frames", 5)),
        columns=ColumnsConfig.parse(data.get("columns")),
        repair_columns=RepairColumnsConfig.parse(data.get("repair_columns")),
        cameras={
            str(name): CameraConfig.parse(str(name), settings)
            for name, settings in cameras_raw.items()
        },
        action=ActionConfig.parse(data["action"]),
        continuity=ContinuityConfig.parse(data.get("continuity")),
        task_catalog=_resolve(base, data.get("task_catalog")),
        require_terminal_success=bool(data.get("require_terminal_success", True)),
        provenance_files=tuple(
            _resolve(base, item) for item in data.get("provenance_files", [])
        ),  # type: ignore[arg-type]
    )
    result.validate()
    return result
