"""Strict configuration and preflight checks for repaired-dataset post-training."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml

from .trainable_filter import valid_global_indices


def _unknown(section: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"unknown {section} keys: {sorted(extra)}")


def _resolve(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass(frozen=True)
class LearningRateConfig:
    warmup_steps: int = 200
    peak_lr: float = 2.0e-5
    decay_steps: int = 4_000
    decay_lr: float = 1.0e-6

    @classmethod
    def parse(cls, value: Mapping[str, Any] | None) -> "LearningRateConfig":
        data = dict(value or {})
        _unknown("learning_rate", data, set(cls.__dataclass_fields__))
        result = cls(
            warmup_steps=int(data.get("warmup_steps", cls.warmup_steps)),
            peak_lr=float(data.get("peak_lr", cls.peak_lr)),
            decay_steps=int(data.get("decay_steps", cls.decay_steps)),
            decay_lr=float(data.get("decay_lr", cls.decay_lr)),
        )
        if result.warmup_steps < 0 or result.decay_steps <= result.warmup_steps:
            raise ValueError("learning-rate decay_steps must be greater than warmup_steps >= 0")
        if result.peak_lr <= 0 or result.decay_lr < 0:
            raise ValueError("learning rates must be non-negative and peak_lr must be positive")
        return result


@dataclass(frozen=True)
class PostTrainingConfig:
    settings_path: Path
    openpi_source: Path
    openpi_environment: Path
    openpi_commit: str
    base_config: str
    run_name: str
    experiment_name: str
    project_name: str
    checkpoint_base_dir: Path
    initialization_checkpoint: Path
    normalization_asset_id: str
    dataset: Path
    sampling_mode: str
    trainable_index_manifest: Path | None
    gpus: tuple[int, ...]
    action_horizon: int
    batch_size: int
    num_workers: int
    num_train_steps: int
    log_interval: int
    save_interval: int
    keep_period: int | None
    seed: int
    extra_delta_transform: bool
    mask_zero_arm_action_loss: bool
    zero_arm_mask_mode: str
    zero_arm_action_dims: int
    zero_arm_position_threshold_m: float
    zero_arm_orientation_threshold_rad: float
    zero_arm_position_action_scale_m: float
    zero_arm_orientation_action_scale_rad: float
    zero_arm_gripper_change_eps: float
    zero_arm_gripper_state_change_threshold: float
    zero_arm_keep_chunk_start: bool
    learning_rate: LearningRateConfig
    resume: bool
    overwrite: bool
    wandb_enabled: bool
    pytorch_training_precision: str

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoint_base_dir / self.run_name / self.experiment_name


def load_training_config(path: str | Path) -> PostTrainingConfig:
    settings_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("training settings must be a YAML mapping")
    data = dict(raw)
    allowed = {
        "openpi_source",
        "openpi_environment",
        "openpi_commit",
        "base_config",
        "run_name",
        "experiment_name",
        "project_name",
        "checkpoint_base_dir",
        "initialization_checkpoint",
        "normalization_asset_id",
        "dataset",
        "sampling_mode",
        "trainable_index_manifest",
        "gpus",
        "action_horizon",
        "batch_size",
        "num_workers",
        "num_train_steps",
        "log_interval",
        "save_interval",
        "keep_period",
        "seed",
        "extra_delta_transform",
        "mask_zero_arm_action_loss",
        "zero_arm_mask_mode",
        "zero_arm_action_dims",
        "zero_arm_action_norm_threshold",
        "zero_arm_position_threshold_m",
        "zero_arm_orientation_threshold_rad",
        "zero_arm_position_action_scale_m",
        "zero_arm_orientation_action_scale_rad",
        "zero_arm_gripper_change_eps",
        "zero_arm_gripper_state_change_threshold",
        "zero_arm_keep_chunk_start",
        "learning_rate",
        "resume",
        "overwrite",
        "wandb_enabled",
        "pytorch_training_precision",
    }
    _unknown("training", data, allowed)
    required = {
        "openpi_source",
        "openpi_environment",
        "openpi_commit",
        "base_config",
        "run_name",
        "experiment_name",
        "checkpoint_base_dir",
        "initialization_checkpoint",
        "normalization_asset_id",
        "dataset",
        "gpus",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"missing training keys: {sorted(missing)}")
    base = settings_path.parent
    gpus = tuple(int(item) for item in data["gpus"])
    keep_period = data.get("keep_period", 500)
    sampling_mode = str(data.get("sampling_mode", "transition_aware"))
    manifest_value = data.get("trainable_index_manifest")
    mask_zero_arm_action_loss = bool(data.get("mask_zero_arm_action_loss", False))
    position_threshold = data.get("zero_arm_position_threshold_m", 0.002)
    orientation_threshold = data.get("zero_arm_orientation_threshold_rad", 0.02)
    if mask_zero_arm_action_loss and (position_threshold is None or orientation_threshold is None):
        raise ValueError(
            "zero-arm norm masking was removed; enabled masks require physical "
            "zero_arm_position_threshold_m and zero_arm_orientation_threshold_rad"
        )
    if position_threshold is None:
        position_threshold = 0.002
    if orientation_threshold is None:
        orientation_threshold = 0.02
    result = PostTrainingConfig(
        settings_path=settings_path,
        openpi_source=_resolve(base, data["openpi_source"], "openpi_source"),
        openpi_environment=_resolve(base, data["openpi_environment"], "openpi_environment"),
        openpi_commit=str(data["openpi_commit"]),
        base_config=str(data["base_config"]),
        run_name=str(data["run_name"]),
        experiment_name=str(data["experiment_name"]),
        project_name=str(data.get("project_name", "vla-mender")),
        checkpoint_base_dir=_resolve(base, data["checkpoint_base_dir"], "checkpoint_base_dir"),
        initialization_checkpoint=_resolve(
            base, data["initialization_checkpoint"], "initialization_checkpoint"
        ),
        normalization_asset_id=str(data["normalization_asset_id"]),
        dataset=_resolve(base, data["dataset"], "dataset"),
        sampling_mode=sampling_mode,
        trainable_index_manifest=(
            None
            if manifest_value is None
            else _resolve(base, manifest_value, "trainable_index_manifest")
        ),
        gpus=gpus,
        action_horizon=int(data.get("action_horizon", 50)),
        batch_size=int(data.get("batch_size", 32)),
        num_workers=int(data.get("num_workers", 2)),
        num_train_steps=int(data.get("num_train_steps", 4_000)),
        log_interval=int(data.get("log_interval", 100)),
        save_interval=int(data.get("save_interval", 500)),
        keep_period=None if keep_period is None else int(keep_period),
        seed=int(data.get("seed", 42)),
        extra_delta_transform=bool(data.get("extra_delta_transform", True)),
        mask_zero_arm_action_loss=mask_zero_arm_action_loss,
        zero_arm_mask_mode=str(data.get("zero_arm_mask_mode", "command")),
        zero_arm_action_dims=int(data.get("zero_arm_action_dims", 6)),
        zero_arm_position_threshold_m=float(position_threshold),
        zero_arm_orientation_threshold_rad=float(orientation_threshold),
        zero_arm_position_action_scale_m=float(data.get("zero_arm_position_action_scale_m", 0.05)),
        zero_arm_orientation_action_scale_rad=float(
            data.get("zero_arm_orientation_action_scale_rad", 0.5)
        ),
        zero_arm_gripper_change_eps=float(data.get("zero_arm_gripper_change_eps", 1.0e-4)),
        zero_arm_gripper_state_change_threshold=float(
            data.get("zero_arm_gripper_state_change_threshold", 5.0e-5)
        ),
        zero_arm_keep_chunk_start=bool(data.get("zero_arm_keep_chunk_start", True)),
        learning_rate=LearningRateConfig.parse(data.get("learning_rate")),
        resume=bool(data.get("resume", False)),
        overwrite=bool(data.get("overwrite", False)),
        wandb_enabled=bool(data.get("wandb_enabled", True)),
        pytorch_training_precision=str(data.get("pytorch_training_precision", "bfloat16")),
    )
    if not result.gpus or len(set(result.gpus)) != len(result.gpus) or min(result.gpus) < 0:
        raise ValueError("gpus must contain unique non-negative device indices")
    if result.action_horizon <= 0 or result.batch_size <= 0 or result.num_train_steps <= 0:
        raise ValueError("action_horizon, batch_size, and num_train_steps must be positive")
    if result.batch_size % len(result.gpus):
        raise ValueError("global batch_size must be divisible by the number of GPUs")
    if result.num_workers < 0 or result.log_interval <= 0 or result.save_interval <= 0:
        raise ValueError("num_workers must be non-negative; log/save intervals must be positive")
    if result.resume and result.overwrite:
        raise ValueError("resume and overwrite cannot both be true")
    if result.pytorch_training_precision not in {"bfloat16", "float32"}:
        raise ValueError("pytorch_training_precision must be bfloat16 or float32")
    if result.zero_arm_action_dims <= 0:
        raise ValueError("zero-arm action dimensions must be positive")
    if result.zero_arm_mask_mode not in {"command", "state_delta"}:
        raise ValueError("zero_arm_mask_mode must be command or state_delta")
    for label, value in {
        "zero_arm_position_action_scale_m": result.zero_arm_position_action_scale_m,
        "zero_arm_orientation_action_scale_rad": result.zero_arm_orientation_action_scale_rad,
        "zero_arm_gripper_state_change_threshold": result.zero_arm_gripper_state_change_threshold,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{label} must be positive")
    if result.zero_arm_gripper_change_eps < 0.0:
        raise ValueError("zero_arm_gripper_change_eps must be non-negative")
    if result.mask_zero_arm_action_loss:
        if result.zero_arm_action_dims < 6:
            raise ValueError("physical zero-arm masking requires at least 6 arm dimensions")
        if (
            result.zero_arm_position_threshold_m <= 0.0
            or result.zero_arm_orientation_threshold_rad <= 0.0
        ):
            raise ValueError("physical zero-arm thresholds must be positive")
    if result.sampling_mode not in {"transition_aware", "native"}:
        raise ValueError("sampling_mode must be transition_aware or native")
    if result.sampling_mode == "transition_aware" and result.trainable_index_manifest is None:
        raise ValueError("transition_aware sampling requires trainable_index_manifest")
    for label, value in {
        "openpi_commit": result.openpi_commit,
        "base_config": result.base_config,
        "run_name": result.run_name,
        "experiment_name": result.experiment_name,
        "normalization_asset_id": result.normalization_asset_id,
    }.items():
        if not value:
            raise ValueError(f"{label} must not be empty")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_training_inputs(
    config: PostTrainingConfig, *, check_checkpoint_state: bool = True
) -> dict[str, Any]:
    """Fail closed before torchrun starts or allocates a model."""

    if not (config.openpi_source / "src/openpi/training/config.py").is_file():
        raise FileNotFoundError(f"invalid OpenPI source tree: {config.openpi_source}")
    torchrun = config.openpi_environment / "bin/torchrun"
    python = config.openpi_environment / "bin/python"
    if not torchrun.is_file() or not os.access(torchrun, os.X_OK) or not python.is_file():
        raise FileNotFoundError(
            f"OpenPI environment lacks bin/python or bin/torchrun: {config.openpi_environment}"
        )
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.openpi_source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != config.openpi_commit:
        raise ValueError(
            f"OpenPI commit mismatch: configured {config.openpi_commit}, actual {actual_commit}"
        )
    model = config.initialization_checkpoint / "model.safetensors"
    norm_stats = (
        config.initialization_checkpoint / config.normalization_asset_id / "norm_stats.json"
    )
    if not model.is_file():
        raise FileNotFoundError(f"missing PyTorch initialization weights: {model}")
    if not norm_stats.is_file():
        raise FileNotFoundError(f"missing normalization statistics: {norm_stats}")
    validation_path = config.dataset / "meta/validation_report.json"
    info_path = config.dataset / "meta/info.json"
    if not validation_path.is_file() or not info_path.is_file():
        raise FileNotFoundError(f"dataset is incomplete or unvalidated: {config.dataset}")
    validation = _read_json(validation_path)
    if validation.get("valid") is not True:
        raise ValueError(f"dataset validation did not pass: {validation_path}")
    info = _read_json(info_path)
    if config.sampling_mode == "native":
        # Match OpenPI's native LeRobot behavior: every dataset row is a sample
        # start, chunks may cross source-phase boundaries, and short tails use
        # LeRobot's standard padding.  The manifest is deliberately ignored.
        valid_start_count = int(info.get("total_frames", 0))
        if valid_start_count <= 0:
            raise ValueError(f"dataset has no native sample starts: {info_path}")
    else:
        assert config.trainable_index_manifest is not None
        if (
            config.trainable_index_manifest.resolve()
            != (config.dataset / "meta/trainable_index_manifest.json").resolve()
        ):
            raise ValueError("trainable_index_manifest must belong to the configured dataset")
        trainable = _read_json(config.trainable_index_manifest)
        if int(trainable.get("schema_version", 0)) != 1 or not isinstance(
            trainable.get("frames"), list
        ):
            raise ValueError(f"invalid trainable-index manifest: {config.trainable_index_manifest}")
        if int(trainable.get("action_horizon", -1)) != config.action_horizon:
            raise ValueError("training action_horizon differs from the dataset manifest")
        approved = valid_global_indices(trainable, config.action_horizon)
        valid_start_count = int(trainable.get("valid_start_count", 0))
        if valid_start_count != len(approved):
            raise ValueError(
                f"trainable valid_start_count is {valid_start_count}, recomputed {len(approved)}"
            )
    local_batch_size = config.batch_size // len(config.gpus)
    if valid_start_count < local_batch_size:
        raise ValueError(
            f"valid starts ({valid_start_count}) are fewer than per-GPU batch ({local_batch_size})"
        )
    if check_checkpoint_state and config.resume:
        if not config.checkpoint_dir.is_dir() or not any(
            child.is_dir() and child.name.isdigit() for child in config.checkpoint_dir.iterdir()
        ):
            raise FileNotFoundError(
                f"resume requested but no numeric checkpoint exists: {config.checkpoint_dir}"
            )
    elif check_checkpoint_state and config.checkpoint_dir.exists() and not config.overwrite:
        raise FileExistsError(
            f"checkpoint directory already exists; choose resume or explicit overwrite: {config.checkpoint_dir}"
        )
    return {
        "valid": True,
        "dataset": str(config.dataset),
        "sampling_mode": config.sampling_mode,
        "valid_start_count": valid_start_count,
        "openpi_commit": actual_commit,
        "initialization_checkpoint": str(config.initialization_checkpoint),
        "checkpoint_dir": str(config.checkpoint_dir),
        "world_size": len(config.gpus),
        "global_batch_size": config.batch_size,
        "local_batch_size": local_batch_size,
        "mask_zero_arm_action_loss": config.mask_zero_arm_action_loss,
        "zero_arm_mask_mode": config.zero_arm_mask_mode,
    }
