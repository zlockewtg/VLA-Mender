#!/usr/bin/env python3
"""Run configurable single- or multi-task LIBERO evaluation campaigns."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RUNTIME_SCRIPT = SCRIPT_DIR / "_runtime.py"
ALL_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)
SUITE_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}
STATE_PROVIDERS = ("official", "randomized_bddl", "manifest")
OFFICIAL_STATE_COUNT = 50
COMPILE_MODES = (
    "none",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)

DEFAULT_EVALUATION: dict[str, Any] = {
    "seed": 7,
    "policy_seed_offset": 0,
    "trials_per_state": 1,
    "control_freq": 20,
    "max_steps": 300,
    "num_steps_wait": 10,
    "action_chunk": 5,
    "inference_steps": 5,
    "binary_gripper": False,
    "gripper_hysteresis_threshold": 0.2,
    "compile_mode": "none",
}
DEFAULT_INITIAL_STATES: dict[str, Any] = {
    "provider": None,
    "count": 50,
    "indices": None,
    "manifest": None,
    "cache_root": "/mnt/public/tgy/data/libero_randomized_scenes",
    "seed_start": 100_000,
    "wait_steps": 10,
    "validation_hold_steps": 5,
    "maximum_stabilization_drift": 0.05,
    "max_attempts": None,
}
DEFAULT_RESOURCES: dict[str, Any] = {"gpus": [0], "num_envs": None}
DEFAULT_ARTIFACTS: dict[str, Any] = {
    "mode": "lerobot",
    "training_schema_only": True,
    "save_images": False,
    "image_writer_threads": 8,
}


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    key: str
    suite: str
    task_id: int
    checkpoint: Path
    config_name: str
    openpi_source: Path
    openpi_commit: str | None
    openpi_norm_stats: Path | None
    initial_states: dict[str, Any]
    evaluation: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Campaign:
    config_path: Path
    output: Path
    python: Path
    libero_config_path: Path | None
    tasks: tuple[TaskSpec, ...]
    resources: dict[str, Any]
    artifacts: dict[str, Any]
    repo_id_prefix: str
    resume: bool
    overwrite: bool
    dry_run: bool


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"config file does not exist: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ValueError("campaign config must be a YAML mapping")
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError("campaign config must use schema_version: 1")
    allowed = {
        "schema_version",
        "checkpoint",
        "config_name",
        "openpi_source",
        "openpi_commit",
        "openpi_norm_stats",
        "python",
        "libero_config_path",
        "output_parent",
        "tasks",
        "initial_states",
        "evaluation",
        "resources",
        "artifacts",
    }
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError(f"unknown campaign config keys: {unknown}")
    return document


def _path(base: Path, value: Any | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(str(value)).expanduser()
    return (
        (base / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.") or "task"


def _parse_task_token(value: str) -> dict[str, Any]:
    try:
        suite, task_id = value.rsplit(":", 1)
        parsed_id = int(task_id)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "task must use SUITE:TASK_ID, for example libero_goal:0"
        ) from exc
    if suite not in ALL_SUITES or not 0 <= parsed_id < SUITE_TASK_COUNTS[suite]:
        raise argparse.ArgumentTypeError(f"invalid LIBERO task: {value}")
    return {"suite": suite, "task_id": parsed_id}


def _parse_csv_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers: {value}"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", action="append", type=_parse_task_token)
    parser.add_argument("--suite", choices=ALL_SUITES)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config-name")
    parser.add_argument("--openpi-source", type=Path)
    parser.add_argument("--openpi-commit")
    parser.add_argument("--openpi-norm-stats", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--libero-config-path", type=Path)
    parser.add_argument("--state-provider", choices=STATE_PROVIDERS)
    parser.add_argument("--state-count", type=int)
    parser.add_argument("--initial-state-indices", type=_parse_csv_ints)
    parser.add_argument("--state-manifest", type=Path)
    parser.add_argument("--scene-cache", type=Path)
    parser.add_argument("--scene-seed-start", type=int)
    parser.add_argument("--scene-wait-steps", type=int)
    parser.add_argument("--scene-validation-hold-steps", type=int)
    parser.add_argument("--scene-max-attempts", type=int)
    parser.add_argument("--maximum-stabilization-drift", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--policy-seed-offset", type=int)
    parser.add_argument("--trials-per-state", type=int)
    parser.add_argument("--control-freq", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-steps-wait", type=int)
    parser.add_argument("--action-chunk", type=int)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument(
        "--binary-gripper", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--gripper-hysteresis-threshold", type=float)
    parser.add_argument("--compile-mode", choices=COMPILE_MODES)
    parser.add_argument("--gpus", type=_parse_csv_ints)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--artifact-mode", choices=("lerobot", "videos_only"))
    parser.add_argument(
        "--training-schema-only", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--save-images", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--image-writer-threads", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-id-prefix", default="local/libero-eval")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _merge_section(
    defaults: dict[str, Any], document: Any, name: str
) -> dict[str, Any]:
    if document is None:
        document = {}
    if not isinstance(document, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = sorted(set(document) - set(defaults))
    if unknown:
        raise ValueError(f"unknown {name} keys: {unknown}")
    return {**copy.deepcopy(defaults), **document}


def _apply_cli(document: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    value = copy.deepcopy(document)
    scalar_overrides = {
        "checkpoint": args.checkpoint,
        "config_name": args.config_name,
        "openpi_source": args.openpi_source,
        "openpi_commit": args.openpi_commit,
        "openpi_norm_stats": args.openpi_norm_stats,
        "python": args.python,
        "libero_config_path": args.libero_config_path,
    }
    for key, override in scalar_overrides.items():
        if override is not None:
            value[key] = override

    initial = dict(value.get("initial_states") or {})
    initial_cli = {
        "provider": args.state_provider,
        "count": args.state_count,
        "indices": args.initial_state_indices,
        "manifest": args.state_manifest,
        "cache_root": args.scene_cache,
        "seed_start": args.scene_seed_start,
        "wait_steps": args.scene_wait_steps,
        "validation_hold_steps": args.scene_validation_hold_steps,
        "max_attempts": args.scene_max_attempts,
        "maximum_stabilization_drift": args.maximum_stabilization_drift,
    }
    for key, override in initial_cli.items():
        if override is not None:
            initial[key] = override
    if args.state_count is not None:
        initial["indices"] = None
    if args.initial_state_indices is not None:
        initial["count"] = None
    value["initial_states"] = initial

    evaluation = dict(value.get("evaluation") or {})
    evaluation_cli = {
        "seed": args.seed,
        "policy_seed_offset": args.policy_seed_offset,
        "trials_per_state": args.trials_per_state,
        "control_freq": args.control_freq,
        "max_steps": args.max_steps,
        "num_steps_wait": args.num_steps_wait,
        "action_chunk": args.action_chunk,
        "inference_steps": args.num_inference_steps,
        "binary_gripper": args.binary_gripper,
        "gripper_hysteresis_threshold": args.gripper_hysteresis_threshold,
        "compile_mode": args.compile_mode,
    }
    for key, override in evaluation_cli.items():
        if override is not None:
            evaluation[key] = override
    value["evaluation"] = evaluation

    resources = dict(value.get("resources") or {})
    if args.gpus is not None:
        resources["gpus"] = args.gpus
    if args.num_envs is not None:
        resources["num_envs"] = args.num_envs
    value["resources"] = resources

    artifacts = dict(value.get("artifacts") or {})
    artifact_cli = {
        "mode": args.artifact_mode,
        "training_schema_only": args.training_schema_only,
        "save_images": args.save_images,
        "image_writer_threads": args.image_writer_threads,
    }
    for key, override in artifact_cli.items():
        if override is not None:
            artifacts[key] = override
    value["artifacts"] = artifacts

    if args.task and (args.suite is not None or args.task_id is not None):
        raise ValueError("--task cannot be combined with --suite/--task-id")
    if args.task:
        value["tasks"] = args.task
    elif args.suite is not None or args.task_id is not None:
        if args.suite is None or args.task_id is None:
            raise ValueError("--suite and --task-id must be supplied together")
        value["tasks"] = [{"suite": args.suite, "task_id": args.task_id}]

    # CLI task/evaluation/state values are outcome settings, so they must also
    # replace task-local YAML overrides rather than merely changing the global
    # defaults inherited by a task.
    tasks = value.get("tasks")
    if isinstance(tasks, list):
        task_scalar_overrides = {
            key: override
            for key, override in scalar_overrides.items()
            if key not in {"python", "libero_config_path"} and override is not None
        }
        task_initial_overrides = {
            key: override
            for key, override in initial_cli.items()
            if override is not None
        }
        task_evaluation_overrides = {
            key: override
            for key, override in evaluation_cli.items()
            if override is not None
        }
        for raw_task in tasks:
            if not isinstance(raw_task, dict):
                continue
            raw_task.update(task_scalar_overrides)
            if task_initial_overrides:
                task_initial = dict(raw_task.get("initial_states") or {})
                task_initial.update(task_initial_overrides)
                if args.state_count is not None:
                    task_initial["indices"] = None
                if args.initial_state_indices is not None:
                    task_initial["count"] = None
                raw_task["initial_states"] = task_initial
            if task_evaluation_overrides:
                task_evaluation = dict(raw_task.get("evaluation") or {})
                task_evaluation.update(task_evaluation_overrides)
                raw_task["evaluation"] = task_evaluation
            if args.max_steps is not None:
                raw_task["max_steps"] = args.max_steps
    return value


def _validate_positive(
    section: dict[str, Any], names: tuple[str, ...], prefix: str
) -> None:
    for name in names:
        if int(section[name]) <= 0:
            raise ValueError(f"{prefix}.{name} must be positive")


def _normalize_task(
    raw: dict[str, Any],
    *,
    global_config: dict[str, Any],
    base: Path,
) -> TaskSpec:
    if not isinstance(raw, dict):
        raise ValueError("each tasks[] entry must be a mapping")
    allowed = {
        "key",
        "suite",
        "task_id",
        "checkpoint",
        "config_name",
        "openpi_source",
        "openpi_commit",
        "openpi_norm_stats",
        "initial_states",
        "evaluation",
        "max_steps",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown task keys: {unknown}")
    suite = str(raw.get("suite", ""))
    task_id = int(raw.get("task_id", -1))
    if suite not in ALL_SUITES or not 0 <= task_id < SUITE_TASK_COUNTS[suite]:
        raise ValueError(f"invalid task identity: {suite}:{task_id}")
    key = _slug(str(raw.get("key") or f"{suite}-task{task_id:03d}"))

    checkpoint = _path(base, raw.get("checkpoint", global_config.get("checkpoint")))
    openpi_source = _path(
        base, raw.get("openpi_source", global_config.get("openpi_source"))
    )
    if checkpoint is None or openpi_source is None:
        raise ValueError(f"task {key} requires checkpoint and openpi_source")
    config_name = str(
        raw.get("config_name", global_config.get("config_name", "pi0_libero"))
    )
    norm_stats = _path(
        base, raw.get("openpi_norm_stats", global_config.get("openpi_norm_stats"))
    )

    initial = _merge_section(
        DEFAULT_INITIAL_STATES,
        {
            **dict(global_config.get("initial_states") or {}),
            **dict(raw.get("initial_states") or {}),
        },
        f"tasks[{key}].initial_states",
    )
    evaluation_raw = {
        **dict(global_config.get("evaluation") or {}),
        **dict(raw.get("evaluation") or {}),
    }
    if "max_steps" in raw:
        evaluation_raw["max_steps"] = raw["max_steps"]
    evaluation = _merge_section(
        DEFAULT_EVALUATION, evaluation_raw, f"tasks[{key}].evaluation"
    )

    provider = initial.get("provider")
    if provider not in STATE_PROVIDERS:
        raise ValueError(
            f"task {key}: initial_states.provider must be explicitly selected"
        )
    count, indices = initial.get("count"), initial.get("indices")
    if count is not None and indices is not None:
        raise ValueError(
            f"task {key}: initial_states.count and indices are mutually exclusive"
        )
    if count is None and indices is None:
        raise ValueError(f"task {key}: initial_states requires count or indices")
    if count is not None and int(count) <= 0:
        raise ValueError(f"task {key}: initial_states.count must be positive")
    if indices is not None:
        indices = [int(item) for item in indices]
        if (
            not indices
            or any(item < 0 for item in indices)
            or len(indices) != len(set(indices))
        ):
            raise ValueError(
                f"task {key}: initial_states.indices must be unique non-negative integers"
            )
        initial["indices"] = indices
    if provider == "randomized_bddl" and indices is not None:
        raise ValueError(
            f"task {key}: randomized_bddl does not support explicit indices"
        )
    if provider == "official" and (
        (count is not None and int(count) > OFFICIAL_STATE_COUNT)
        or (indices is not None and max(indices) >= OFFICIAL_STATE_COUNT)
    ):
        raise ValueError(
            f"task {key}: official initial-state indices must be in "
            f"[0, {OFFICIAL_STATE_COUNT - 1}]"
        )
    manifest = _path(base, initial.get("manifest"))
    cache_root = _path(base, initial.get("cache_root"))
    initial["manifest"] = str(manifest) if manifest else None
    initial["cache_root"] = str(cache_root) if cache_root else None
    if provider == "manifest" and manifest is None:
        raise ValueError(
            f"task {key}: provider=manifest requires initial_states.manifest"
        )
    if provider == "randomized_bddl" and cache_root is None:
        raise ValueError(f"task {key}: provider=randomized_bddl requires cache_root")
    if float(initial["maximum_stabilization_drift"]) <= 0:
        raise ValueError(f"task {key}: maximum_stabilization_drift must be positive")
    if int(initial["wait_steps"]) < 0 or int(initial["validation_hold_steps"]) < 0:
        raise ValueError(f"task {key}: scene wait steps must be non-negative")
    if initial["max_attempts"] is not None and int(initial["max_attempts"]) < int(
        count or 0
    ):
        raise ValueError(f"task {key}: max_attempts must be at least count")

    _validate_positive(
        evaluation,
        (
            "trials_per_state",
            "control_freq",
            "max_steps",
            "action_chunk",
            "inference_steps",
        ),
        f"tasks[{key}].evaluation",
    )
    if int(evaluation["num_steps_wait"]) < 0:
        raise ValueError(f"task {key}: num_steps_wait must be non-negative")
    if not 0 <= float(evaluation["gripper_hysteresis_threshold"]) <= 1:
        raise ValueError(f"task {key}: invalid gripper_hysteresis_threshold")
    if evaluation["compile_mode"] not in COMPILE_MODES:
        raise ValueError(f"task {key}: invalid compile_mode")

    return TaskSpec(
        key=key,
        suite=suite,
        task_id=task_id,
        checkpoint=checkpoint,
        config_name=config_name,
        openpi_source=openpi_source,
        openpi_commit=raw.get("openpi_commit", global_config.get("openpi_commit")),
        openpi_norm_stats=norm_stats,
        initial_states=initial,
        evaluation=evaluation,
    )


def resolve_campaign(args: argparse.Namespace) -> Campaign:
    config_path = args.config.expanduser().resolve()
    base = config_path.parent
    document = _apply_cli(_read_yaml(config_path), args)
    raw_tasks = document.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("campaign requires a non-empty tasks[] list or --task")
    tasks = tuple(
        _normalize_task(raw, global_config=document, base=base) for raw in raw_tasks
    )
    keys = [task.key for task in tasks]
    if len(keys) != len(set(keys)):
        raise ValueError(f"task keys must be unique: {keys}")

    resources = _merge_section(
        DEFAULT_RESOURCES, document.get("resources"), "resources"
    )
    resources["gpus"] = [int(item) for item in resources["gpus"]]
    if not resources["gpus"] or any(item < 0 for item in resources["gpus"]):
        raise ValueError("resources.gpus must contain non-negative GPU IDs")
    if len(resources["gpus"]) != len(set(resources["gpus"])):
        raise ValueError("resources.gpus must not contain duplicates")
    resources["num_envs"] = int(resources["num_envs"] or len(resources["gpus"]))
    if resources["num_envs"] <= 0:
        raise ValueError("resources.num_envs must be positive")

    artifacts = _merge_section(
        DEFAULT_ARTIFACTS, document.get("artifacts"), "artifacts"
    )
    if artifacts["mode"] not in {"lerobot", "videos_only"}:
        raise ValueError("artifacts.mode must be lerobot or videos_only")
    if artifacts["mode"] == "videos_only" and artifacts["save_images"]:
        raise ValueError("videos_only and save_images are mutually exclusive")
    if int(artifacts["image_writer_threads"]) <= 0:
        raise ValueError("artifacts.image_writer_threads must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.resume and args.output is None:
        raise ValueError("--resume requires an explicit --output")
    if args.resume and artifacts["mode"] == "videos_only":
        raise ValueError("videos_only does not support resume")

    python = _path(base, document.get("python", "/opt/venv/openpi/bin/python"))
    libero_path = _path(base, document.get("libero_config_path"))
    assert python is not None
    if args.output is not None:
        output = _path(base, args.output)
        assert output is not None
    else:
        parent = _path(
            base, document.get("output_parent", "/mnt/public/tgy/data/libero_eval")
        )
        assert parent is not None
        output = parent / f"campaign_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    return Campaign(
        config_path=config_path,
        output=output,
        python=python,
        libero_config_path=libero_path,
        tasks=tasks,
        resources=resources,
        artifacts=artifacts,
        repo_id_prefix=args.repo_id_prefix,
        resume=args.resume,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def selected_indices(task: TaskSpec) -> list[int]:
    indices = task.initial_states.get("indices")
    return (
        list(indices)
        if indices is not None
        else list(range(int(task.initial_states["count"])))
    )


def split_indices(indices: list[int], workers: int) -> list[list[int]]:
    if workers <= 0 or workers > len(indices):
        raise ValueError("worker count must be in [1, number of selected states]")
    base, remainder = divmod(len(indices), workers)
    shards: list[list[int]] = []
    start = 0
    for worker_id in range(workers):
        size = base + int(worker_id < remainder)
        shards.append(indices[start : start + size])
        start += size
    return shards


def scene_cache_dir(task: TaskSpec) -> Path:
    state = task.initial_states
    root = Path(str(state["cache_root"]))
    return (
        root
        / task.suite
        / f"task_{task.task_id:03d}"
        / (
            f"count_{int(state['count']):03d}_seed_{int(state['seed_start'])}_"
            f"cf_{int(task.evaluation['control_freq'])}hz_wait_{int(state['wait_steps'])}_"
            f"hold_{int(state['validation_hold_steps'])}_"
            f"drift_{str(state['maximum_stabilization_drift']).replace('.', 'p')}"
        )
    )


def scene_generator_command(
    campaign: Campaign, task: TaskSpec, cache: Path
) -> list[str]:
    state = task.initial_states
    command = [
        str(campaign.python),
        "-u",
        str(RUNTIME_SCRIPT),
        "generate",
        "--suite",
        task.suite,
        "--task-id",
        str(task.task_id),
        "--count",
        str(state["count"]),
        "--output",
        str(cache),
        "--seed-start",
        str(state["seed_start"]),
        "--wait-steps",
        str(state["wait_steps"]),
        "--validation-hold-steps",
        str(state["validation_hold_steps"]),
        "--control-freq",
        str(task.evaluation["control_freq"]),
        "--maximum-stabilization-drift",
        str(state["maximum_stabilization_drift"]),
    ]
    if state["max_attempts"] is not None:
        command.extend(("--max-attempts", str(state["max_attempts"])))
    return command


def worker_command(
    campaign: Campaign,
    task: TaskSpec,
    *,
    worker_id: int,
    indices: list[int],
    dataset: Path,
    manifest: Path | None,
    resume: bool,
) -> list[str]:
    evaluation = task.evaluation
    artifacts = campaign.artifacts
    command = [
        str(campaign.python),
        "-u",
        str(RUNTIME_SCRIPT),
        "worker",
        "--checkpoint",
        str(task.checkpoint),
        "--config-name",
        task.config_name,
        "--device",
        "cuda",
        "--suites",
        task.suite,
        "--task-ids",
        str(task.task_id),
        "--initial-state-indices",
        *[str(index) for index in indices],
        "--trials-per-initial-state",
        str(evaluation["trials_per_state"]),
        "--seed",
        str(evaluation["seed"]),
        "--policy-seed-offset",
        str(evaluation["policy_seed_offset"]),
        "--control-freq",
        str(evaluation["control_freq"]),
        "--fps",
        str(evaluation["control_freq"]),
        "--max-steps",
        str(evaluation["max_steps"]),
        "--post-success-steps",
        "0",
        "--num-steps-wait",
        str(evaluation["num_steps_wait"]),
        "--action-chunk",
        str(evaluation["action_chunk"]),
        "--num-inference-steps",
        str(evaluation["inference_steps"]),
        "--compile-mode",
        str(evaluation["compile_mode"]),
        "--gripper-hysteresis-threshold",
        str(evaluation["gripper_hysteresis_threshold"]),
        "--image-writer-threads",
        str(artifacts["image_writer_threads"]),
        "--output",
        str(dataset),
        "--repo-id",
        f"{campaign.repo_id_prefix}-{task.key}-worker{worker_id:02d}",
        "--openpi-source",
        str(task.openpi_source),
    ]
    if task.openpi_commit:
        command.extend(("--openpi-commit", task.openpi_commit))
    if task.openpi_norm_stats:
        command.extend(("--openpi-norm-stats", str(task.openpi_norm_stats)))
    if manifest is not None:
        command.extend(("--custom-initial-state-manifest", str(manifest)))
    for enabled, flag in (
        (bool(evaluation["binary_gripper"]), "--binary-gripper"),
        (bool(artifacts["training_schema_only"]), "--training-schema-only"),
        (bool(artifacts["save_images"]), "--save-images"),
        (artifacts["mode"] == "videos_only", "--videos-only"),
        (resume, "--resume"),
    ):
        if enabled:
            command.append(flag)
    return command


def build_workers(
    campaign: Campaign,
    task: TaskSpec,
    *,
    manifest: Path | None,
) -> list[dict[str, Any]]:
    indices = selected_indices(task)
    num_envs = min(int(campaign.resources["num_envs"]), len(indices))
    shards = split_indices(indices, num_envs)
    task_root = campaign.output / "tasks" / task.key
    workers = []
    for worker_id, shard in enumerate(shards):
        root = task_root / "workers" / f"worker_{worker_id:02d}"
        dataset = root / "dataset"
        results = dataset / "results.json"
        resume = campaign.resume and results.is_file()
        workers.append(
            {
                "worker_id": worker_id,
                "gpu": campaign.resources["gpus"][
                    worker_id % len(campaign.resources["gpus"])
                ],
                "indices": shard,
                "root": root,
                "dataset": dataset,
                "results": results,
                "log": root / "eval.log",
                "resume": resume,
                "command": worker_command(
                    campaign,
                    task,
                    worker_id=worker_id,
                    indices=shard,
                    dataset=dataset,
                    manifest=manifest,
                    resume=resume,
                ),
            }
        )
    return workers


def _task_contract(
    campaign: Campaign,
    task: TaskSpec,
    workers: list[dict[str, Any]],
    manifest: Path | None,
) -> dict[str, Any]:
    manifest_identity = None
    if task.initial_states["provider"] == "manifest" and manifest is not None:
        if manifest.is_file():
            manifest_hash = None
            try:
                manifest_hash = _file_hash(manifest)
                manifest_document = json.loads(manifest.read_text(encoding="utf-8"))
                state_file = manifest_document.get("state_file")
                state_path = (
                    (manifest.parent / str(state_file)).resolve()
                    if state_file
                    else None
                )
                manifest_identity = {
                    "manifest_sha256": manifest_hash,
                    "state_file_sha256": (
                        _file_hash(state_path)
                        if state_path is not None and state_path.is_file()
                        else None
                    ),
                    "validation_report_sha256": (
                        _file_hash(manifest.parent / "validation_report.json")
                        if (manifest.parent / "validation_report.json").is_file()
                        else None
                    ),
                    "identity_error": None,
                }
            except (OSError, TypeError, ValueError) as exc:
                manifest_identity = {
                    "manifest_sha256": manifest_hash,
                    "state_file_sha256": None,
                    "validation_report_sha256": None,
                    "identity_error": f"{type(exc).__name__}: {exc}",
                }
        else:
            manifest_identity = {
                "manifest_sha256": None,
                "state_file_sha256": None,
                "validation_report_sha256": None,
                "identity_error": "manifest does not exist",
            }
    return {
        "schema_version": 1,
        "key": task.key,
        "suite": task.suite,
        "task_id": task.task_id,
        "checkpoint": str(task.checkpoint),
        "config_name": task.config_name,
        "openpi_source": str(task.openpi_source),
        "openpi_commit": task.openpi_commit,
        "openpi_norm_stats": str(task.openpi_norm_stats)
        if task.openpi_norm_stats
        else None,
        "initial_states": task.initial_states,
        "evaluation": task.evaluation,
        "artifacts": campaign.artifacts,
        "repo_id_prefix": campaign.repo_id_prefix,
        "manifest": str(manifest) if manifest else None,
        "manifest_identity": manifest_identity,
        "num_envs": len(workers),
        "workers": [
            {
                "worker_id": worker["worker_id"],
                "initial_state_indices": worker["indices"],
                "dataset": str(worker["dataset"]),
            }
            for worker in workers
        ],
    }


def campaign_contract(
    campaign: Campaign, task_contracts: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": str(campaign.config_path),
        "output": str(campaign.output),
        "python": str(campaign.python),
        "libero_config_path": str(campaign.libero_config_path)
        if campaign.libero_config_path
        else None,
        "gpus": campaign.resources["gpus"],
        "num_envs": campaign.resources["num_envs"],
        "artifacts": campaign.artifacts,
        "repo_id_prefix": campaign.repo_id_prefix,
        "tasks": task_contracts,
    }


def _contract_identity(contract: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(contract)
    value.pop("generated_at", None)
    value.pop("gpus", None)
    value.pop("output", None)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_remove_output(path: Path) -> None:
    if path == Path(path.anchor) or len(path.parts) < 3:
        raise ValueError(f"refusing to recursively remove broad output path: {path}")
    shutil.rmtree(path)


def subprocess_env(campaign: Campaign, task: TaskSpec, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TF_ENABLE_ONEDNN_OPTS": "0",
            "MUJOCO_GL": "egl",
            "PYTHONFAULTHANDLER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
        }
    )
    paths = [
        str(REPO_ROOT / "vla-mender" / "src"),
        str(task.openpi_source / "src"),
        str(task.openpi_source / "packages" / "openpi-client" / "src"),
        os.environ.get("LIBERO_PYTHONPATH", "/opt/venv/openvla/libero"),
    ]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if campaign.libero_config_path is None:
        env.pop("LIBERO_CONFIG_PATH", None)
    else:
        env["LIBERO_CONFIG_PATH"] = str(campaign.libero_config_path)
    return env


def run_logged(
    command: list[str], *, log: Path, env: dict[str, str], cwd: Path, prefix: str
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(prefix + line)
            sys.stdout.flush()
            stream.write(line)
            stream.flush()
        return process.wait()


def _episode_keys(document: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (int(item["initial_state_index"]), int(item.get("trial_index", 0)))
        for item in document.get("episodes", [])
    ]


def _worker_complete(worker: dict[str, Any], trials: int) -> bool:
    if not worker["results"].is_file():
        return False
    document = json.loads(worker["results"].read_text(encoding="utf-8"))
    expected = {
        (index, trial) for trial in range(trials) for index in worker["indices"]
    }
    keys = _episode_keys(document)
    return len(keys) == len(set(keys)) and set(keys) == expected


def summarize_task(
    task: TaskSpec, workers: list[dict[str, Any]], returncodes: dict[int, int]
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []
    trials = int(task.evaluation["trials_per_state"])
    for worker in workers:
        error = None
        document: dict[str, Any] = {}
        if worker["results"].is_file():
            document = json.loads(worker["results"].read_text(encoding="utf-8"))
            episodes.extend(document.get("episodes", []))
        expected = {
            (index, trial) for trial in range(trials) for index in worker["indices"]
        }
        keys = _episode_keys(document)
        if len(keys) != len(set(keys)):
            error = "duplicate initial-state/trial results"
        elif set(keys) != expected:
            error = f"coverage mismatch: expected={sorted(expected)}, actual={sorted(set(keys))}"
        code = returncodes.get(worker["worker_id"], 1)
        if error is not None:
            code = 1
            returncodes[worker["worker_id"]] = 1
        worker_summaries.append(
            {
                "worker_id": worker["worker_id"],
                "gpu": worker["gpu"],
                "initial_state_indices": worker["indices"],
                "returncode": code,
                "validation_error": error,
                "results": str(worker["results"]),
            }
        )
    keys = _episode_keys({"episodes": episodes})
    successes = sum(bool(item.get("success")) for item in episodes)
    return {
        "schema_version": 1,
        "key": task.key,
        "suite": task.suite,
        "task_id": task.task_id,
        "requested_episodes": len(selected_indices(task)) * trials,
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else None,
        "failed_workers": [
            item["worker_id"] for item in worker_summaries if item["returncode"] != 0
        ],
        "duplicate_episode_keys": len(keys) != len(set(keys)),
        "workers": worker_summaries,
        "episode_results": sorted(
            episodes,
            key=lambda item: (
                int(item["initial_state_index"]),
                int(item.get("trial_index", 0)),
            ),
        ),
    }


def run_task(
    campaign: Campaign,
    task: TaskSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_root = campaign.output / "tasks" / task.key
    provider = task.initial_states["provider"]
    if provider == "official":
        manifest = None
        generator = None
    elif provider == "manifest":
        manifest = Path(str(task.initial_states["manifest"]))
        generator = None
    else:
        cache = scene_cache_dir(task)
        manifest = cache / "manifest.json"
        generator = scene_generator_command(campaign, task, cache)

    workers = build_workers(campaign, task, manifest=manifest)
    contract = _task_contract(campaign, task, workers, manifest)
    print(json.dumps(contract, indent=2, ensure_ascii=False), flush=True)
    if generator is not None:
        print("Generator: " + shlex.join(generator), flush=True)
    for worker in workers:
        print(
            f"Task {task.key} worker {worker['worker_id']:02d} GPU {worker['gpu']} "
            f"states={worker['indices']}: {shlex.join(worker['command'])}",
            flush=True,
        )
    if campaign.dry_run:
        return contract, {
            "key": task.key,
            "suite": task.suite,
            "task_id": task.task_id,
            "requested_episodes": len(selected_indices(task))
            * int(task.evaluation["trials_per_state"]),
            "dry_run": True,
        }

    _write_json(task_root / "task_contract.json", contract)
    if not task.checkpoint.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {task.checkpoint}"
        )
    if not task.openpi_source.is_dir():
        raise FileNotFoundError(f"OpenPI source does not exist: {task.openpi_source}")
    if generator is not None:
        code = run_logged(
            generator,
            log=task_root / "scene_generation.log",
            env=subprocess_env(campaign, task, int(campaign.resources["gpus"][0])),
            cwd=task.openpi_source,
            prefix=f"[{task.key} scenes] ",
        )
        if code != 0 or not manifest.is_file():
            summary = {
                "schema_version": 1,
                "key": task.key,
                "suite": task.suite,
                "task_id": task.task_id,
                "requested_episodes": len(selected_indices(task))
                * int(task.evaluation["trials_per_state"]),
                "episodes": 0,
                "successes": 0,
                "success_rate": None,
                "failed_workers": [],
                "task_error": f"scene generator failed with status {code}",
            }
            _write_json(task_root / "summary.json", summary)
            return contract, summary
    elif manifest is not None and not manifest.is_file():
        summary = {
            "schema_version": 1,
            "key": task.key,
            "suite": task.suite,
            "task_id": task.task_id,
            "requested_episodes": len(selected_indices(task))
            * int(task.evaluation["trials_per_state"]),
            "episodes": 0,
            "successes": 0,
            "success_rate": None,
            "failed_workers": [],
            "task_error": f"initial-state manifest not found: {manifest}",
        }
        _write_json(task_root / "summary.json", summary)
        return contract, summary
    else:
        source = (
            "official LIBERO states" if manifest is None else f"manifest {manifest}"
        )
        (task_root / "scene_generation.log").write_text(
            f"No scene generation required; using {source}.\n", encoding="utf-8"
        )

    returncodes: dict[int, int] = {}
    runnable = []
    trials = int(task.evaluation["trials_per_state"])
    for worker in workers:
        if campaign.resume and _worker_complete(worker, trials):
            returncodes[worker["worker_id"]] = 0
            print(
                f"Skipping completed task {task.key} worker {worker['worker_id']:02d}",
                flush=True,
            )
            continue
        if (
            campaign.resume
            and worker["dataset"].exists()
            and not worker["results"].is_file()
        ):
            returncodes[worker["worker_id"]] = 1
            print(
                f"Cannot resume task {task.key} worker {worker['worker_id']:02d} without results.json",
                file=sys.stderr,
            )
            continue
        runnable.append(worker)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(runnable))
    ) as executor:
        futures = {
            executor.submit(
                run_logged,
                worker["command"],
                log=worker["log"],
                env=subprocess_env(campaign, task, int(worker["gpu"])),
                cwd=task.openpi_source,
                prefix=f"[{task.key} w{worker['worker_id']:02d} gpu{worker['gpu']}] ",
            ): worker
            for worker in runnable
        }
        for future in concurrent.futures.as_completed(futures):
            worker = futures[future]
            try:
                returncodes[worker["worker_id"]] = future.result()
            except Exception as exc:  # pragma: no cover - defensive subprocess boundary
                returncodes[worker["worker_id"]] = 1
                print(f"Worker {worker['worker_id']} failed: {exc}", file=sys.stderr)
    summary = summarize_task(task, workers, returncodes)
    _write_json(task_root / "summary.json", summary)
    return contract, summary


def run_campaign(campaign: Campaign) -> int:
    if not campaign.dry_run:
        if not campaign.python.is_file():
            raise FileNotFoundError(
                f"Python interpreter does not exist: {campaign.python}"
            )
        if not RUNTIME_SCRIPT.is_file():
            raise FileNotFoundError(RUNTIME_SCRIPT)
        if campaign.overwrite and campaign.output.exists():
            _safe_remove_output(campaign.output)
        elif (
            campaign.output.exists()
            and any(campaign.output.iterdir())
            and not campaign.resume
        ):
            raise FileExistsError(
                f"output already exists: {campaign.output}; use --resume or --overwrite"
            )

    previews: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task in campaign.tasks:
        provider = task.initial_states["provider"]
        manifest = (
            None
            if provider == "official"
            else Path(str(task.initial_states["manifest"]))
            if provider == "manifest"
            else scene_cache_dir(task) / "manifest.json"
        )
        workers = build_workers(campaign, task, manifest=manifest)
        previews.append((_task_contract(campaign, task, workers, manifest), {}))
    contract = campaign_contract(campaign, [item[0] for item in previews])
    print(json.dumps(contract, indent=2, ensure_ascii=False), flush=True)
    actual_worker_counts = [
        min(int(campaign.resources["num_envs"]), len(selected_indices(task)))
        for task in campaign.tasks
    ]
    if max(actual_worker_counts) > len(campaign.resources["gpus"]):
        print(
            "WARNING: multiple workers may share a GPU and load independent checkpoint copies",
            file=sys.stderr,
        )
    if campaign.dry_run:
        for task in campaign.tasks:
            run_task(campaign, task)
        return 0

    campaign.output.mkdir(parents=True, exist_ok=True)
    contract_path = campaign.output / "eval_contract.json"
    if campaign.resume:
        if not contract_path.is_file():
            raise FileNotFoundError(f"cannot resume without {contract_path}")
        saved = json.loads(contract_path.read_text(encoding="utf-8"))
        if _contract_identity(saved) != _contract_identity(contract):
            raise ValueError("resume campaign contract differs from the saved contract")
    else:
        _write_json(contract_path, contract)

    task_summaries = []
    for task in campaign.tasks:
        try:
            _, summary = run_task(campaign, task)
        except Exception as exc:
            summary = {
                "schema_version": 1,
                "key": task.key,
                "suite": task.suite,
                "task_id": task.task_id,
                "requested_episodes": len(selected_indices(task))
                * int(task.evaluation["trials_per_state"]),
                "episodes": 0,
                "successes": 0,
                "success_rate": None,
                "failed_workers": [],
                "task_error": f"{type(exc).__name__}: {exc}",
            }
            _write_json(campaign.output / "tasks" / task.key / "summary.json", summary)
            print(f"Task {task.key} failed: {exc}", file=sys.stderr)
        task_summaries.append(summary)

    episodes = sum(int(item.get("episodes", 0)) for item in task_summaries)
    successes = sum(int(item.get("successes", 0)) for item in task_summaries)
    failed_tasks = [
        item["key"]
        for item in task_summaries
        if item.get("task_error")
        or item.get("failed_workers")
        or item.get("duplicate_episode_keys")
        or int(item.get("episodes", 0)) != int(item["requested_episodes"])
    ]
    summary = {
        "schema_version": 1,
        "requested_tasks": len(campaign.tasks),
        "requested_episodes": sum(
            int(item["requested_episodes"]) for item in task_summaries
        ),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else None,
        "failed_tasks": failed_tasks,
        "tasks": task_summaries,
    }
    _write_json(campaign.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 1 if failed_tasks else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    campaign = resolve_campaign(args)
    return run_campaign(campaign)


if __name__ == "__main__":
    raise SystemExit(main())
