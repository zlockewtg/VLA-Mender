"""Declarative multi-task repair configuration and pre-repair handoff loader."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .util import (
    read_json,
    readable_slug,
    resolve_beneath,
    resolve_path,
    safe_component,
    sha256_file,
)


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}


def _executable_path(base: Path, value: str | Path) -> Path:
    """Make an executable absolute without resolving a virtualenv symlink."""

    path = Path(value).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


class RepairConfigError(ValueError):
    """Raised when a repair campaign contract is incomplete or inconsistent."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RepairConfigError(f"{label} must be a mapping")
    return dict(value)


def _only(raw: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise RepairConfigError(f"unknown {label} fields: {unknown}")


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    output_dir: Path
    parallel_tasks: int


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    source_root: Path
    knowledge_root: Path


@dataclass(frozen=True)
class EnvironmentConfig:
    python: Path
    libero_root: Path
    working_directory: Path
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ServiceConfig:
    profile: str = "default"
    manage: bool = True
    keep_alive: bool = True
    port_base: int = 14000
    port_stride: int = 100
    startup_timeout_s: float = 900.0


@dataclass(frozen=True)
class ResourceConfig:
    gpus: tuple[int, ...]
    gpus_per_task: int
    workers_per_gpu: int
    services: ServiceConfig = field(default_factory=ServiceConfig)


@dataclass(frozen=True)
class RuntimeConfig:
    max_steps: int = 1000
    job_timeout_s: float = 1200.0
    infrastructure_retries: int = 2
    resume: bool = True
    backend: str = "libero"


@dataclass(frozen=True)
class ArtifactConfig:
    evidence_dedupe: str = "auto"


@dataclass(frozen=True)
class RepairBehaviorConfig:
    debug_parts: int | None = None
    validation_parts: int | None = None
    soft_task_hours: float = 4.0
    smoke_min_seeds: int = 3
    smoke_max_seeds: int = 8
    allow_abandon: bool = True
    consecutive_no_gain_candidates: int = 3
    per_seed_policy_attempts: int = 8


@dataclass(frozen=True)
class TaskSource:
    run_root: Path


@dataclass(frozen=True)
class RepairConfig:
    source_path: Path
    campaign: CampaignConfig
    project: ProjectConfig
    environment: EnvironmentConfig
    resources: ResourceConfig
    runtime: RuntimeConfig
    artifacts: ArtifactConfig
    repair: RepairBehaviorConfig
    tasks: tuple[TaskSource, ...]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("source_path")

        def normalize(item: Any) -> Any:
            if isinstance(item, Path):
                return str(item)
            if isinstance(item, tuple):
                return [normalize(child) for child in item]
            if isinstance(item, dict):
                return {key: normalize(child) for key, child in item.items()}
            if isinstance(item, list):
                return [normalize(child) for child in item]
            return item

        normalized = normalize(value)
        if self.schema_version == 2:
            normalized["repair"] = {
                "budget": {"soft_task_hours": self.repair.soft_task_hours},
                "smoke": {
                    "min_seeds": self.repair.smoke_min_seeds,
                    "max_seeds": self.repair.smoke_max_seeds,
                },
                "exploration_review": {
                    "consecutive_no_gain_candidates": (
                        self.repair.consecutive_no_gain_candidates
                    ),
                    "per_seed_policy_attempts": self.repair.per_seed_policy_attempts,
                },
                "allow_abandon": self.repair.allow_abandon,
            }
        else:
            normalized["repair"] = {
                "debug_parts": self.repair.debug_parts,
                "validation_parts": self.repair.validation_parts,
            }
        return normalized


def load_repair_config(path: str | Path) -> RepairConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RepairConfigError(f"repair YAML must contain a mapping: {source}")
    _only(
        raw,
        {
            "schema_version",
            "campaign",
            "project",
            "environment",
            "resources",
            "runtime",
            "artifacts",
            "repair",
            "tasks",
        },
        "top-level",
    )
    schema_version = int(raw.get("schema_version", 0))
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise RepairConfigError(
            f"schema_version must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    base = source.parent

    campaign_raw = _mapping(raw.get("campaign"), "campaign")
    _only(campaign_raw, {"name", "output_dir", "parallel_tasks"}, "campaign")
    campaign = CampaignConfig(
        name=str(campaign_raw.get("name", "")).strip(),
        output_dir=resolve_path(base, campaign_raw.get("output_dir", ".")),
        parallel_tasks=int(campaign_raw.get("parallel_tasks", 1)),
    )

    project_raw = _mapping(raw.get("project"), "project")
    _only(project_raw, {"root", "source_root", "knowledge_root"}, "project")
    project_root = resolve_path(base, project_raw.get("root", base))
    project = ProjectConfig(
        root=project_root,
        source_root=resolve_path(project_root, project_raw.get("source_root", "vla-mender/src")),
        knowledge_root=resolve_path(
            project_root, project_raw.get("knowledge_root", "vla-mender/knowledge")
        ),
    )

    environment_raw = _mapping(raw.get("environment"), "environment")
    _only(environment_raw, {"python", "libero_root", "working_directory", "env"}, "environment")
    extra_env = _mapping(environment_raw.get("env"), "environment.env")
    environment = EnvironmentConfig(
        python=_executable_path(base, environment_raw.get("python", "/usr/bin/python3")),
        libero_root=resolve_path(base, environment_raw.get("libero_root", ".")),
        working_directory=resolve_path(
            base, environment_raw.get("working_directory", project.root)
        ),
        env={str(key): str(value) for key, value in extra_env.items()},
    )

    resources_raw = _mapping(raw.get("resources"), "resources")
    _only(
        resources_raw,
        {"gpus", "gpus_per_task", "workers_per_gpu", "services"},
        "resources",
    )
    services_raw = _mapping(resources_raw.get("services"), "resources.services")
    _only(
        services_raw,
        {"profile", "manage", "keep_alive", "port_base", "port_stride", "startup_timeout_s"},
        "resources.services",
    )
    service = ServiceConfig(
        profile=str(services_raw.get("profile", "default")),
        manage=bool(services_raw.get("manage", True)),
        keep_alive=bool(services_raw.get("keep_alive", True)),
        port_base=int(services_raw.get("port_base", 14000)),
        port_stride=int(services_raw.get("port_stride", 100)),
        startup_timeout_s=float(services_raw.get("startup_timeout_s", 900.0)),
    )
    gpu_values = resources_raw.get("gpus", [0])
    if not isinstance(gpu_values, list):
        raise RepairConfigError("resources.gpus must be a list")
    resources = ResourceConfig(
        gpus=tuple(int(value) for value in gpu_values),
        gpus_per_task=int(resources_raw.get("gpus_per_task", 1)),
        workers_per_gpu=int(resources_raw.get("workers_per_gpu", 1)),
        services=service,
    )

    runtime_raw = _mapping(raw.get("runtime"), "runtime")
    _only(
        runtime_raw,
        {"max_steps", "job_timeout_s", "infrastructure_retries", "resume", "backend"},
        "runtime",
    )
    runtime = RuntimeConfig(
        max_steps=int(runtime_raw.get("max_steps", 1000)),
        job_timeout_s=float(runtime_raw.get("job_timeout_s", 1200.0)),
        infrastructure_retries=int(runtime_raw.get("infrastructure_retries", 2)),
        resume=bool(runtime_raw.get("resume", True)),
        backend=str(runtime_raw.get("backend", "libero")),
    )

    artifacts_raw = _mapping(raw.get("artifacts"), "artifacts")
    _only(artifacts_raw, {"evidence_dedupe"}, "artifacts")
    artifacts = ArtifactConfig(
        evidence_dedupe=str(artifacts_raw.get("evidence_dedupe", "auto")),
    )

    repair_raw = _mapping(raw.get("repair"), "repair")
    if schema_version == 1:
        _only(repair_raw, {"initial_split"}, "repair")
        split_raw = _mapping(repair_raw.get("initial_split"), "repair.initial_split")
        _only(split_raw, {"debug", "validation"}, "repair.initial_split")
        repair = RepairBehaviorConfig(
            debug_parts=int(split_raw.get("debug", 2)),
            validation_parts=int(split_raw.get("validation", 3)),
        )
    else:
        _only(
            repair_raw,
            {"budget", "smoke", "exploration_review", "allow_abandon"},
            "repair",
        )
        budget_raw = _mapping(repair_raw.get("budget"), "repair.budget")
        smoke_raw = _mapping(repair_raw.get("smoke"), "repair.smoke")
        review_raw = _mapping(
            repair_raw.get("exploration_review"), "repair.exploration_review"
        )
        _only(budget_raw, {"soft_task_hours"}, "repair.budget")
        _only(smoke_raw, {"min_seeds", "max_seeds"}, "repair.smoke")
        _only(
            review_raw,
            {"consecutive_no_gain_candidates", "per_seed_policy_attempts"},
            "repair.exploration_review",
        )
        repair = RepairBehaviorConfig(
            soft_task_hours=float(budget_raw.get("soft_task_hours", 4.0)),
            smoke_min_seeds=int(smoke_raw.get("min_seeds", 3)),
            smoke_max_seeds=int(smoke_raw.get("max_seeds", 8)),
            allow_abandon=bool(repair_raw.get("allow_abandon", True)),
            consecutive_no_gain_candidates=int(
                review_raw.get("consecutive_no_gain_candidates", 3)
            ),
            per_seed_policy_attempts=int(review_raw.get("per_seed_policy_attempts", 8)),
        )

    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise RepairConfigError("tasks must be a non-empty list")
    tasks: list[TaskSource] = []
    for index, task_value in enumerate(tasks_raw):
        task_raw = _mapping(task_value, f"tasks[{index}]")
        _only(task_raw, {"run_root"}, f"tasks[{index}]")
        if "run_root" not in task_raw:
            raise RepairConfigError(f"tasks[{index}].run_root is required")
        tasks.append(TaskSource(run_root=resolve_path(base, task_raw["run_root"])))

    config = RepairConfig(
        source_path=source,
        campaign=campaign,
        project=project,
        environment=environment,
        resources=resources,
        runtime=runtime,
        artifacts=artifacts,
        repair=repair,
        tasks=tuple(tasks),
        schema_version=schema_version,
    )
    _validate_config(config)
    return config


def _validate_config(config: RepairConfig) -> None:
    if not config.campaign.name:
        raise RepairConfigError("campaign.name cannot be empty")
    if config.campaign.parallel_tasks < 1:
        raise RepairConfigError("campaign.parallel_tasks must be positive")
    if not config.resources.gpus or len(set(config.resources.gpus)) != len(config.resources.gpus):
        raise RepairConfigError("resources.gpus must be non-empty and unique")
    if any(gpu < 0 for gpu in config.resources.gpus):
        raise RepairConfigError("GPU IDs cannot be negative")
    if config.resources.gpus_per_task < 1:
        raise RepairConfigError("resources.gpus_per_task must be positive")
    required_gpu_slots = config.campaign.parallel_tasks * config.resources.gpus_per_task
    if required_gpu_slots > len(config.resources.gpus):
        raise RepairConfigError(
            "parallel_tasks * resources.gpus_per_task cannot exceed the number of GPUs"
        )
    if config.resources.workers_per_gpu < 1:
        raise RepairConfigError("workers_per_gpu must be positive")
    if config.resources.services.port_base < 1024 or config.resources.services.port_stride < 3:
        raise RepairConfigError("service port_base must be >=1024 and port_stride >=3")
    if config.runtime.max_steps < 1 or config.runtime.job_timeout_s <= 0:
        raise RepairConfigError("runtime step and timeout limits must be positive")
    if config.runtime.infrastructure_retries < 0:
        raise RepairConfigError("infrastructure_retries cannot be negative")
    if config.runtime.backend not in {"libero", "fake"}:
        raise RepairConfigError("runtime.backend must be libero or fake")
    if config.artifacts.evidence_dedupe not in {"auto", "hardlink", "off"}:
        raise RepairConfigError(
            "artifacts.evidence_dedupe must be auto, hardlink, or off"
        )
    if config.schema_version == 1:
        if config.resources.gpus_per_task != 1:
            raise RepairConfigError(
                "schema v1 supports exactly one GPU per task; use schema v2 for GPU groups"
            )
        assert config.repair.debug_parts is not None
        assert config.repair.validation_parts is not None
        if min(config.repair.debug_parts, config.repair.validation_parts) < 1:
            raise RepairConfigError("initial split parts must be positive")
    else:
        if config.repair.soft_task_hours <= 0:
            raise RepairConfigError("repair.budget.soft_task_hours must be positive")
        if config.repair.smoke_min_seeds < 1:
            raise RepairConfigError("repair.smoke.min_seeds must be positive")
        if config.repair.smoke_max_seeds < config.repair.smoke_min_seeds:
            raise RepairConfigError(
                "repair.smoke.max_seeds must be greater than or equal to min_seeds"
            )
        if config.repair.consecutive_no_gain_candidates < 1:
            raise RepairConfigError(
                "repair.exploration_review.consecutive_no_gain_candidates must be positive"
            )
        if config.repair.per_seed_policy_attempts < 1:
            raise RepairConfigError(
                "repair.exploration_review.per_seed_policy_attempts must be positive"
            )
    if len({task.run_root for task in config.tasks}) != len(config.tasks):
        raise RepairConfigError("task run_root values must be unique")
    for label, path, kind in (
        ("project.root", config.project.root, "dir"),
        ("project.source_root", config.project.source_root, "dir"),
        ("project.knowledge_root", config.project.knowledge_root, "dir"),
        ("environment.python", config.environment.python, "file"),
        ("environment.libero_root", config.environment.libero_root, "dir"),
        ("environment.working_directory", config.environment.working_directory, "dir"),
    ):
        exists = path.is_file() if kind == "file" else path.is_dir()
        if not exists:
            raise RepairConfigError(f"{label} does not exist: {path}")


def _task_identity(
    run_root: Path, index: int, seen: set[str]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    resolved_path = run_root / "experiment.resolved.yaml"
    summary_path = run_root / "rollout" / "summary.json"
    if not resolved_path.is_file() or not summary_path.is_file():
        raise RepairConfigError(
            f"run_root lacks experiment.resolved.yaml or rollout/summary.json: {run_root}"
        )
    settings = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or not isinstance(settings.get("task"), dict):
        raise RepairConfigError(f"invalid resolved experiment task: {resolved_path}")
    summary = read_json(summary_path)
    task = dict(settings["task"])
    suite = str(task.get("suite", ""))
    task_id = int(task.get("task_id", -1))
    if not suite or task_id < 0:
        raise RepairConfigError(f"invalid suite/task_id in {resolved_path}")
    base = safe_component(f"{run_root.name}_{suite}_task{task_id}")
    task_key = base
    suffix = 2
    while task_key in seen:
        task_key = f"{base}_{suffix}"
        suffix += 1
    seen.add(task_key)
    return task_key, settings, summary


def _initial_partitions(jobs: list[dict[str, Any]], behavior: RepairBehaviorConfig) -> None:
    assert behavior.debug_parts is not None
    assert behavior.validation_parts is not None
    groups: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        groups.setdefault(str(job["failure_mode_id"]), []).append(job)
    total_parts = behavior.debug_parts + behavior.validation_parts
    for values in groups.values():
        ordered = sorted(
            values,
            key=lambda item: hashlib.sha256(str(item["job_id"]).encode("utf-8")).hexdigest(),
        )
        if len(ordered) < total_parts:
            for item in ordered:
                item["initial_partition"] = "open"
            continue
        debug_count = max(
            1,
            min(
                len(ordered) - 1,
                round(len(ordered) * behavior.debug_parts / total_parts),
            ),
        )
        for index, item in enumerate(ordered):
            item["initial_partition"] = "debug" if index < debug_count else "validation"


def resolve_repair_inputs(config: RepairConfig) -> dict[str, Any]:
    """Resolve trusted pre-repair outputs into a campaign inventory."""

    tasks: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_slugs: set[str] = set()
    for task_index, source in enumerate(config.tasks):
        run_root = source.run_root.resolve()
        task_key, settings, summary = _task_identity(run_root, task_index, seen_keys)
        handoff_root = run_root / "repair_handoff"
        handoff_path = handoff_root / "manifest.json"
        unified_handoff = handoff_path.is_file()
        if unified_handoff:
            handoff = read_json(handoff_path)
            if (
                int(handoff.get("schema_version", 0)) != 1
                or handoff.get("artifact_type") != "vla_mender.repair_handoff"
                or handoff.get("complete") is not True
            ):
                raise RepairConfigError(f"invalid/incomplete repair handoff: {handoff_path}")
            diagnosis = handoff.get("diagnosis")
            source_jobs = handoff.get("resets")
            public_resets = source_jobs
            replay_reports = source_jobs
            artifact_root = handoff_root
            artifact_paths = {
                "repair_handoff": str(handoff_root),
                "handoff_manifest": str(handoff_path),
            }
            if not isinstance(diagnosis, dict):
                raise RepairConfigError(f"repair handoff lacks diagnosis object: {handoff_path}")
            if (
                int(diagnosis.get("schema_version", 0)) != 1
                or not isinstance(diagnosis.get("failure_modes"), list)
                or not isinstance(diagnosis.get("episodes"), list)
            ):
                raise RepairConfigError(f"invalid diagnosis in repair handoff: {handoff_path}")
            handoff_summary = handoff.get("summary")
            if not isinstance(handoff_summary, dict):
                raise RepairConfigError(f"repair handoff lacks summary object: {handoff_path}")
            if (
                int(handoff_summary.get("failure_episode_count", -1))
                != len(diagnosis["episodes"])
                or int(handoff_summary.get("failure_mode_count", -1))
                != len(diagnosis["failure_modes"])
                or int(handoff_summary.get("reset_count", -1))
                != len(source_jobs or [])
                or int(handoff_summary.get("replay_verified_count", -1))
                != len(source_jobs or [])
                or handoff_summary.get("all_replays_verified") is not True
            ):
                raise RepairConfigError(f"incomplete/inconsistent handoff summary: {handoff_path}")
        else:
            # Read-only compatibility for pre-repair runs created before the
            # unified handoff contract. New runs must publish repair_handoff/.
            failure_root = run_root / "failure_diagnosis"
            required = {
                "diagnosis": failure_root / "diagnosis.json",
                "jobs": failure_root / "repair_jobs.json",
                "public_bank": failure_root / "public_reset_bank.json",
                "replay": failure_root / "replay_verification.json",
            }
            missing = [label for label, path in required.items() if not path.is_file()]
            if missing:
                raise RepairConfigError(
                    f"task {run_root} lacks repair_handoff/manifest.json and legacy "
                    f"repair artifacts: {missing}"
                )
            diagnosis = read_json(required["diagnosis"])
            job_manifest = read_json(required["jobs"])
            public_bank = read_json(required["public_bank"])
            replay_verification = read_json(required["replay"])
            source_jobs = job_manifest.get("jobs")
            public_resets = public_bank.get("resets")
            replay_reports = replay_verification.get("reports")
            artifact_root = failure_root
            artifact_paths = {
                "diagnosis": str(required["diagnosis"]),
                "repair_jobs": str(required["jobs"]),
                "public_reset_bank": str(required["public_bank"]),
                "replay_verification": str(required["replay"]),
            }
        if not isinstance(source_jobs, list) or not isinstance(public_resets, list):
            raise RepairConfigError(f"invalid repair job/reset bank schema under {artifact_root}")
        if unified_handoff:
            selection = handoff.get("selection")
            if not isinstance(selection, dict):
                raise RepairConfigError(f"repair handoff lacks selection object: {handoff_path}")
            frames_per_failure = int(selection.get("frames_per_failure", 0))
            diagnosed_episodes = {
                int(item["episode_index"])
                for item in diagnosis["episodes"]
                if isinstance(item, dict) and "episode_index" in item
            }
            reset_counts: dict[int, int] = {}
            job_ids: list[str] = []
            reset_keys: list[tuple[int, int]] = []
            required_reset_fields = {
                "job_id",
                "episode_index",
                "requested_frame_index",
                "reset_frame_index",
                "failure_mode_id",
                "target_control_space",
                "reset_dynamics",
                "verified",
                "reset_state",
                "reset_state_file_sha256",
                "agent_view",
                "agent_view_file_sha256",
            }
            for item in source_jobs:
                if not isinstance(item, dict):
                    raise RepairConfigError(f"non-object reset in {handoff_path}")
                missing_fields = sorted(required_reset_fields - set(item))
                if missing_fields:
                    raise RepairConfigError(
                        f"reset in {handoff_path} lacks required fields: {missing_fields}"
                    )
                episode = int(item["episode_index"])
                requested = int(item["requested_frame_index"])
                frame = int(item["reset_frame_index"])
                if requested != frame:
                    raise RepairConfigError(
                        f"requested/reset frame mismatch for {item.get('job_id')}"
                    )
                reset_counts[episode] = reset_counts.get(episode, 0) + 1
                job_ids.append(str(item["job_id"]))
                reset_keys.append((episode, frame))
            if (
                frames_per_failure < 1
                or set(reset_counts) != diagnosed_episodes
                or any(count != frames_per_failure for count in reset_counts.values())
            ):
                raise RepairConfigError(
                    f"handoff reset coverage does not match diagnosed failures: {handoff_path}"
                )
            if len(job_ids) != len(set(job_ids)) or len(reset_keys) != len(set(reset_keys)):
                raise RepairConfigError(f"duplicate repair job/reset in {handoff_path}")
        reset_index = {
            (
                int(item["episode_index"]),
                int(item.get("requested_frame_index", item.get("reset_frame_index", -1))),
            ): item
            for item in public_resets
            if isinstance(item, dict)
        }
        if not isinstance(replay_reports, list):
            raise RepairConfigError(f"invalid replay verification schema: {artifact_root}")
        replay_index = {
            (
                int(item["episode_index"]),
                int(item.get("requested_frame_index", item.get("reset_frame_index", -1))),
            ): item
            for item in replay_reports
            if isinstance(item, dict)
        }
        task_settings = dict(settings["task"])
        rollout_settings = dict(settings.get("rollout", {}))
        task_jobs: list[dict[str, Any]] = []
        for source_job in source_jobs:
            if not isinstance(source_job, dict):
                raise RepairConfigError(f"non-object job in {artifact_root}")
            episode = int(source_job["episode_index"])
            frame = int(
                source_job.get("reset_frame_index", source_job.get("requested_frame_index", -1))
            )
            public = reset_index.get((episode, frame))
            if public is None or not bool(public.get("verified")):
                raise RepairConfigError(
                    f"unverified/missing public reset for episode={episode} frame={frame}"
                )
            replay = replay_index.get((episode, frame))
            if replay is None or not bool(replay.get("verified")):
                raise RepairConfigError(
                    f"unverified/missing replay report for episode={episode} frame={frame}"
                )
            reset_value = str(source_job["reset_state"])
            agent_value = str(source_job["agent_view"])
            reset_path = resolve_beneath(artifact_root, reset_value)
            if not reset_path.is_file():
                reset_path = resolve_beneath(
                    artifact_root / "private_reset_states", Path(reset_value).name
                )
            agent_path = resolve_beneath(artifact_root, agent_value)
            if not agent_path.is_file():
                agent_path = resolve_beneath(artifact_root / "agent_views", Path(agent_value).name)
            if not reset_path.is_file() or not agent_path.is_file():
                raise RepairConfigError(
                    f"missing reset state or agent view for {source_job.get('job_id')}"
                )
            episode_path = run_root / "rollout" / "episodes" / f"episode_{episode:06d}.json"
            episode_record = read_json(episode_path)
            reset_hash = str(
                public.get("private_state_sha256")
                or public.get("reset_state_sha256")
                or sha256_file(reset_path)
            )
            actual_file_hash = sha256_file(reset_path)
            actual_view_hash = sha256_file(agent_path)
            if unified_handoff:
                if source_job.get("reset_state_file_sha256") != actual_file_hash:
                    raise RepairConfigError(
                        f"reset state file hash mismatch for {source_job.get('job_id')}"
                    )
                if source_job.get("agent_view_file_sha256") != actual_view_hash:
                    raise RepairConfigError(
                        f"agent view file hash mismatch for {source_job.get('job_id')}"
                    )
            resolved = {
                "job_id": f"{task_key}:{source_job['job_id']}",
                "source_job_id": str(source_job["job_id"]),
                "task_key": task_key,
                "run_root": str(run_root),
                "suite": str(task_settings["suite"]),
                "task_id": int(task_settings["task_id"]),
                "task_description": str(
                    task_settings.get("task_description")
                    or summary.get("task", {}).get("description")
                    or ""
                ),
                "episode_index": episode,
                "reset_frame_index": frame,
                "scene_model_seed": int(episode_record["scene_model_seed"]),
                "failure_phase": str(source_job.get("failure_phase", "")),
                "failure_mode_id": str(source_job["failure_mode_id"]),
                "failure_category": str(source_job.get("failure_category", "")),
                "failure_mode": str(source_job.get("failure_mode", "")),
                "target_control_space": str(source_job["target_control_space"]),
                "reset_dynamics": str(source_job["reset_dynamics"]),
                "control_frequency_hz": int(rollout_settings.get("control_frequency_hz", 20)),
                "reset_state": str(reset_path),
                "reset_hash": reset_hash,
                "reset_file_sha256": actual_file_hash,
                "agent_view": str(agent_path),
                "agent_view_sha256": str(
                    public.get("agent_view_sha256") or sha256_file(agent_path)
                ),
                "agent_view_file_sha256": actual_view_hash,
                "source_episode": str(episode_path),
                "public_reset": public,
            }
            if config.schema_version == 2:
                resolved["seed_slug"] = f"episode_{episode:06d}_frame_{frame:06d}"
            task_jobs.append(resolved)
            jobs.append(resolved)
        if config.schema_version == 1:
            _initial_partitions(task_jobs, config.repair)
        mode_counts: dict[str, int] = {}
        for item in task_jobs:
            mode_id = str(item["failure_mode_id"])
            mode_counts[mode_id] = mode_counts.get(mode_id, 0) + 1
        description = task_jobs[0]["task_description"] if task_jobs else ""
        task_slug = readable_slug(
            description,
            fallback=readable_slug(f"{task_settings['suite']}_task_{task_settings['task_id']}")
        )
        if task_slug in seen_slugs:
            task_slug = readable_slug(
                f"{task_slug}_{task_settings['suite']}_task_{task_settings['task_id']}"
            )
        slug_base = task_slug
        slug_suffix = 2
        while task_slug in seen_slugs:
            task_slug = f"{slug_base}_{slug_suffix}"
            slug_suffix += 1
        seen_slugs.add(task_slug)
        resolved_task = {
                "task_key": task_key,
                "run_root": str(run_root),
                "suite": str(task_settings["suite"]),
                "task_id": int(task_settings["task_id"]),
                "description": description,
                "job_count": len(task_jobs),
                "failure_mode_counts": mode_counts,
                **artifact_paths,
                "successful_episodes": str(run_root / "rollout" / "successful_episodes.json"),
                "rollout_summary": str(run_root / "rollout" / "summary.json"),
            }
        if config.schema_version == 2:
            resolved_task["task_slug"] = task_slug
        tasks.append(resolved_task)
    if not jobs:
        raise RepairConfigError("selected task runs contain no repair jobs")
    resolved = {
        "schema_version": config.schema_version,
        "campaign": config.to_dict(),
        "tasks": tasks,
        "jobs": jobs,
    }
    return resolved
