"""Public API used by prompt-launched IDE coordinator and task agents."""

from __future__ import annotations

import fcntl
import os
from concurrent.futures import FIRST_COMPLETED, Future, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from .artifacts import (
    AttemptStore,
    ProgramRecord,
    ProgramStore,
    ReadableAttemptStore,
    TaskProgramStore,
)
from .experience import ExperienceLibrary
from .runtime import EvaluationHandle, TaskLocalRuntime
from .state import FailureModeSession, SoftBudgetReviewRequired
from .util import (
    atomic_write_json,
    locked_file,
    read_json,
    safe_component,
    utc_now,
)


class GpuLease:
    """Process-scoped, crash-released exclusive task/GPU ownership."""

    def __init__(self, campaign_root: Path, task_key: str, gpu_id: int):
        self.path = campaign_root / "runtime" / "gpu_leases" / f"gpu_{gpu_id}.lock"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.task_key = task_key
        self.gpu_id = int(gpu_id)
        self._stream: Any = None

    def acquire(self) -> None:
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.seek(0)
            owner = stream.read().strip() or "unknown owner"
            stream.close()
            raise RuntimeError(f"GPU {self.gpu_id} is already leased: {owner}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"task={self.task_key} pid={os.getpid()} acquired_at={utc_now()}\n")
        stream.flush()
        self._stream = stream

    def release(self) -> None:
        if self._stream is None:
            return
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()
        self._stream = None


class NamedLease:
    """Small reusable exclusive lock with human-readable ownership."""

    def __init__(self, path: Path, owner: str):
        self.path = path
        self.owner = owner
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: Any = None

    def try_acquire(self) -> bool:
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            return False
        stream.seek(0)
        stream.truncate()
        stream.write(f"owner={self.owner} pid={os.getpid()} acquired_at={utc_now()}\n")
        stream.flush()
        self._stream = stream
        return True

    def acquire(self) -> None:
        if not self.try_acquire():
            raise RuntimeError(f"resource is already leased: {self.path}")

    def release(self) -> None:
        if self._stream is None:
            return
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()
        self._stream = None


class RepairCampaign:
    """One experiment-local repair inventory, runtime, and experience library."""

    def __init__(self, resolved_settings: str | Path):
        self.settings_path = Path(resolved_settings).resolve()
        raw = yaml.safe_load(self.settings_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"invalid repair settings: {self.settings_path}")
        self.settings: dict[str, Any] = raw
        self.schema_version = int(raw.get("schema_version", 1))
        if self.schema_version not in {1, 2}:
            raise ValueError(f"unsupported repair schema_version: {self.schema_version}")
        self.root = Path(str(raw["campaign"]["output_dir"])).resolve()
        if self.settings_path.parent != self.root:
            raise ValueError("repair_resolved.yaml must reside in campaign.output_dir")
        jobs = raw.get("jobs")
        if jobs is None and raw.get("resolved_jobs"):
            # Read-only compatibility for campaigns generated before the
            # resolved settings and job inventory were merged.
            inventory_path = Path(str(raw["resolved_jobs"])).resolve()
            if inventory_path.parent != self.root:
                raise ValueError("repair_jobs_resolved.json must reside in campaign.output_dir")
            inventory = read_json(inventory_path)
            if not isinstance(inventory, dict):
                raise ValueError(f"invalid repair job inventory: {inventory_path}")
            if int(inventory.get("schema_version", -1)) != self.schema_version:
                raise ValueError("resolved repair settings and job inventory schemas differ")
            jobs = inventory.get("jobs")
        if not isinstance(raw.get("resolved_tasks"), list) or not isinstance(jobs, list):
            raise ValueError("resolved repair tasks and jobs must be lists")
        self.jobs = [dict(item) for item in jobs]
        self.tasks = {str(item["task_key"]): dict(item) for item in raw["resolved_tasks"]}
        job_ids = [str(item["job_id"]) for item in self.jobs]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("resolved repair job IDs must be unique")
        unknown_task_keys = sorted({str(item["task_key"]) for item in self.jobs} - set(self.tasks))
        if unknown_task_keys:
            raise ValueError(f"repair jobs reference unknown tasks: {unknown_task_keys}")
        self.program_store = ProgramStore(self.root)
        artifact_settings = raw.get("artifacts", {})
        if not isinstance(artifact_settings, dict):
            raise ValueError("repair artifacts settings must be a mapping")
        self.attempt_store = (
            ReadableAttemptStore(
                self.root,
                evidence_dedupe=str(
                    artifact_settings.get("evidence_dedupe", "auto")
                ),
            )
            if self.schema_version >= 2
            else AttemptStore(self.root)
        )
        self.experience = ExperienceLibrary(
            self.root,
            self.program_store,
            schema_version=self.schema_version,
        )
        self._sessions: dict[str, TaskSession] = {}

    @classmethod
    def open(cls, resolved_settings: str | Path) -> RepairCampaign:
        return cls(resolved_settings)

    def open_task(
        self,
        task_key: str,
        *,
        gpu_id: int | None = None,
        gpu_ids: Iterable[int] | None = None,
    ) -> TaskSession:
        if task_key not in self.tasks:
            raise KeyError(f"unknown repair task: {task_key}")
        if task_key in self._sessions and not self._sessions[task_key].closed:
            raise RuntimeError(f"task is already open in this campaign process: {task_key}")
        declared = [int(value) for value in self.settings["resources"]["gpus"]]
        required = int(self.settings["resources"].get("gpus_per_task", 1))
        if gpu_id is not None and gpu_ids is not None:
            raise ValueError("pass either gpu_id or gpu_ids, not both")
        if gpu_ids is not None:
            selected = tuple(int(value) for value in gpu_ids)
        elif gpu_id is not None:
            if required != 1:
                raise ValueError(
                    f"this campaign requires {required} GPUs per task; pass gpu_ids=[...]"
                )
            selected = (int(gpu_id),)
        elif int(self.settings["campaign"]["parallel_tasks"]) == 1:
            selected = tuple(declared[:required])
        else:
            raise ValueError("multi-task campaigns must pass an explicit gpu_ids group")
        if len(selected) != required or len(set(selected)) != len(selected):
            raise ValueError(
                f"task requires exactly {required} unique GPUs; received={list(selected)}"
            )
        unknown = sorted(set(selected) - set(declared))
        if unknown:
            raise ValueError(
                f"GPUs are not declared by the campaign: {unknown}; available={declared}"
            )
        slots = tuple(declared.index(value) for value in selected)
        session = TaskSession(self, task_key, selected, slots)
        self._sessions[task_key] = session
        return session

    def close(self) -> None:
        for session in list(self._sessions.values()):
            session.close(status="closed")

    def __enter__(self) -> RepairCampaign:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class TaskSession:
    def __init__(
        self,
        campaign: RepairCampaign,
        task_key: str,
        gpu_ids: tuple[int, ...],
        gpu_slots: tuple[int, ...],
    ):
        self.campaign = campaign
        self.task_key = task_key
        self.gpu_ids = gpu_ids
        self.gpu_slots = gpu_slots
        # Backward-compatible primary GPU attributes for single-GPU callers.
        self.gpu_id = gpu_ids[0]
        self.gpu_slot = gpu_slots[0]
        self.task = dict(campaign.tasks[task_key])
        self.task_slug = str(self.task.get("task_slug") or safe_component(task_key))
        self.task_root = campaign.root / "tasks" / self.task_slug
        if campaign.schema_version >= 2:
            self.task_root.mkdir(parents=True, exist_ok=True)
        self.programs = (
            None
            if campaign.schema_version >= 2
            else TaskProgramStore(campaign.program_store, task_key)
        )
        self._jobs = [job for job in campaign.jobs if str(job["task_key"]) == task_key]
        self._leases = [GpuLease(campaign.root, task_key, value) for value in gpu_ids]
        self._task_lease = NamedLease(
            campaign.root / "runtime" / "task_leases" / f"{safe_component(task_key)}.lock",
            task_key,
        )
        self._slot_lease: NamedLease | None = None
        self._runtime: TaskLocalRuntime | None = None
        self._runtimes: list[TaskLocalRuntime] = []
        self.closed = False
        self.status_path = (
            self.task_root / "status.json"
            if campaign.schema_version >= 2
            else campaign.root / "runtime" / "tasks" / safe_component(task_key) / "status.json"
        )
        self.task_state_path = self.task_root / "task_state.json"
        self.task_state_lock = self.task_root / "task_state.lock"
        if campaign.schema_version >= 2:
            self._initialize_task_state()
        self._write_status("opened")

    def _initialize_task_state(self) -> None:
        with locked_file(self.task_state_lock):
            if self.task_state_path.exists():
                return
            soft_hours = float(self.campaign.settings["repair"]["budget"]["soft_task_hours"])
            atomic_write_json(
                self.task_state_path,
                {
                    "schema_version": 2,
                    "task_key": self.task_key,
                    "task_slug": self.task_slug,
                    "status": "not_started",
                    "budget": {
                        "soft_task_hours": soft_hours,
                        "total_authorized_hours": soft_hours,
                        "started_at": None,
                        "review_required": False,
                        "extensions": [],
                    },
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                },
            )

    def _write_status(self, status: str, **extra: Any) -> None:
        atomic_write_json(
            self.status_path,
            {
                "schema_version": 1,
                "task_key": self.task_key,
                "gpu_id": self.gpu_id,
                "gpu_ids": list(self.gpu_ids),
                "status": status,
                "pid": os.getpid(),
                "updated_at": utc_now(),
                **extra,
            },
        )
        if self.campaign.schema_version >= 2:
            with locked_file(self.task_state_lock):
                state = read_json(self.task_state_path)
                state.update({"status": status, "updated_at": utc_now(), **extra})
                atomic_write_json(self.task_state_path, state)

    def jobs(
        self,
        *,
        mode_ids: Iterable[str] | None = None,
        partition: str | None = None,
    ) -> list[dict[str, Any]]:
        allowed_modes = set(mode_ids or ())
        if self.campaign.schema_version >= 2 and partition is not None:
            raise ValueError("v2 repair campaigns expose one open seed pool; partition is removed")
        if partition is not None and partition not in {"debug", "validation", "open"}:
            raise ValueError("partition must be debug, validation, open, or None")
        return [
            dict(job)
            for job in self._jobs
            if (not allowed_modes or str(job["failure_mode_id"]) in allowed_modes)
            and (partition is None or str(job.get("initial_partition", "open")) == partition)
        ]

    def _start_budget_clock(self) -> None:
        if self.campaign.schema_version < 2:
            return
        with locked_file(self.task_state_lock):
            state = read_json(self.task_state_path)
            if state["budget"].get("started_at") is None:
                state["budget"]["started_at"] = utc_now()
                state["updated_at"] = utc_now()
                atomic_write_json(self.task_state_path, state)

    def budget_snapshot(self) -> dict[str, Any]:
        if self.campaign.schema_version < 2:
            raise RuntimeError("soft budget is available only for v2 campaigns")
        with locked_file(self.task_state_lock):
            state = read_json(self.task_state_path)
            budget = dict(state["budget"])
        started = budget.get("started_at")
        elapsed_hours = 0.0
        if started:
            elapsed_hours = max(
                0.0,
                (datetime.fromisoformat(utc_now()) - datetime.fromisoformat(str(started))).total_seconds()
                / 3600.0,
            )
        budget["elapsed_hours"] = elapsed_hours
        budget["remaining_hours"] = max(
            0.0, float(budget["total_authorized_hours"]) - elapsed_hours
        )
        budget["over_budget"] = elapsed_hours >= float(budget["total_authorized_hours"])
        return budget

    def require_budget_decision(self) -> None:
        if self.campaign.schema_version < 2:
            return
        snapshot = self.budget_snapshot()
        if not snapshot["over_budget"]:
            return
        with locked_file(self.task_state_lock):
            state = read_json(self.task_state_path)
            state["budget"]["review_required"] = True
            state["status"] = "budget_review_required"
            state["updated_at"] = utc_now()
            atomic_write_json(self.task_state_path, state)
        snapshot["review_required"] = True
        self._write_status("budget_review_required", budget=snapshot)
        raise SoftBudgetReviewRequired(
            "task crossed its soft repair budget; extend the budget with a reason or abandon seeds"
        )

    def extend_budget(self, additional_hours: float, *, reason: str) -> dict[str, Any]:
        if self.campaign.schema_version < 2:
            raise RuntimeError("soft budget is available only for v2 campaigns")
        if additional_hours <= 0 or not reason.strip():
            raise ValueError("budget extension requires positive hours and a non-empty reason")
        with locked_file(self.task_state_lock):
            state = read_json(self.task_state_path)
            budget = state["budget"]
            budget["total_authorized_hours"] = (
                float(budget["total_authorized_hours"]) + float(additional_hours)
            )
            budget["review_required"] = False
            budget.setdefault("extensions", []).append(
                {
                    "additional_hours": float(additional_hours),
                    "reason": reason,
                    "decided_at": utc_now(),
                }
            )
            state["status"] = "running"
            state["updated_at"] = utc_now()
            atomic_write_json(self.task_state_path, state)
        self._write_status("running")
        return self.budget_snapshot()

    def ensure_runtime(self) -> None:
        if self.closed:
            raise RuntimeError("task session is closed")
        if self._runtime is not None:
            return
        self._task_lease.acquire()
        acquired_leases: list[GpuLease] = []
        try:
            slot_count = int(self.campaign.settings["campaign"]["parallel_tasks"])
            for index in range(slot_count):
                lease = NamedLease(
                    self.campaign.root / "runtime" / "task_slots" / f"slot_{index}.lock",
                    self.task_key,
                )
                if lease.try_acquire():
                    self._slot_lease = lease
                    break
            if self._slot_lease is None:
                raise RuntimeError(
                    f"campaign already has {slot_count} active task agents; no task slot is free"
                )
            for lease in sorted(self._leases, key=lambda value: value.gpu_id):
                lease.acquire()
                acquired_leases.append(lease)
            for gpu_id, gpu_slot in zip(self.gpu_ids, self.gpu_slots, strict=True):
                runtime = TaskLocalRuntime(
                    self.campaign.root,
                    settings=self.campaign.settings,
                    task_key=self.task_key,
                    gpu_id=gpu_id,
                    gpu_slot=gpu_slot,
                    attempt_store=self.campaign.attempt_store,
                )
                self._runtimes.append(runtime)
                runtime.ensure()
            self._runtime = self._runtimes[0]
            self._start_budget_clock()
        except Exception:
            for runtime in reversed(self._runtimes):
                runtime.close()
            self._runtimes = []
            self._runtime = None
            for lease in reversed(acquired_leases):
                lease.release()
            if self._slot_lease is not None:
                self._slot_lease.release()
                self._slot_lease = None
            self._task_lease.release()
            raise
        self._write_status(
            "running",
            gpu_ids=list(self.gpu_ids),
            workers_per_gpu=int(
                self.campaign.settings["resources"]["workers_per_gpu"]
            ),
        )

    def evaluate_async(
        self,
        program_id: str,
        *,
        reset_ids: Iterable[str] | None = None,
        mode_ids: Iterable[str] | None = None,
        partition: str | None = None,
        force: bool = False,
    ) -> EvaluationHandle:
        if self.campaign.schema_version >= 2:
            raise RuntimeError(
                "v2 campaigns evaluate readable candidates through open_failure_mode()"
            )
        self.ensure_runtime()
        assert self.programs is not None
        program = self.programs.get(program_id)
        candidates = self.jobs(mode_ids=mode_ids, partition=partition)
        selected_ids = set(reset_ids or ())
        if selected_ids:
            candidates = [
                job
                for job in candidates
                if str(job["job_id"]) in selected_ids or str(job["source_job_id"]) in selected_ids
            ]
            found = {
                value
                for job in candidates
                for value in (str(job["job_id"]), str(job["source_job_id"]))
                if value in selected_ids
            }
            missing = sorted(selected_ids - found)
            if missing:
                raise KeyError(f"reset IDs are not part of task {self.task_key}: {missing}")
        if not candidates:
            raise ValueError("evaluation selected no prepared reset jobs")
        assert self._runtime is not None
        if len(self._runtimes) != 1:
            raise RuntimeError("legacy evaluate_async supports exactly one GPU per task")
        return self._runtime.evaluate(program, candidates, force=force)

    def evaluate(self, program_id: str, **selectors: Any) -> list[dict[str, Any]]:
        return self.evaluate_async(program_id, **selectors).results()

    def open_failure_mode(self, mode_id: str) -> FailureModeSession:
        return FailureModeSession(self, mode_id)

    def _evaluate_program(
        self,
        program: ProgramRecord,
        jobs: Iterable[dict[str, Any]],
        *,
        execution_mode: str = "canonical",
    ) -> list[dict[str, Any]]:
        if self.campaign.schema_version < 2:
            raise RuntimeError("readable candidate evaluation requires a v2 campaign")
        selected_jobs = [dict(job) for job in jobs]
        cached_results: list[dict[str, Any]] = []
        if (
            execution_mode == "canonical"
            and isinstance(self.campaign.attempt_store, ReadableAttemptStore)
        ):
            store = self.campaign.attempt_store
            uncached_jobs: list[dict[str, Any]] = []
            for job in selected_jobs:
                cached = store.committed(program, job)
                if cached is None:
                    uncached_jobs.append(job)
                    continue
                path = store.cached_result_path(program, job)
                cached_results.append(
                    {
                        **cached,
                        "attempt_path": str(path.parent),
                        "cached": True,
                        "canonical": True,
                    }
                )
            selected_jobs = uncached_jobs
        if not selected_jobs:
            return sorted(
                cached_results,
                key=lambda item: str(item.get("job_id", "")),
            )

        self.ensure_runtime()
        pending_jobs = iter(selected_jobs)
        pending: dict[Future[dict[str, Any]], TaskLocalRuntime] = {}
        workers_per_gpu = int(self.campaign.settings["resources"]["workers_per_gpu"])

        def submit_next(runtime: TaskLocalRuntime) -> bool:
            try:
                job = next(pending_jobs)
            except StopIteration:
                return False
            handle = runtime.evaluate(
                program,
                [job],
                execution_mode=execution_mode,
            )
            pending[handle.futures[0]] = runtime
            return True

        for runtime in self._runtimes:
            for _ in range(workers_per_gpu):
                if not submit_next(runtime):
                    break

        results: list[dict[str, Any]] = cached_results
        while pending:
            completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in completed:
                runtime = pending.pop(future)
                results.append(future.result())
                submit_next(runtime)
        return sorted(results, key=lambda item: str(item.get("job_id", "")))

    def _coverage_summary(self) -> dict[str, Any]:
        mode_ids = sorted({str(job["failure_mode_id"]) for job in self._jobs})
        modes = [self.open_failure_mode(mode_id).state() for mode_id in mode_ids]
        active = sum(len(mode["active_failed_seed_ids"]) for mode in modes)
        solved = sum(len(mode["current_success_seed_ids"]) for mode in modes)
        abandoned = sum(len(mode["abandoned_seed_ids"]) for mode in modes)
        total = sum(len(mode["seed_ids"]) for mode in modes)
        if active:
            status = "in_progress"
        elif abandoned:
            status = "completed_partial"
        else:
            status = "completed"
        return {
            "status": status,
            "total": total,
            "solved": solved,
            "abandoned": abandoned,
            "active_failed": active,
            "finished_at": utc_now(),
        }

    def finish(self) -> dict[str, Any]:
        if self.campaign.schema_version < 2:
            self.close(status="completed")
            return {"status": "completed"}
        summary = self._coverage_summary()
        self.close(status=str(summary["status"]), summary=summary)
        return summary

    def report_problem(self, summary: str, *, details: str = "") -> Path:
        path = self.status_path.parent / "problem_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# Repair problem report: {self.task_key}\n\n{summary.strip()}\n\n{details.strip()}\n",
            encoding="utf-8",
        )
        self._write_status("problem_reported", problem_report=str(path))
        return path

    def close(self, *, status: str = "completed", **extra: Any) -> None:
        if self.closed:
            return
        if self.campaign.schema_version >= 2 and status in {"completed", "completed_partial"}:
            expected = self._coverage_summary()["status"]
            if status != expected:
                raise RuntimeError(
                    f"task coverage requires status {expected}, not caller-supplied {status}"
                )
        for runtime in reversed(self._runtimes):
            runtime.close()
        self._runtimes = []
        self._runtime = None
        for lease in reversed(self._leases):
            lease.release()
        if self._slot_lease is not None:
            self._slot_lease.release()
            self._slot_lease = None
        self._task_lease.release()
        self.closed = True
        self._write_status(status, **extra)

    def __enter__(self) -> TaskSession:
        self.ensure_runtime()
        return self

    def __exit__(self, *_: Any) -> None:
        if self.campaign.schema_version >= 2:
            self.finish()
        else:
            self.close()
