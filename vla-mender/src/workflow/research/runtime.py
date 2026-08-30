"""Task-local fixed worker pool for repair rollouts."""

from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .artifacts import AttemptStore, ProgramRecord, ReadableAttemptStore
from .services import TaskServiceManager
from .util import (
    add_loopback_no_proxy,
    atomic_write_json,
    locked_file,
    read_json,
    safe_component,
    sha256_file,
    utc_now,
)


def _python_path(project: dict[str, Any]) -> str:
    additions = [
        str(project["source_root"]),
        str(Path(str(project["knowledge_root"])).resolve().parent),
    ]
    inherited = os.environ.get("PYTHONPATH", "")
    if inherited:
        additions.append(inherited)
    return os.pathsep.join(additions)


def _timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _preflight_contract(
    project: dict[str, Any], environment: dict[str, Any]
) -> dict[str, Any]:
    """Return the explicit environment fields that make a preflight reusable."""

    return {
        "python": str(environment["python"]),
        "working_directory": str(environment["working_directory"]),
        "libero_root": str(environment["libero_root"]),
        "source_root": str(project["source_root"]),
        "knowledge_root": str(project["knowledge_root"]),
        "python_path": _python_path(project),
        "extra_env": {
            str(key): str(value)
            for key, value in sorted(environment.get("env", {}).items())
        },
    }


@dataclass
class EvaluationHandle:
    futures: list[Future[dict[str, Any]]]

    def done(self) -> bool:
        return all(future.done() for future in self.futures)

    def results(self) -> list[dict[str, Any]]:
        values = [future.result() for future in as_completed(self.futures)]
        return sorted(values, key=lambda item: str(item.get("job_id", "")))


class TaskLocalRuntime:
    """One task owns one GPU, one service group, and a fixed rollout pool."""

    def __init__(
        self,
        campaign_root: str | Path,
        *,
        settings: dict[str, Any],
        task_key: str,
        gpu_id: int,
        gpu_slot: int,
        attempt_store: AttemptStore | ReadableAttemptStore,
    ):
        self.campaign_root = Path(campaign_root).resolve()
        self.settings = settings
        self.task_key = task_key
        self.gpu_id = int(gpu_id)
        self.gpu_slot = int(gpu_slot)
        self.attempt_store = attempt_store
        resources = dict(settings["resources"])
        service = dict(resources["services"])
        project = dict(settings["project"])
        environment = dict(settings["environment"])
        self.runtime_settings = dict(settings["runtime"])
        self.environment_settings = environment
        self.project_settings = project
        self.pool = ThreadPoolExecutor(
            max_workers=int(resources["workers_per_gpu"]),
            thread_name_prefix=f"repair-{safe_component(task_key)}-gpu-{self.gpu_id}",
        )
        self.service_manager = TaskServiceManager(
            self.campaign_root,
            project_root=project["root"],
            python=environment["python"],
            gpu_id=self.gpu_id,
            gpu_slot=self.gpu_slot,
            profile=str(service["profile"]),
            port_base=int(service["port_base"]),
            port_stride=int(service["port_stride"]),
            manage=bool(service["manage"]),
            keep_alive=bool(service["keep_alive"]),
            startup_timeout_s=float(service["startup_timeout_s"]),
            extra_env={str(key): str(value) for key, value in environment.get("env", {}).items()},
        )
        self.service_environment: dict[str, str] = {}
        self._closed = False

    def ensure(self) -> None:
        if self._closed:
            raise RuntimeError("task runtime is closed")
        if str(self.runtime_settings["backend"]) == "libero" and not self.service_environment:
            self._ensure_libero_preflight()
            self.service_environment = self.service_manager.ensure()

    def _ensure_libero_preflight(self) -> None:
        state_path = self.campaign_root / "runtime" / "environment_preflight.json"
        lock_path = self.campaign_root / "runtime" / "environment_preflight.lock"
        contract = _preflight_contract(
            self.project_settings,
            self.environment_settings,
        )
        with locked_file(lock_path):
            if state_path.is_file():
                state = read_json(state_path)
                if state.get("contract") == contract and state.get("status") == "ok":
                    return
            environment = dict(os.environ)
            environment.update(
                {
                    str(key): str(value)
                    for key, value in self.environment_settings.get("env", {}).items()
                }
            )
            add_loopback_no_proxy(environment)
            environment["PYTHONPATH"] = _python_path(self.project_settings)
            environment["VLA_MENDER_LIBERO_ROOT"] = str(self.environment_settings["libero_root"])
            check = "\n".join(
                [
                    "import libero, robosuite, yaml",
                    "from workflow.research.libero_backend import execute_repair_job",
                    "from knowledge.api.franka.libero_osc_reduced_skill_library "
                    "import FrankaLiberoApiReducedOscSkillLibrary",
                    "from knowledge.api.franka.libero_reduced_skill_library "
                    "import FrankaLiberoApiReducedSkillLibrary",
                ]
            )
            try:
                completed = subprocess.run(
                    [str(self.environment_settings["python"]), "-c", check],
                    cwd=str(self.environment_settings["working_directory"]),
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=120.0,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "repair Python environment preflight timed out after 120 seconds"
                ) from exc
            if completed.returncode != 0:
                raise RuntimeError(
                    "repair Python environment is missing native LIBERO/knowledge dependencies:\n"
                    + completed.stderr[-8000:]
                )
            atomic_write_json(
                state_path,
                {
                    "schema_version": 1,
                    "status": "ok",
                    "contract": contract,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "checked_at": utc_now(),
                },
            )

    def evaluate(
        self,
        program: ProgramRecord,
        jobs: Iterable[dict[str, Any]],
        *,
        force: bool = False,
        execution_mode: str | None = None,
    ) -> EvaluationHandle:
        self.ensure()
        mode = execution_mode or ("stability" if force else "canonical")
        if mode not in {"canonical", "stability", "diagnostic_retry"}:
            raise ValueError(f"unknown evaluation execution mode: {mode}")
        futures = [
            self.pool.submit(
                self._run_one,
                program,
                dict(job),
                force=force,
                execution_mode=mode,
            )
            for job in jobs
        ]
        return EvaluationHandle(futures)

    def _run_one(
        self,
        program: ProgramRecord,
        job: dict[str, Any],
        *,
        force: bool,
        execution_mode: str,
    ) -> dict[str, Any]:
        if isinstance(self.attempt_store, ReadableAttemptStore):
            return self._run_one_readable(
                program, job, execution_mode=execution_mode
            )
        job_id = str(job["job_id"])
        reusable = bool(self.runtime_settings["resume"]) and not force
        cached = self.attempt_store.committed(self.task_key, str(job["job_id"]), program.sha256)
        if cached is not None and reusable:
            path = self.attempt_store.result_path(self.task_key, job_id, program.sha256)
            return {**cached, "attempt_path": str(path.parent), "cached": True}
        execution_lock = self.attempt_store.result_path(
            self.task_key, job_id, program.sha256
        ).parent / "execution.lock"
        with locked_file(execution_lock):
            cached = self.attempt_store.committed(self.task_key, job_id, program.sha256)
            if cached is not None and reusable:
                path = self.attempt_store.result_path(self.task_key, job_id, program.sha256)
                return {**cached, "attempt_path": str(path.parent), "cached": True}
            return self._execute_one(
                program,
                job,
                canonical=reusable,
                execution_mode=execution_mode,
            )

    def _run_one_readable(
        self,
        program: ProgramRecord,
        job: dict[str, Any],
        *,
        execution_mode: str,
    ) -> dict[str, Any]:
        store = self.attempt_store
        assert isinstance(store, ReadableAttemptStore)
        reusable = execution_mode == "canonical"
        cached = store.committed(program, job) if reusable else None
        if cached is not None:
            path = store.cached_result_path(program, job)
            return {**cached, "attempt_path": str(path.parent), "cached": True}
        with locked_file(store.execution_lock(program, job)):
            cached = store.committed(program, job) if reusable else None
            if cached is not None:
                path = store.cached_result_path(program, job)
                return {**cached, "attempt_path": str(path.parent), "cached": True}
            return self._execute_one(
                program,
                job,
                canonical=execution_mode in {"canonical", "diagnostic_retry"},
                execution_mode=execution_mode,
            )

    def _execute_one(
        self,
        program: ProgramRecord,
        job: dict[str, Any],
        *,
        canonical: bool,
        execution_mode: str,
    ) -> dict[str, Any]:
        attempts = int(self.runtime_settings["infrastructure_retries"]) + 1
        last_result: dict[str, Any] | None = None
        for retry_index in range(attempts):
            if isinstance(self.attempt_store, ReadableAttemptStore):
                temporary = self.attempt_store.begin(program, job)
            else:
                temporary = self.attempt_store.begin(
                    self.task_key, str(job["job_id"]), program.sha256
                )
            request = {
                "schema_version": 1,
                "backend": str(self.runtime_settings["backend"]),
                "job": job,
                "program_path": str(program.path),
                "program_sha256": program.sha256,
                "attempt_dir": str(temporary),
                "libero_root": str(self.environment_settings["libero_root"]),
                "max_steps": int(self.runtime_settings["max_steps"]),
                "retry_index": retry_index,
                "created_at": utc_now(),
            }
            request_path = temporary / "worker_request.json"
            atomic_write_json(request_path, request)
            environment = dict(os.environ)
            environment.update(
                {
                    str(key): str(value)
                    for key, value in self.environment_settings.get("env", {}).items()
                }
            )
            environment.update(self.service_environment)
            add_loopback_no_proxy(environment)
            environment["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            environment["PYTHONPATH"] = _python_path(self.project_settings)
            environment["VLA_MENDER_LIBERO_ROOT"] = str(self.environment_settings["libero_root"])
            command = [
                str(self.environment_settings["python"]),
                "-m",
                "workflow.research.worker",
                "--request",
                str(request_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(self.environment_settings["working_directory"]),
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=float(self.runtime_settings["job_timeout_s"]),
                    check=False,
                )
                (temporary / "worker_stdout.log").write_text(completed.stdout, encoding="utf-8")
                (temporary / "worker_stderr.log").write_text(completed.stderr, encoding="utf-8")
                result_path = temporary / "worker_result.json"
                if result_path.is_file():
                    last_result = read_json(result_path)
                else:
                    last_result = {
                        "schema_version": 1,
                        "outcome": "infrastructure_failure",
                        "success": False,
                        "job_id": job["job_id"],
                        "task_key": self.task_key,
                        "failure_mode_id": job["failure_mode_id"],
                        "error": (
                            f"worker exited {completed.returncode} without worker_result.json: "
                            f"{completed.stderr[-4000:]}"
                        ),
                        "finished_at": utc_now(),
                    }
            except subprocess.TimeoutExpired as exc:
                (temporary / "worker_stdout.log").write_text(
                    _timeout_output(exc.stdout), encoding="utf-8"
                )
                (temporary / "worker_stderr.log").write_text(
                    _timeout_output(exc.stderr), encoding="utf-8"
                )
                last_result = {
                    "schema_version": 1,
                    "outcome": "infrastructure_failure",
                    "success": False,
                    "job_id": job["job_id"],
                    "task_key": self.task_key,
                    "failure_mode_id": job["failure_mode_id"],
                    "error": (
                        f"worker timeout after {self.runtime_settings['job_timeout_s']} seconds"
                    ),
                    "finished_at": utc_now(),
                }
            last_result["program_id"] = program.id
            last_result["program_sha256"] = program.sha256
            last_result["source_job_id"] = str(job.get("source_job_id", job["job_id"]))
            last_result["seed_slug"] = str(job.get("seed_slug", ""))
            last_result["retry_index"] = retry_index
            last_result["gpu_id"] = self.gpu_id
            if isinstance(self.attempt_store, ReadableAttemptStore):
                last_result["evidence_sha256"] = {
                    name: sha256_file(temporary / name)
                    for name in (
                        "wide.mp4",
                        "wrist.mp4",
                        "trajectory.json",
                        "terminal_observation.json",
                    )
                    if (temporary / name).is_file()
                }
                # These two files are IPC envelopes. Their durable facts live
                # in candidate/program metadata and canonical result.json.
                (temporary / "worker_request.json").unlink(missing_ok=True)
                (temporary / "worker_result.json").unlink(missing_ok=True)
                for log_name in ("worker_stdout.log", "worker_stderr.log"):
                    log_path = temporary / log_name
                    if log_path.is_file() and log_path.stat().st_size == 0:
                        log_path.unlink()
                outcome = str(last_result.get("outcome"))
                if outcome in {"reset_failure", "infrastructure_failure"}:
                    result_path = self.attempt_store.commit_diagnostic(
                        temporary,
                        program=program,
                        job=job,
                        result=last_result,
                    )
                    if outcome == "infrastructure_failure" and retry_index + 1 < attempts:
                        continue
                    return {
                        **last_result,
                        "attempt_path": str(result_path.parent),
                        "cached": False,
                        "canonical": False,
                    }
            if str(last_result.get("outcome")) != "infrastructure_failure":
                if isinstance(self.attempt_store, ReadableAttemptStore):
                    result_path = self.attempt_store.commit(
                        temporary,
                        program=program,
                        job=job,
                        result=last_result,
                        canonical=canonical,
                    )
                else:
                    result_path = self.attempt_store.commit(
                        temporary,
                        task_key=self.task_key,
                        job_id=str(job["job_id"]),
                        program_sha256=program.sha256,
                        result=last_result,
                        canonical=canonical,
                    )
                return {
                    **last_result,
                    "attempt_path": str(result_path.parent),
                    "cached": False,
                    "canonical": canonical,
                    "execution_mode": execution_mode,
                }
            shutil.rmtree(temporary, ignore_errors=True)
        assert last_result is not None
        return {**last_result, "cached": False}

    def close(self) -> None:
        if self._closed:
            return
        self.pool.shutdown(wait=True, cancel_futures=False)
        self.service_manager.stop()
        self._closed = True
