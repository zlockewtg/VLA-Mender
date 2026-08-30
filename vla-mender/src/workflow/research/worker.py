"""Isolated worker entry point for one exact-reset repair attempt."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from .util import atomic_write_json, read_json, utc_now


def execute_request(request: dict[str, Any]) -> dict[str, Any]:
    backend = str(request["backend"])
    attempt_dir = Path(str(request["attempt_dir"]))
    job = dict(request["job"])
    if backend == "fake":
        from .fake_backend import execute_fake_job

        return execute_fake_job(
            job,
            program_path=str(request["program_path"]),
            attempt_dir=attempt_dir,
        )
    if backend == "libero":
        from .libero_backend import execute_repair_job

        return execute_repair_job(
            job,
            program_path=str(request["program_path"]),
            attempt_dir=attempt_dir,
            libero_root=str(request["libero_root"]),
            max_steps=int(request["max_steps"]),
        )
    raise ValueError(f"unsupported repair backend: {backend}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request.resolve())
    attempt_dir = Path(str(request["attempt_dir"]))
    result_path = attempt_dir / "worker_result.json"
    try:
        result = execute_request(request)
        result["worker_outcome"] = "completed"
        atomic_write_json(result_path, result)
        return 0
    except Exception as exc:
        outcome = (
            "reset_failure"
            if exc.__class__.__name__ == "ResetVerificationError"
            else "infrastructure_failure"
        )
        result = {
            "schema_version": 1,
            "outcome": outcome,
            "success": False,
            "worker_outcome": "failed",
            "job_id": request.get("job", {}).get("job_id", ""),
            "task_key": request.get("job", {}).get("task_key", ""),
            "failure_mode_id": request.get("job", {}).get("failure_mode_id", ""),
            "error": traceback.format_exc(),
            "finished_at": utc_now(),
        }
        atomic_write_json(result_path, result)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
