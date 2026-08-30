"""Dependency-light backend used to verify the repair scheduler itself."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .util import atomic_write_json, atomic_write_text, utc_now


def _fake_directives(program_path: str | Path) -> dict[str, Any]:
    tree = ast.parse(Path(program_path).read_text(encoding="utf-8"), filename=str(program_path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {
                "RESULT",
                "FAKE_SUCCESS_SEEDS",
                "FAKE_OUTCOMES",
            }:
                values[target.id] = ast.literal_eval(value)
    return values


def execute_fake_job(
    job: dict[str, Any], *, program_path: str | Path, attempt_dir: str | Path
) -> dict[str, Any]:
    output = Path(attempt_dir)
    output.mkdir(parents=True, exist_ok=True)
    directives = _fake_directives(program_path)
    seed_id = str(job.get("source_job_id", job["job_id"]))
    outcomes = directives.get("FAKE_OUTCOMES", {})
    if isinstance(outcomes, dict) and seed_id in outcomes:
        outcome = str(outcomes[seed_id])
    elif "FAKE_SUCCESS_SEEDS" in directives:
        outcome = "success" if seed_id in set(directives["FAKE_SUCCESS_SEEDS"]) else "policy_failure"
    else:
        success = bool(directives.get("RESULT", not bool(job.get("fake_failure", False))))
        outcome = "success" if success else "policy_failure"
    success = outcome == "success"
    result = {
        "schema_version": 1,
        "outcome": outcome,
        "success": success,
        "task_completed": success,
        "job_id": job["job_id"],
        "task_key": job["task_key"],
        "failure_mode_id": job["failure_mode_id"],
        "finished_at": utc_now(),
    }
    atomic_write_text(output / "wide.mp4", f"fake wide {seed_id}\n")
    atomic_write_text(output / "wrist.mp4", f"fake wrist {seed_id}\n")
    atomic_write_json(output / "trajectory.json", {"states": [], "actions": []})
    atomic_write_json(output / "terminal_observation.json", {"task_completed": success})
    atomic_write_text(output / "stdout.log", "")
    atomic_write_text(output / "stderr.log", "")
    return result
