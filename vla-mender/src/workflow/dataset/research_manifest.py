"""Adapt a completed standalone repair quality selection to dataset episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def materialize_research_manifest(
    *,
    pre_repair_run: Path,
    repair_run: Path,
    selection_manifest: Path,
    output: Path,
    task_index: int,
) -> dict[str, Any]:
    """Create ordered prefix-plus-repair sources from retained quality rows."""

    pre_repair_run = pre_repair_run.expanduser().resolve()
    repair_run = repair_run.expanduser().resolve()
    selection_manifest = _require_file(selection_manifest, "selection manifest")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite episode manifest: {output}")

    resolved_path = _require_file(repair_run / "repair_resolved.yaml", "resolved repair config")
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(resolved, dict):
        raise ValueError(f"invalid resolved repair config: {resolved_path}")
    tasks = resolved.get("resolved_tasks") or []
    if len(tasks) != 1:
        raise ValueError("research dataset adapter currently requires exactly one repaired task")
    task = tasks[0]
    if Path(str(task["run_root"])).resolve() != pre_repair_run:
        raise ValueError("repair_resolved.yaml points to a different pre-repair run")
    description = str(task["description"])
    jobs = {str(row["source_job_id"]): row for row in resolved.get("jobs", [])}

    handoff_path = _require_file(
        pre_repair_run / "repair_handoff/manifest.json", "repair handoff"
    )
    handoff = _read_json(handoff_path)
    if handoff.get("complete") is not True or (handoff.get("summary") or {}).get(
        "all_replays_verified"
    ) is not True:
        raise ValueError(f"pre-repair handoff is not complete and verified: {handoff_path}")
    resets = {str(row["job_id"]): row for row in handoff.get("resets", [])}

    selection = _read_json(selection_manifest)
    rows = selection.get("trajectories") or []
    if not rows:
        raise ValueError("selection manifest contains no trajectories")
    retained = [row for row in rows if row.get("decision") == "retain"]
    if len(retained) != len(rows):
        raise ValueError("selection manifest for dataset construction must contain retained rows only")
    declared = (selection.get("summary") or {}).get("retained_count")
    if declared is not None and int(declared) != len(retained):
        raise ValueError("selection retained_count does not match trajectory rows")

    episodes: list[dict[str, Any]] = []
    seen_episodes: set[int] = set()
    for position, row in enumerate(retained):
        episode = int(row["episode"])
        if episode in seen_episodes:
            raise ValueError(f"selection repeats source episode {episode}")
        seen_episodes.add(episode)
        source_job_id = str(row["source_job_id"])
        if source_job_id not in jobs or source_job_id not in resets:
            raise ValueError(f"selection references unknown repair seed {source_job_id}")
        job = jobs[source_job_id]
        reset = resets[source_job_id]
        restart = int(row["reset_frame"])
        if int(job["episode_index"]) != episode or int(job["reset_frame_index"]) != restart:
            raise ValueError(f"selection/job identity mismatch for {source_job_id}")
        if int(reset["episode_index"]) != episode or int(reset["reset_frame_index"]) != restart:
            raise ValueError(f"selection/handoff identity mismatch for {source_job_id}")

        result_path = _require_file(Path(str(row["selected_result_ref"])), "selected result")
        try:
            result_path.relative_to(repair_run)
        except ValueError as exc:
            raise ValueError(f"selected result is outside repair run: {result_path}") from exc
        result = _read_json(result_path)
        if not (
            result.get("source_job_id") == source_job_id
            and result.get("success") is True
            and result.get("task_completed") is True
        ):
            raise ValueError(f"selected result is not an admissible success: {result_path}")
        selection_hashes = row.get("evidence_sha256") or {}
        result_hashes = result.get("evidence_sha256") or {}
        if selection_hashes != result_hashes:
            raise ValueError(f"selection/result evidence hashes differ: {result_path}")
        result_dir = result_path.parent
        trajectory = _require_file(result_dir / "trajectory.json", "repair trajectory")
        repair_wide = _require_file(result_dir / "wide.mp4", "repair wide video")
        repair_wrist = _require_file(result_dir / "wrist.mp4", "repair wrist video")

        prefix_trajectory = _require_file(
            pre_repair_run / f"rollout/episodes/episode_{episode:06d}.json",
            "source rollout trajectory",
        )
        prefix_wide = _require_file(
            pre_repair_run / f"rollout/videos/episode_{episode:06d}_wide.mp4",
            "source rollout wide video",
        )
        prefix_wrist = _require_file(
            pre_repair_run / f"rollout/videos/episode_{episode:06d}_wrist.mp4",
            "source rollout wrist video",
        )
        episodes.append(
            {
                "source_episode_id": episode,
                "restart_frame": restart,
                "task_index": int(task_index),
                "task": description,
                "prefix": {
                    "trajectory": str(prefix_trajectory),
                    "images": {
                        "image": {"video": str(prefix_wide)},
                        "wrist_image": {"video": str(prefix_wrist)},
                    },
                },
                "repair": {
                    "trajectory": str(trajectory),
                    "images": {
                        "image": {"video": str(repair_wide)},
                        "wrist_image": {"video": str(repair_wrist)},
                    },
                },
                "continuity": {
                    "handoff_manifest": str(handoff_path),
                    "result": str(result_path),
                    "source_job_id": source_job_id,
                },
                "metadata": {
                    "selection_position": position,
                    "source_job_id": source_job_id,
                    "scene_model_seed": int(job["scene_model_seed"]),
                    "failure_mode_id": str(job["failure_mode_id"]),
                    "selected_variant": row.get("selected_variant"),
                    "selected_program_id": row.get("selected_program_id"),
                    "selected_program_sha256": row.get("selected_program_sha256"),
                    "quality_selection_manifest": str(selection_manifest),
                    "quality_checks": row.get("checks") or {},
                },
            }
        )

    payload = {
        "schema_version": 1,
        "adapter": "vla_mender.research_quality_selection",
        "pre_repair_run": str(pre_repair_run),
        "repair_run": str(repair_run),
        "selection_manifest": str(selection_manifest),
        "task_index": int(task_index),
        "task": description,
        "episodes": episodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-repair-run", type=Path, required=True)
    parser.add_argument("--repair-run", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=0)
    args = parser.parse_args()
    payload = materialize_research_manifest(
        pre_repair_run=args.pre_repair_run,
        repair_run=args.repair_run,
        selection_manifest=args.selection_manifest,
        output=args.output,
        task_index=args.task_index,
    )
    print(json.dumps({"output": str(args.output.resolve()), "episodes": len(payload["episodes"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
