"""Build a dataset episode manifest from observation-only LIBERO repairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def materialize_observation_manifest(
    *,
    rollout_manifest: Path,
    repair_root: Path,
    output: Path,
    candidate_kind: str,
    task_index: int,
    repair_only_candidate_kinds: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Join one spliced repair and optional standalone repairs per failure."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output}")
    rollout = _read(rollout_manifest)
    rollout_info = rollout.get("rollout") or {}
    source_root = Path(str(rollout_info["dataset_root"])).resolve()
    failed_ids = [int(item) for item in rollout_info.get("failed_episode_indices", [])]
    if not failed_ids:
        raise ValueError("rollout manifest contains no failed episode indices")

    summary = _read(repair_root / "batch_summary.json")
    if summary.get("complete") is not True or summary.get("all_successful") is not True:
        raise ValueError("repair batch is not complete and fully successful")
    records = summary.get("records", [])

    def records_for_kind(kind: str) -> dict[int, dict[str, Any]]:
        matching = {
            int(row["episode_index"]): row
            for row in records
            if row.get("candidate_kind") == kind
        }
        if set(matching) != set(failed_ids):
            raise ValueError(
                f"selected {kind} repairs differ from rollout failures: "
                f"missing={sorted(set(failed_ids) - set(matching))}, "
                f"extra={sorted(set(matching) - set(failed_ids))}"
            )
        return matching

    selected_by_kind = {
        kind: records_for_kind(kind)
        for kind in (candidate_kind, *repair_only_candidate_kinds)
    }

    task = str(rollout["task"])
    episodes: list[dict[str, Any]] = []

    def make_episode(
        source_id: int, record: dict[str, Any], *, repair_only: bool
    ) -> dict[str, Any]:
        if not (
            record.get("success") is True
            and record.get("task_completed") is True
            and record.get("evaluator_task_completed") is True
            and record.get("sandbox_rc") == 0
        ):
            raise ValueError(f"repair {record.get('job_id')} is not admissible")
        job_id = str(record["job_id"])
        result_path = Path(str(record["result_path"])).resolve()
        result = _read(result_path)
        if not (
            result.get("task_completed") is True
            and result.get("evaluator_task_completed") is True
            and result.get("truncated") is False
        ):
            raise ValueError(
                f"repair result is not a successful terminal trajectory: {result_path}"
            )
        trajectory = Path(str(result["trajectory"])).resolve()
        prefix_parquet = source_root / f"data/chunk-000/episode_{source_id:06d}.parquet"
        prefix_videos = source_root / "videos/chunk-000"
        repair_videos = trajectory / "videos/chunk-000"
        descriptor = repair_root / "private_reset_descriptors" / job_id / "reset_descriptor.json"
        attempt = repair_root / "attempts" / job_id / "attempt_manifest.json"
        required = [
            trajectory / "data/chunk-000/episode_000000.parquet",
            repair_videos / "observation.images.agentview/episode_000000.mp4",
            repair_videos / "observation.images.wrist/episode_000000.mp4",
            descriptor,
            attempt,
            result_path,
        ]
        if not repair_only:
            required.extend(
                [
                    prefix_parquet,
                    prefix_videos / "image" / f"episode_{source_id:06d}.mp4",
                    prefix_videos / "wrist_image" / f"episode_{source_id:06d}.mp4",
                ]
            )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"episode {source_id} is missing artifacts: {missing}")
        entry: dict[str, Any] = {
            "source_episode_id": (
                f"{source_id}:{record['candidate_kind']}"
                if repair_only_candidate_kinds
                else source_id
            ),
            "task_index": task_index,
            "task": task,
            "repair": {
                "parquet": str(trajectory / "data/chunk-000/episode_000000.parquet"),
                "images": {
                    "image": {
                        "video": str(
                            repair_videos
                            / "observation.images.agentview/episode_000000.mp4"
                        )
                    },
                    "wrist_image": {
                        "video": str(
                            repair_videos
                            / "observation.images.wrist/episode_000000.mp4"
                        )
                    },
                },
            },
            "continuity": {
                "reset_descriptor": str(descriptor),
                "attempt_manifest": str(attempt),
                "result": str(result_path),
            },
            "metadata": {
                "source_rollout_episode_id": source_id,
                "job_id": job_id,
                "candidate_kind": record["candidate_kind"],
                "failure_mode_id": record.get("failure_mode_id"),
                "failure_category": record.get("failure_category"),
                "observation_only_repair": True,
                "private_truth_in_training_rows": False,
            },
        }
        if repair_only:
            entry["mode"] = "repair_only"
        else:
            entry["restart_frame"] = int(record["reset_frame_index"])
            entry["prefix"] = {
                "parquet": str(prefix_parquet),
                "images": {
                    "image": {
                        "video": str(
                            prefix_videos / "image" / f"episode_{source_id:06d}.mp4"
                        )
                    },
                    "wrist_image": {
                        "video": str(
                            prefix_videos
                            / "wrist_image"
                            / f"episode_{source_id:06d}.mp4"
                        )
                    },
                },
            }
        return entry

    for source_id in failed_ids:
        episodes.append(
            make_episode(
                source_id, selected_by_kind[candidate_kind][source_id], repair_only=False
            )
        )
    for kind in repair_only_candidate_kinds:
        for source_id in failed_ids:
            episodes.append(
                make_episode(source_id, selected_by_kind[kind][source_id], repair_only=True)
            )

    payload = {
        "schema_version": 1,
        "rollout_manifest": str(rollout_manifest.resolve()),
        "repair_root": str(repair_root.resolve()),
        "candidate_kind": candidate_kind,
        "repair_only_candidate_kinds": list(repair_only_candidate_kinds),
        "episodes": episodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-manifest", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-kind", default="window_start")
    parser.add_argument(
        "--repair-only-candidate-kind",
        action="append",
        default=[],
        help="Append this successful repair kind as standalone repair_only episodes.",
    )
    parser.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    payload = materialize_observation_manifest(
        rollout_manifest=args.rollout_manifest.resolve(),
        repair_root=args.repair_root.resolve(),
        output=args.output.resolve(),
        candidate_kind=args.candidate_kind,
        task_index=args.task_index,
        repair_only_candidate_kinds=tuple(args.repair_only_candidate_kind),
    )
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "episodes": len(payload["episodes"])},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
