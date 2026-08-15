"""Adapter from VLA-Mender LIBERO eval/repair artifacts to the generic manifest."""

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


def materialize_manifest(
    *,
    eval_root: Path,
    selection_manifest: Path,
    repair_root: Path,
    prefix_dataset: Path,
    output: Path,
    prefix_camera_paths: dict[str, str],
    repair_camera_paths: dict[str, str],
) -> dict[str, Any]:
    """Join explicit selection order, failure windows, and successful repairs."""

    selection = _read(selection_manifest)
    selected_rows = selection.get("episodes") or []
    if not selected_rows:
        raise ValueError("selection manifest contains no episodes")
    selected_ids = [
        row.get("source_global_episode_index", row.get("dataset_episode_index"))
        for row in selected_rows
    ]
    if any(item is None for item in selected_ids):
        raise ValueError("selection episodes lack source-global identities")
    windows = {
        int(row["dataset_episode_index"]): row
        for row in _read(eval_root / "failure_diagnosis/failure_windows.json")["windows"]
    }
    summary = _read(repair_root / "batch_summary.json")
    if summary.get("all_successful") is not True:
        raise ValueError("repair batch is not fully successful")
    repairs = {int(row["dataset_episode_index"]): row for row in summary["records"]}
    prefix_build = _read(prefix_dataset / "meta/build_manifest.json")
    prefixes = {
        int(row.get("source_global_episode_index", row.get("source_episode_id"))): row
        for row in prefix_build["episodes"]
    }
    eval_contract = _read(eval_root / "eval_contract.json")
    source_pythons = {
        str(worker["command"][0])
        for worker in eval_contract.get("workers", [])
        if worker.get("command")
    }
    if len(source_pythons) != 1:
        raise ValueError(f"eval contract does not use one source Python: {source_pythons}")
    expected_source_python = next(iter(source_pythons))

    episodes: list[dict[str, Any]] = []
    for source_id_value in selected_ids:
        source_id = int(source_id_value)
        window = windows[source_id]
        record = repairs[source_id]
        if not (
            record.get("success")
            and record.get("task_completed")
            and record.get("evaluator_task_completed")
            and record.get("sandbox_rc") == 0
            and record.get("truncated") is False
        ):
            raise ValueError(f"repair episode {source_id} is not admissible")
        result_path = Path(record["result_path"]).resolve()
        result = _read(result_path)
        trajectory = Path(result["trajectory"]).resolve()
        prefix_entry = prefixes[source_id]
        prefix_parquet = (prefix_dataset / prefix_entry["parquet"]).resolve()
        worker = int(window["worker"])
        local_episode = int(window["worker_local_episode_index"])
        worker_dataset = eval_root / f"workers/worker_{worker:02d}/dataset"
        task_index = int(prefix_entry["task_index"])
        task = str(prefix_entry["task"])
        prefix_images = {
            name: {
                "column": name,
                "continuity_video": str(
                    worker_dataset
                    / relative.format(episode_index=local_episode)
                ),
            }
            for name, relative in prefix_camera_paths.items()
        }
        repair_images = {
            name: {"video": str(trajectory / relative)}
            for name, relative in repair_camera_paths.items()
        }
        episodes.append(
            {
                "source_episode_id": source_id,
                "restart_frame": int(prefix_entry["vla_prefix_length"]),
                "task_index": task_index,
                "task": task,
                "prefix": {"parquet": str(prefix_parquet), "images": prefix_images},
                "repair": {
                    "parquet": str(trajectory / "data/chunk-000/episode_000000.parquet"),
                    "images": repair_images,
                },
                "continuity": {
                    "reset_descriptor": str(Path(record["reset_descriptor"]).resolve()),
                    "attempt_manifest": str(result_path.parent / "attempt_manifest.json"),
                    "result": str(result_path),
                    "expected_source_python": expected_source_python,
                },
                "metadata": {
                    "worker": worker,
                    "worker_local_episode_index": local_episode,
                    "failure_mode_id": window.get("mode_id"),
                    "scene_model_seed": (window.get("simulator_replay") or {}).get("scene_model_seed"),
                },
            }
        )
    payload = {
        "schema_version": 1,
        "selection_manifest": str(selection_manifest.resolve()),
        "repair_root": str(repair_root.resolve()),
        "prefix_dataset": str(prefix_dataset.resolve()),
        "episodes": episodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _mapping(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("camera mapping must be OUTPUT_COLUMN=RELATIVE_PATH")
    return name, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--prefix-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefix-camera",
        type=_mapping,
        action="append",
        default=[],
        metavar="COLUMN=PATH",
    )
    parser.add_argument(
        "--repair-camera",
        type=_mapping,
        action="append",
        default=[],
        metavar="COLUMN=PATH",
    )
    args = parser.parse_args()
    if not args.prefix_camera or not args.repair_camera:
        parser.error("at least one --prefix-camera and --repair-camera are required")
    payload = materialize_manifest(
        eval_root=args.eval_root.resolve(),
        selection_manifest=args.selection_manifest.resolve(),
        repair_root=args.repair_root.resolve(),
        prefix_dataset=args.prefix_dataset.resolve(),
        output=args.output.resolve(),
        prefix_camera_paths=dict(args.prefix_camera),
        repair_camera_paths=dict(args.repair_camera),
    )
    print(json.dumps({"output": str(args.output.resolve()), "episodes": len(payload["episodes"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
