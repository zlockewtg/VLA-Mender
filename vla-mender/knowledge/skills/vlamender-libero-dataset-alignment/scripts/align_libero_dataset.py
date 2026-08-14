#!/usr/bin/env python3
"""Convert video-backed LeRobot data to the local LIBERO v2.0 Parquet contract."""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


REQUIRED_COLUMNS = [
    "image",
    "wrist_image",
    "state",
    "actions",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]
SOURCE_NUMERIC_COLUMNS = [
    "state",
    "actions",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]
IMAGE_KEYS = ("image", "wrist_image")
ROW_GROUP_SIZE = 100


def _json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values)
    )


def _episode_path(root: Path, episode_index: int) -> Path:
    return (
        root
        / "data"
        / f"chunk-{episode_index // 1000:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def _video_path(root: Path, episode_index: int, key: str) -> Path:
    return (
        root
        / "videos"
        / f"chunk-{episode_index // 1000:03d}"
        / key
        / f"episode_{episode_index:06d}.mp4"
    )


def _first_parquet(root: Path) -> Path:
    paths = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Parquet episodes under {root / 'data'}")
    return paths[0]


def _reference_schema(reference: Path) -> pa.Schema:
    schema = pq.ParquetFile(_first_parquet(reference)).schema_arrow
    if schema.names != REQUIRED_COLUMNS:
        raise ValueError(
            f"reference column order differs from contract: {schema.names}"
        )
    if schema.metadata is None or b"huggingface" not in schema.metadata:
        raise ValueError("reference schema lacks Hugging Face feature metadata")
    return schema


def _task_catalog(reference: Path) -> tuple[list[dict[str, Any]], dict[int, str]]:
    rows = _json_lines(reference / "meta" / "tasks.jsonl")
    mapping = {int(row["task_index"]): str(row["task"]) for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("reference task indices are not unique")
    return rows, mapping


@dataclass
class VectorStats:
    width: int

    def __post_init__(self) -> None:
        self.count = 0
        self.total = np.zeros(self.width, dtype=np.float64)
        self.total_sq = np.zeros(self.width, dtype=np.float64)
        self.minimum = np.full(self.width, np.inf, dtype=np.float64)
        self.maximum = np.full(self.width, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values)
        if array.ndim == 1:
            array = array.reshape(-1, self.width)
        if array.ndim != 2 or array.shape[1] != self.width:
            raise ValueError(
                f"stats expected (*,{self.width}), received {array.shape}"
            )
        array = array.astype(np.float64, copy=False)
        self.count += int(array.shape[0])
        self.total += array.sum(axis=0)
        self.total_sq += np.square(array).sum(axis=0)
        self.minimum = np.minimum(self.minimum, array.min(axis=0))
        self.maximum = np.maximum(self.maximum, array.max(axis=0))

    def result(self) -> dict[str, list[float]]:
        if self.count <= 0:
            raise ValueError("cannot finalize empty statistics")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        return {
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "max": self.maximum.tolist(),
            "min": self.minimum.tolist(),
        }


@dataclass
class ImageStats:
    def __post_init__(self) -> None:
        self.count = 0
        self.total = np.zeros(3, dtype=np.float64)
        self.total_sq = np.zeros(3, dtype=np.float64)
        self.minimum = np.full(3, np.inf, dtype=np.float64)
        self.maximum = np.full(3, -np.inf, dtype=np.float64)

    def update_bgr(self, frame: np.ndarray) -> None:
        if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 256x256x3 frame, received {frame.shape}")
        rgb = frame[:, :, ::-1].astype(np.float64) / 255.0
        self.count += int(rgb.shape[0] * rgb.shape[1])
        self.total += rgb.sum(axis=(0, 1))
        self.total_sq += np.square(rgb).sum(axis=(0, 1))
        self.minimum = np.minimum(self.minimum, rgb.min(axis=(0, 1)))
        self.maximum = np.maximum(self.maximum, rgb.max(axis=(0, 1)))

    def result(self) -> dict[str, list[list[list[float]]]]:
        if self.count <= 0:
            raise ValueError("cannot finalize empty image statistics")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)

        def shaped(values: np.ndarray) -> list[list[list[float]]]:
            return [[[float(value)]] for value in values]

        return {
            "mean": shaped(mean),
            "std": shaped(np.sqrt(variance)),
            "max": shaped(self.maximum),
            "min": shaped(self.minimum),
        }


def _read_source_numeric(source: Path, episode_index: int) -> dict[str, np.ndarray]:
    path = _episode_path(source, episode_index)
    table = pq.read_table(path, columns=SOURCE_NUMERIC_COLUMNS)
    result: dict[str, np.ndarray] = {}
    for name in SOURCE_NUMERIC_COLUMNS:
        values = table[name].to_pylist()
        result[name] = np.asarray(values)
    result["state"] = result["state"].astype(np.float32)
    result["actions"] = result["actions"].astype(np.float32)
    result["timestamp"] = result["timestamp"].astype(np.float32)
    for name in ("frame_index", "episode_index", "index", "task_index"):
        result[name] = result[name].astype(np.int64)
    return result


def _open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    return capture


def _encode_png(frame_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", frame_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6]
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode PNG")
    return encoded.tobytes()


def _decode_episode_images(
    source: Path,
    episode_index: int,
    expected_frames: int,
    image_stats: dict[str, ImageStats],
) -> dict[str, list[bytes]]:
    captures = {
        key: _open_video(_video_path(source, episode_index, key))
        for key in IMAGE_KEYS
    }
    output = {key: [] for key in IMAGE_KEYS}
    try:
        for frame_index in range(expected_frames):
            for key, capture in captures.items():
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(
                        f"{key} video ended early at episode {episode_index}, "
                        f"frame {frame_index}/{expected_frames}"
                    )
                if frame.shape != (256, 256, 3):
                    raise ValueError(
                        f"{key} episode {episode_index} frame {frame_index} has "
                        f"shape {frame.shape}"
                    )
                image_stats[key].update_bgr(frame)
                output[key].append(_encode_png(frame))
        for key, capture in captures.items():
            ok, _ = capture.read()
            if ok:
                raise ValueError(
                    f"{key} video has extra frames after episode {episode_index} "
                    f"length {expected_frames}"
                )
    finally:
        for capture in captures.values():
            capture.release()
    return output


def _array_for_type(values: Any, data_type: pa.DataType) -> pa.Array:
    return pa.array(values, type=data_type)


def _build_table(
    schema: pa.Schema,
    numeric: dict[str, np.ndarray],
    png_bytes: dict[str, list[bytes]],
    global_task_index: int,
) -> tuple[pa.Table, np.ndarray]:
    count = int(numeric["state"].shape[0])
    actions = numeric["actions"].copy()
    actions[:, 6] = np.where(actions[:, 6] >= 0.0, 1.0, -1.0).astype(np.float32)
    paths = [f"frame_{index:06d}.png" for index in range(count)]
    values: dict[str, Any] = {
        "image": [
            {"bytes": value, "path": path}
            for value, path in zip(png_bytes["image"], paths, strict=True)
        ],
        "wrist_image": [
            {"bytes": value, "path": path}
            for value, path in zip(png_bytes["wrist_image"], paths, strict=True)
        ],
        "state": numeric["state"].tolist(),
        "actions": actions.tolist(),
        "timestamp": numeric["timestamp"],
        "frame_index": numeric["frame_index"],
        "episode_index": numeric["episode_index"],
        "index": numeric["index"],
        "task_index": np.full(count, global_task_index, dtype=np.int64),
    }
    arrays = [
        _array_for_type(values[field.name], field.type) for field in schema
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    if not table.schema.equals(schema, check_metadata=True):
        raise AssertionError("constructed table does not match the reference schema")
    return table, actions


def _numeric_stats() -> dict[str, VectorStats]:
    return {
        "state": VectorStats(8),
        "actions": VectorStats(7),
        "timestamp": VectorStats(1),
        "frame_index": VectorStats(1),
        "episode_index": VectorStats(1),
        "index": VectorStats(1),
        "task_index": VectorStats(1),
    }


def _update_numeric_stats(
    accumulators: dict[str, VectorStats],
    numeric: dict[str, np.ndarray],
    converted_actions: np.ndarray,
    global_task_index: int,
) -> None:
    count = int(numeric["state"].shape[0])
    accumulators["state"].update(numeric["state"])
    accumulators["actions"].update(converted_actions)
    for name in ("timestamp", "frame_index", "episode_index", "index"):
        accumulators[name].update(numeric[name].reshape(-1, 1))
    accumulators["task_index"].update(
        np.full((count, 1), global_task_index, dtype=np.int64)
    )


def convert(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    reference = args.reference.resolve()
    output = args.output.resolve()
    if source == output or reference == output:
        raise ValueError("output must differ from source and reference")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    schema = _reference_schema(reference)
    task_rows, task_mapping = _task_catalog(reference)
    if args.task_index not in task_mapping:
        raise ValueError(f"task index {args.task_index} is absent from reference catalog")
    task_text = task_mapping[args.task_index]
    source_episodes = _json_lines(source / "meta" / "episodes.jsonl")
    if args.max_episodes is not None:
        source_episodes = source_episodes[: args.max_episodes]
    if not source_episodes:
        raise ValueError("source contains no selected episodes")
    episode_ids = [int(row["episode_index"]) for row in source_episodes]
    if episode_ids != list(range(len(episode_ids))):
        raise ValueError(
            "standalone conversion requires source episode indices contiguous from zero"
        )
    for row in source_episodes:
        tasks = [str(value) for value in row["tasks"]]
        if tasks != [task_text]:
            raise ValueError(
                f"source episode {row['episode_index']} task {tasks} does not match "
                f"reference task {args.task_index}: {task_text}"
            )

    staging = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    (staging / "data").mkdir(parents=True)
    (staging / "meta").mkdir(parents=True)

    image_stats = {key: ImageStats() for key in IMAGE_KEYS}
    numeric_stats = _numeric_stats()
    gripper = VectorStats(1)
    sign_counts = {"negative": 0, "zero": 0, "positive": 0}
    total_frames = 0
    output_episodes: list[dict[str, Any]] = []

    try:
        for position, episode in enumerate(source_episodes, start=1):
            episode_index = int(episode["episode_index"])
            expected_frames = int(episode["length"])
            numeric = _read_source_numeric(source, episode_index)
            if len(numeric["state"]) != expected_frames:
                raise ValueError(
                    f"episode {episode_index} Parquet rows {len(numeric['state'])} "
                    f"!= metadata length {expected_frames}"
                )
            if not np.all(numeric["episode_index"] == episode_index):
                raise ValueError(f"episode {episode_index} contains a wrong episode_index")
            if not np.array_equal(
                numeric["frame_index"], np.arange(expected_frames, dtype=np.int64)
            ):
                raise ValueError(f"episode {episode_index} frame_index is not contiguous")
            if not np.allclose(
                numeric["timestamp"],
                np.arange(expected_frames, dtype=np.float32) / np.float32(10.0),
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError(f"episode {episode_index} is not exact 10 Hz")

            original_gripper = numeric["actions"][:, 6]
            gripper.update(original_gripper.reshape(-1, 1))
            sign_counts["negative"] += int(np.count_nonzero(original_gripper < 0))
            sign_counts["zero"] += int(np.count_nonzero(original_gripper == 0))
            sign_counts["positive"] += int(np.count_nonzero(original_gripper > 0))

            png_bytes = _decode_episode_images(
                source,
                episode_index,
                expected_frames,
                image_stats,
            )
            table, converted_actions = _build_table(
                schema, numeric, png_bytes, args.task_index
            )
            output_path = _episode_path(staging, episode_index)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                table,
                output_path,
                compression="snappy",
                row_group_size=ROW_GROUP_SIZE,
                use_dictionary=True,
            )
            _update_numeric_stats(
                numeric_stats, numeric, converted_actions, args.task_index
            )
            output_episodes.append(
                {
                    "episode_index": episode_index,
                    "tasks": [task_text],
                    "length": expected_frames,
                }
            )
            total_frames += expected_frames
            print(
                f"[{position:03d}/{len(source_episodes):03d}] episode "
                f"{episode_index:06d}: {expected_frames} frames",
                flush=True,
            )

        reference_info = json.loads((reference / "meta" / "info.json").read_text())
        chunks_size = int(reference_info["chunks_size"])
        info = dict(reference_info)
        info.update(
            {
                "codebase_version": "v2.0",
                "total_episodes": len(output_episodes),
                "total_frames": total_frames,
                "total_tasks": len(task_rows),
                "total_videos": 0,
                "total_chunks": math.ceil(len(output_episodes) / chunks_size),
                "fps": 10,
                "splits": {"train": f"0:{len(output_episodes)}"},
            }
        )
        _write_json(staging / "meta" / "info.json", info)
        _write_jsonl(staging / "meta" / "tasks.jsonl", task_rows)
        _write_jsonl(staging / "meta" / "episodes.jsonl", output_episodes)
        stats: dict[str, Any] = {
            "image": image_stats["image"].result(),
            "wrist_image": image_stats["wrist_image"].result(),
        }
        stats.update({name: accumulator.result() for name, accumulator in numeric_stats.items()})
        _write_json(staging / "meta" / "stats.json", stats)
        manifest = {
            "schema_version": 1,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_root": str(source),
            "reference_root": str(reference),
            "output_root": str(output),
            "target_contract": "local LIBERO LeRobot v2.0 embedded-PNG Parquet",
            "global_task_index": args.task_index,
            "task": task_text,
            "episodes": len(output_episodes),
            "frames": total_frames,
            "image_conversion": {
                "source": "external MP4 decoded by OpenCV",
                "target": "lossless PNG bytes embedded in Parquet image structs",
                "shape": [256, 256, 3],
            },
            "schema_conversion": {
                "reference_schema_metadata_copied": True,
                "dropped_source_columns": [
                    "done",
                    "is_success",
                    "intervene_flag",
                ],
                "state_type": "fixed_size_list<float32>[8]",
                "actions_type": "fixed_size_list<float32>[7]",
            },
            "gripper_conversion": {
                "dimension": 6,
                "rule": "g >= 0 -> +1.0; g < 0 -> -1.0",
                "source": gripper.result(),
                "source_sign_counts": sign_counts,
                "output_unique_values": [-1.0, 1.0],
                "motion_dimensions_0_to_5": "preserved exactly",
            },
        }
        _write_json(staging / "meta" / "conversion_manifest.json", manifest)
        os.replace(staging, output)
    except BaseException:
        print(f"conversion failed; incomplete output retained at {staging}", file=sys.stderr)
        raise

    print(f"converted dataset: {output}", flush=True)
    print(f"episodes={len(output_episodes)} frames={total_frames}", flush=True)
    return 0


def _decode_png(value: bytes, label: str) -> np.ndarray:
    encoded = np.frombuffer(value, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot decode PNG: {label}")
    if frame.shape != (256, 256, 3):
        raise ValueError(f"PNG {label} has shape {frame.shape}")
    return frame


def _validate_row_groups(parquet: pq.ParquetFile, path: Path) -> None:
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(row_group_index)
        if row_group.num_rows > ROW_GROUP_SIZE:
            raise ValueError(f"{path} row group exceeds {ROW_GROUP_SIZE} rows")
        for column_index in range(parquet.metadata.num_columns):
            column = row_group.column(column_index)
            if column.compression != "SNAPPY":
                raise ValueError(
                    f"{path} column {column.path_in_schema} is not Snappy compressed"
                )


def validate(args: argparse.Namespace) -> int:
    dataset = args.dataset.resolve()
    reference = args.reference.resolve()
    source = args.source.resolve() if args.source is not None else None
    expected_schema = _reference_schema(reference)
    task_rows, task_mapping = _task_catalog(reference)
    if args.task_index not in task_mapping:
        raise ValueError(f"task index {args.task_index} absent from reference")
    task_text = task_mapping[args.task_index]
    errors: list[str] = []
    episodes = _json_lines(dataset / "meta" / "episodes.jsonl")
    total_frames = 0
    expected_global_index = 0

    try:
        if (dataset / "videos").exists():
            raise ValueError("aligned dataset unexpectedly contains a videos directory")
        if (dataset / "meta" / "tasks.jsonl").read_bytes() != (
            reference / "meta" / "tasks.jsonl"
        ).read_bytes():
            raise ValueError("tasks.jsonl does not exactly match the reference catalog")
        info = json.loads((dataset / "meta" / "info.json").read_text())
        reference_info = json.loads((reference / "meta" / "info.json").read_text())
        if info["features"] != reference_info["features"]:
            raise ValueError("info.json feature contract differs from reference")
        if info["codebase_version"] != "v2.0" or int(info["fps"]) != 10:
            raise ValueError("info.json version/fps differs from v2.0/10 Hz")
        if int(info["total_tasks"]) != len(task_rows):
            raise ValueError("info.json total_tasks differs from copied task catalog")

        for position, episode in enumerate(episodes, start=1):
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            if episode_index != position - 1:
                raise ValueError("episode indices are not contiguous from zero")
            if episode["tasks"] != [task_text]:
                raise ValueError(f"episode {episode_index} has the wrong task text")
            path = _episode_path(dataset, episode_index)
            parquet = pq.ParquetFile(path)
            if not parquet.schema_arrow.equals(expected_schema, check_metadata=True):
                raise ValueError(f"episode {episode_index} Arrow schema differs")
            if parquet.metadata.num_rows != length:
                raise ValueError(f"episode {episode_index} row count differs")
            _validate_row_groups(parquet, path)
            table = pq.read_table(path)
            values = {name: table[name].to_pylist() for name in REQUIRED_COLUMNS}
            state = np.asarray(values["state"], dtype=np.float32)
            actions = np.asarray(values["actions"], dtype=np.float32)
            timestamps = np.asarray(values["timestamp"], dtype=np.float32)
            frames = np.asarray(values["frame_index"], dtype=np.int64)
            episode_values = np.asarray(values["episode_index"], dtype=np.int64)
            indices = np.asarray(values["index"], dtype=np.int64)
            tasks = np.asarray(values["task_index"], dtype=np.int64)
            if not np.array_equal(frames, np.arange(length, dtype=np.int64)):
                raise ValueError(f"episode {episode_index} frame_index differs")
            if not np.all(episode_values == episode_index):
                raise ValueError(f"episode {episode_index} episode_index differs")
            if not np.array_equal(
                indices,
                np.arange(
                    expected_global_index,
                    expected_global_index + length,
                    dtype=np.int64,
                ),
            ):
                raise ValueError(f"episode {episode_index} global index differs")
            if not np.all(tasks == args.task_index):
                raise ValueError(f"episode {episode_index} task_index differs")
            if not np.allclose(
                timestamps,
                np.arange(length, dtype=np.float32) / np.float32(10.0),
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError(f"episode {episode_index} timestamps differ")
            if not set(np.unique(actions[:, 6]).tolist()) <= {-1.0, 1.0}:
                raise ValueError(f"episode {episode_index} gripper is not binary")

            source_numeric = (
                _read_source_numeric(source, episode_index)
                if source is not None
                else None
            )
            captures = (
                {
                    key: _open_video(_video_path(source, episode_index, key))
                    for key in IMAGE_KEYS
                }
                if source is not None
                else {}
            )
            try:
                if source_numeric is not None:
                    if not np.array_equal(state, source_numeric["state"]):
                        raise ValueError(f"episode {episode_index} state changed")
                    if not np.array_equal(actions[:, :6], source_numeric["actions"][:, :6]):
                        raise ValueError(f"episode {episode_index} motion actions changed")
                    expected_gripper = np.where(
                        source_numeric["actions"][:, 6] >= 0.0, 1.0, -1.0
                    ).astype(np.float32)
                    if not np.array_equal(actions[:, 6], expected_gripper):
                        raise ValueError(f"episode {episode_index} gripper rule differs")
                    for name, converted in (
                        ("timestamp", timestamps),
                        ("frame_index", frames),
                        ("episode_index", episode_values),
                        ("index", indices),
                    ):
                        if not np.array_equal(converted, source_numeric[name]):
                            raise ValueError(
                                f"episode {episode_index} source {name} changed"
                            )

                for frame_index in range(length):
                    expected_path = f"frame_{frame_index:06d}.png"
                    for key in IMAGE_KEYS:
                        item = values[key][frame_index]
                        if item["path"] != expected_path:
                            raise ValueError(
                                f"episode {episode_index} {key} path differs at "
                                f"frame {frame_index}"
                            )
                        decoded = _decode_png(
                            item["bytes"],
                            f"episode {episode_index} {key} frame {frame_index}",
                        )
                        if source is not None:
                            ok, source_frame = captures[key].read()
                            if not ok:
                                raise ValueError(
                                    f"source {key} ended early at episode "
                                    f"{episode_index} frame {frame_index}"
                                )
                            if not np.array_equal(decoded, source_frame):
                                raise ValueError(
                                    f"episode {episode_index} {key} PNG pixels changed "
                                    f"at frame {frame_index}"
                                )
                for key, capture in captures.items():
                    ok, _ = capture.read()
                    if ok:
                        raise ValueError(
                            f"source {key} has extra frames at episode {episode_index}"
                        )
            finally:
                for capture in captures.values():
                    capture.release()

            expected_global_index += length
            total_frames += length
            print(
                f"[{position:03d}/{len(episodes):03d}] validated episode "
                f"{episode_index:06d}: {length} frames",
                flush=True,
            )

        if int(info["total_episodes"]) != len(episodes):
            raise ValueError("info.json total_episodes differs")
        if int(info["total_frames"]) != total_frames:
            raise ValueError("info.json total_frames differs")
        if int(info["total_videos"]) != 0:
            raise ValueError("info.json total_videos must be zero")
        required_stats = set(REQUIRED_COLUMNS)
        stats = json.loads((dataset / "meta" / "stats.json").read_text())
        if set(stats) != required_stats:
            raise ValueError("stats.json keys differ from training columns")
    except BaseException as exc:
        errors.append(str(exc))

    report = {
        "schema_version": 1,
        "valid": not errors,
        "errors": errors,
        "dataset_root": str(dataset),
        "reference_root": str(reference),
        "source_root": str(source) if source is not None else None,
        "global_task_index": args.task_index,
        "episodes": len(episodes),
        "frames": total_frames,
        "checks": {
            "exact_reference_arrow_schema": not errors,
            "embedded_png_pixels_equal_source_video": source is not None and not errors,
            "numeric_source_equivalence": source is not None and not errors,
            "binary_gripper": not errors,
            "ten_hz_timestamps": not errors,
            "global_task_catalog": not errors,
        },
    }
    if args.report is not None:
        _write_json(args.report.resolve(), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--source", type=Path, required=True)
    convert_parser.add_argument("--reference", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True)
    convert_parser.add_argument("--task-index", type=int, required=True)
    convert_parser.add_argument("--max-episodes", type=int)
    convert_parser.set_defaults(function=convert)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--dataset", type=Path, required=True)
    validate_parser.add_argument("--reference", type=Path, required=True)
    validate_parser.add_argument("--source", type=Path)
    validate_parser.add_argument("--task-index", type=int, required=True)
    validate_parser.add_argument("--report", type=Path)
    validate_parser.set_defaults(function=validate)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
