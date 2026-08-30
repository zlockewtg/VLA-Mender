"""Atomic, fail-closed builder for VLA-prefix plus repair-suffix datasets."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import DatasetBuildConfig
from .continuity import flow_metrics, require, validate_simulator_evidence
from .manifest import EpisodeSource, ImageSource, SegmentSource, load_episode_manifest
from .media import (
    decode_embedded_image,
    embedded_png_frames,
    transform_image,
    validate_shape,
    video_frame,
    video_png_frames,
)


PREFIX_PHASE = "original_vla_prefix"
REPAIR_PHASE = "repair_suffix"
GUARD_PHASE = "intervention_guard"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class Accumulator:
    def __init__(self, width: int) -> None:
        self.count = 0
        self.sum = np.zeros(width, dtype=np.float64)
        self.sumsq = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf, dtype=np.float64)
        self.maximum = np.full(width, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1, self.sum.size)
        require(array.size > 0 and np.isfinite(array).all(), "cannot accumulate empty/non-finite values")
        self.count += len(array)
        self.sum += array.sum(axis=0)
        self.sumsq += np.square(array).sum(axis=0)
        self.minimum = np.minimum(self.minimum, array.min(axis=0))
        self.maximum = np.maximum(self.maximum, array.max(axis=0))

    def value(self) -> dict[str, Any]:
        require(self.count > 0, "statistics accumulator is empty")
        mean = self.sum / self.count
        variance = np.maximum(self.sumsq / self.count - np.square(mean), 0.0)
        return {
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "max": self.maximum.tolist(),
            "min": self.minimum.tolist(),
        }


class ImageAccumulator(Accumulator):
    def __init__(self) -> None:
        super().__init__(3)

    def update_image(self, rgb: np.ndarray) -> None:
        self.update(rgb.astype(np.float64).reshape(-1, 3) / 255.0)

    def value(self) -> dict[str, Any]:
        return {key: [[[item]] for item in values] for key, values in super().value().items()}


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _trajectory_table(
    source: SegmentSource, config: DatasetBuildConfig, *, repair: bool
) -> pa.Table:
    """Adapt VLA-Mender public trajectory JSON to the builder's tabular contract."""

    require(source.trajectory is not None, "trajectory source path is missing")
    payload = json.loads(source.trajectory.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"trajectory must be an object: {source.trajectory}")
    states = payload.get("states")
    actions = payload.get("actions")
    require(isinstance(states, list) and states, f"trajectory has no states: {source.trajectory}")
    require(isinstance(actions, list) and actions, f"trajectory has no actions: {source.trajectory}")
    require(
        len(states) == len(actions),
        f"trajectory state/action lengths differ: {source.trajectory}",
    )
    if not repair:
        return pa.table(
            {
                config.columns.state: states,
                config.columns.action: actions,
            }
        )

    rewards = payload.get("rewards")
    successes = payload.get("success_flags", payload.get("successes"))
    require(
        isinstance(rewards, list) and len(rewards) == len(states),
        f"repair trajectory rewards do not align: {source.trajectory}",
    )
    require(
        isinstance(successes, list) and len(successes) == len(states),
        f"repair trajectory success flags do not align: {source.trajectory}",
    )
    columns = config.repair_columns
    values: dict[str, Any] = {
        columns.state: states,
        columns.action: actions,
        columns.action_valid: [True] * len(states),
    }
    if columns.reward is not None:
        values[columns.reward] = rewards
    if columns.done is not None:
        values[columns.done] = successes
    return pa.table(values)


def _load_segment_table(
    source: SegmentSource, config: DatasetBuildConfig, *, repair: bool
) -> pa.Table:
    if source.parquet is not None:
        require(source.parquet.is_file(), f"missing segment parquet: {source.parquet}")
        return pq.read_table(source.parquet)
    require(
        source.trajectory is not None and source.trajectory.is_file(),
        f"missing segment trajectory: {source.trajectory}",
    )
    return _trajectory_table(source, config, repair=repair)


def _segment_payload_path(source: SegmentSource) -> Path:
    path = source.parquet or source.trajectory
    require(path is not None, "segment has no payload path")
    return path


def _action_transform(actions: np.ndarray, config: DatasetBuildConfig) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float32).copy()
    require(result.ndim == 2 and result.shape[1] == config.action.action_dim, "bad action shape")
    require(np.isfinite(result).all(), "actions contain non-finite values")
    require(
        float(np.max(np.abs(result))) <= config.action.maximum_absolute_value,
        "action exceeds configured maximum_absolute_value",
    )
    if config.action.gripper_index is not None:
        index = config.action.gripper_index % config.action.action_dim
        result[:, index] = np.where(
            result[:, index] >= config.action.gripper_threshold,
            config.action.gripper_high,
            config.action.gripper_low,
        )
    return result


def _repair_payload(
    table: pa.Table, config: DatasetBuildConfig
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    columns = config.repair_columns
    required = {columns.state, columns.action, columns.action_valid}
    if config.require_terminal_success:
        required.update(item for item in (columns.reward, columns.done) if item is not None)
        require(columns.reward is not None and columns.done is not None, "terminal success columns are disabled")
    require(
        required <= set(table.column_names),
        f"repair parquet is missing columns: {sorted(required - set(table.column_names))}",
    )
    valid = np.asarray(table[columns.action_valid].to_pylist(), dtype=bool)
    require(valid.size > 0 and bool(valid[0]), "repair has no valid action rows")
    invalid = np.flatnonzero(~valid)
    length = int(invalid[0]) if invalid.size else len(valid)
    require(bool(np.all(valid[:length])) and not bool(np.any(valid[length:])), "repair valid rows are not a prefix")
    states = np.asarray(table[columns.state].to_pylist()[:length], dtype=np.float32)
    actions = _action_transform(
        np.asarray(table[columns.action].to_pylist()[:length], dtype=np.float32), config
    )
    require(states.shape == (length, config.action.state_dim), f"bad repair state shape: {states.shape}")
    require(np.isfinite(states).all(), "repair states contain non-finite values")
    transition: dict[str, Any] = {
        "valid_action_rows": length,
        "invalid_terminal_rows_removed": len(table) - length,
    }
    if config.require_terminal_success:
        rewards = np.asarray(table[columns.reward].to_pylist(), dtype=np.float64)
        done = np.asarray(table[columns.done].to_pylist(), dtype=bool)
        successes = np.flatnonzero((rewards[:length] > 0.0) & done[:length])
        require(successes.size > 0, "repair contains no successful valid transition")
        require(rewards[length - 1] > 0.0 and done[length - 1], "last valid repair action is not successful")
        transition.update(
            {
                "first_success_frame": int(successes[0]),
                "last_success_frame": int(successes[-1]),
            }
        )
    return states, actions, transition


def _segment_images(
    source: ImageSource,
    table: pa.Table,
    *,
    fallback_column: str,
    start: int,
    length: int,
    width: int,
    height: int,
    horizontal_flip: bool,
    fps: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    if source.video is not None:
        encoded, decoded = video_png_frames(
            source.video,
            start + length,
            width=width,
            height=height,
            horizontal_flip=horizontal_flip,
            expected_fps=fps,
        )
        return encoded[start:], decoded[start:]
    column = source.column or fallback_column
    require(column in table.column_names, f"missing embedded image column: {column}")
    items = table[column].to_pylist()[start : start + length]
    require(len(items) == length, f"embedded column {column} ended before requested range")
    return embedded_png_frames(
        items, width=width, height=height, horizontal_flip=horizontal_flip
    )


def _image_at(
    source: ImageSource,
    table: pa.Table,
    *,
    fallback_column: str,
    index: int,
    width: int,
    height: int,
    horizontal_flip: bool,
    fps: int,
) -> np.ndarray:
    authoritative_video = source.continuity_video or source.video
    if authoritative_video is not None:
        rgb = video_frame(authoritative_video, index, expected_fps=fps)
    else:
        column = source.column or fallback_column
        require(column in table.column_names and index < len(table), f"missing image {column}[{index}]")
        rgb = decode_embedded_image(table[column][index].as_py())
    rgb = transform_image(rgb, horizontal_flip=horizontal_flip)
    validate_shape(rgb, width=width, height=height, label=f"continuity image {index}")
    return rgb


def _source_for_camera(
    episode: EpisodeSource, output_column: str, *, repair: bool, fallback: str
) -> ImageSource:
    segment = episode.repair if repair else episode.prefix
    return segment.images.get(output_column, ImageSource(column=fallback))


def _make_table(
    schema: pa.Schema,
    config: DatasetBuildConfig,
    images: Mapping[str, list[dict[str, Any]]],
    states: np.ndarray,
    actions: np.ndarray,
    *,
    episode_index: int,
    global_start: int,
    task_index: int,
) -> pa.Table:
    columns = config.columns
    length = len(states)
    frame = np.arange(length, dtype=np.int64)
    values: dict[str, Any] = {
        **images,
        columns.state: states.tolist(),
        columns.action: actions.tolist(),
        columns.timestamp: (frame.astype(np.float32) / config.fps).tolist(),
        columns.frame_index: frame.tolist(),
        columns.episode_index: np.full(length, episode_index, dtype=np.int64).tolist(),
        columns.global_index: np.arange(global_start, global_start + length, dtype=np.int64).tolist(),
        columns.task_index: np.full(length, task_index, dtype=np.int64).tolist(),
    }
    require(
        set(values) == set(schema.names),
        f"output/reference columns differ: output={sorted(values)} reference={schema.names}",
    )
    return pa.Table.from_pydict(values, schema=schema)


def valid_start_indices(frames: list[dict[str, Any]], action_horizon: int) -> list[int]:
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in frames:
        by_episode[int(row["episode_index"])].append(row)
    valid: list[int] = []
    for rows in by_episode.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
        for start, row in enumerate(rows):
            if not row["trainable"]:
                continue
            targets = rows[start : min(len(rows), start + action_horizon)]
            if all(
                target["trainable"]
                and target["continuous_segment_id"] == row["continuous_segment_id"]
                for target in targets
            ):
                valid.append(int(row["index"]))
    return valid


def _validate_reference(config: DatasetBuildConfig) -> tuple[dict[str, Any], pa.Schema, list[dict[str, Any]]]:
    root = config.reference_dataset
    require(root.is_dir(), f"missing reference dataset: {root}")
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    require(bool(parquet_files), f"reference dataset has no parquet files: {root}")
    schema = pq.read_schema(parquet_files[0])
    task_path = config.task_catalog or (root / "meta/tasks.jsonl")
    tasks = _read_tasks(task_path)
    require(bool(tasks), f"task catalog is empty: {task_path}")
    return info, schema, tasks


def _require_reference_vector_contract(
    reference_info: dict[str, Any],
    reference_schema: pa.Schema,
    config: DatasetBuildConfig,
) -> None:
    """Fail before loading episodes when configured vectors differ from reference."""

    configured = {
        config.columns.state: config.action.state_dim,
        config.columns.action: config.action.action_dim,
    }
    info_features = reference_info.get("features")
    require(isinstance(info_features, dict), "reference info.json has no features object")
    for name, width in configured.items():
        index = reference_schema.get_field_index(name)
        require(index >= 0, f"reference schema has no {name} column")
        field = reference_schema.field(index)
        require(
            pa.types.is_fixed_size_list(field.type),
            f"reference column {name} is not a fixed-size list: {field.type}",
        )
        reference_width = int(field.type.list_size)
        require(
            width == reference_width,
            f"configured {name} width {width} differs from reference width {reference_width}",
        )
        feature = info_features.get(name)
        require(isinstance(feature, dict), f"reference info.json has no {name} feature")
        require(
            feature.get("shape") == [reference_width],
            f"reference info.json {name} shape differs from parquet schema",
        )


def _action_summary(actions: np.ndarray) -> dict[str, Any]:
    delta = np.diff(actions, axis=0) if len(actions) > 1 else np.zeros((0, actions.shape[1]))
    return {
        "frames": len(actions),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "action_delta_abs_p99": (
            np.quantile(np.abs(delta), 0.99, axis=0).tolist()
            if len(delta)
            else [0.0] * actions.shape[1]
        ),
    }


def build_dataset(config: DatasetBuildConfig) -> dict[str, Any]:
    """Build and atomically publish the configured dataset."""

    config.validate()
    output = config.output.resolve()
    require(not output.exists(), f"refusing to overwrite existing output: {output}")
    reference_info, reference_schema, task_catalog = _validate_reference(config)
    _require_reference_vector_contract(reference_info, reference_schema, config)
    episodes = load_episode_manifest(config.episodes_manifest)
    task_by_index = {int(item["task_index"]): str(item["task"]) for item in task_catalog}
    for episode in episodes:
        require(
            task_by_index.get(episode.task_index) == episode.task,
            f"task catalog mismatch for task_index={episode.task_index}",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    complete = False
    try:
        data_dir = staging / "data/chunk-000"
        meta_dir = staging / "meta"
        data_dir.mkdir(parents=True)
        meta_dir.mkdir(parents=True)
        numeric = {
            config.columns.state: Accumulator(config.action.state_dim),
            config.columns.action: Accumulator(config.action.action_dim),
            config.columns.timestamp: Accumulator(1),
            config.columns.frame_index: Accumulator(1),
            config.columns.episode_index: Accumulator(1),
            config.columns.global_index: Accumulator(1),
            config.columns.task_index: Accumulator(1),
        }
        image_stats = {name: ImageAccumulator() for name in config.cameras}
        action_groups: dict[str, list[np.ndarray]] = defaultdict(list)
        manifest_episodes: list[dict[str, Any]] = []
        episode_rows: list[dict[str, Any]] = []
        episode_stats: list[dict[str, Any]] = []
        trainable_frames: list[dict[str, Any]] = []
        global_index = 0
        total_prefix = 0
        total_repair = 0
        repair_only_episode_count = 0
        repair_only_frame_count = 0
        maximum_splice_error = 0.0

        for episode_index, source in enumerate(episodes):
            repair_table = _load_segment_table(source.repair, config, repair=True)
            columns = config.columns
            repair_only = source.mode == "repair_only"
            restart = 0 if repair_only else source.restart_frame
            prefix_table: pa.Table | None = None
            prefix_states = np.empty((0, config.action.state_dim), dtype=np.float32)
            prefix_actions = np.empty((0, config.action.action_dim), dtype=np.float32)
            if not repair_only:
                require(source.prefix is not None, "prefix_plus_repair episode has no prefix")
                prefix_table = _load_segment_table(source.prefix, config, repair=False)
                require(restart < len(prefix_table), f"restart frame {restart} outside prefix source")
                require(
                    {columns.state, columns.action} <= set(prefix_table.column_names),
                    "prefix payload misses state/action columns",
                )
                prefix_states = np.asarray(
                    prefix_table[columns.state].to_pylist()[:restart], dtype=np.float32
                )
                prefix_actions = _action_transform(
                    np.asarray(
                        prefix_table[columns.action].to_pylist()[:restart], dtype=np.float32
                    ),
                    config,
                )
                require(
                    prefix_states.shape == (restart, config.action.state_dim),
                    f"bad prefix state shape: {prefix_states.shape}",
                )
                require(np.isfinite(prefix_states).all(), "prefix states contain non-finite values")
            repair_states, repair_actions, transition = _repair_payload(repair_table, config)
            splice_error: float | None = None
            if not repair_only:
                require(prefix_table is not None, "prefix table was not loaded")
                restart_state = np.asarray(
                    prefix_table[columns.state][restart].as_py(), dtype=np.float32
                )
                splice_error = float(np.max(np.abs(restart_state - repair_states[0])))
                require(
                    splice_error <= config.continuity.splice_state_tolerance,
                    f"splice state error {splice_error} exceeds tolerance",
                )
                maximum_splice_error = max(maximum_splice_error, splice_error)

            simulator_evidence = None
            if config.continuity.require_simulator_evidence:
                require(bool(source.continuity), f"episode {source.source_episode_id} has no simulator evidence")
                simulator_evidence = validate_simulator_evidence(
                    source.continuity,
                    fields=config.continuity.signature_fields,
                    tolerance=config.continuity.simulator_state_tolerance,
                )

            images: dict[str, list[dict[str, Any]]] = {}
            visual: dict[str, Any] = {}
            for output_column, camera in config.cameras.items():
                repair_source = _source_for_camera(
                    source, output_column, repair=True, fallback=output_column
                )
                prefix_encoded: list[dict[str, Any]] = []
                prefix_decoded: list[np.ndarray] = []
                prefix_source: ImageSource | None = None
                if not repair_only:
                    require(prefix_table is not None, "prefix table was not loaded")
                    prefix_source = _source_for_camera(
                        source, output_column, repair=False, fallback=camera.prefix_column
                    )
                    prefix_encoded, prefix_decoded = _segment_images(
                        prefix_source,
                        prefix_table,
                        fallback_column=camera.prefix_column,
                        start=0,
                        length=restart,
                        width=camera.width,
                        height=camera.height,
                        horizontal_flip=False,
                        fps=config.fps,
                    )
                repair_encoded, repair_decoded = _segment_images(
                    repair_source,
                    repair_table,
                    fallback_column=output_column,
                    start=0,
                    length=len(repair_states),
                    width=camera.width,
                    height=camera.height,
                    horizontal_flip=camera.repair_flip_horizontal,
                    fps=config.fps,
                )
                if repair_only:
                    visual[output_column] = {
                        "check": "not_applicable_repair_only",
                        "repair_flip_horizontal": camera.repair_flip_horizontal,
                        "payload_shape_verified": True,
                        "verified": True,
                    }
                else:
                    require(prefix_source is not None and prefix_table is not None, "missing prefix media")
                    source_restart = _image_at(
                        prefix_source,
                        prefix_table,
                        fallback_column=camera.prefix_column,
                        index=restart,
                        width=camera.width,
                        height=camera.height,
                        horizontal_flip=False,
                        fps=config.fps,
                    )
                    repair_zero = repair_decoded[0]
                    metrics = flow_metrics(source_restart, repair_zero)
                    require(
                        metrics["flow_median_px"]
                        <= config.continuity.max_flow_median_px,
                        f"{output_column} median flow too large: {metrics}",
                    )
                    require(
                        metrics["flow_p90_px"] <= config.continuity.max_flow_p90_px,
                        f"{output_column} p90 flow too large: {metrics}",
                    )
                    visual[output_column] = {
                        **metrics,
                        "repair_flip_horizontal": camera.repair_flip_horizontal,
                        "verified": True,
                    }
                images[output_column] = prefix_encoded + repair_encoded
                for rgb in prefix_decoded + repair_decoded:
                    image_stats[output_column].update_image(rgb)

            states = np.concatenate([prefix_states, repair_states], axis=0)
            actions = np.concatenate([prefix_actions, repair_actions], axis=0)
            length = len(states)
            table = _make_table(
                reference_schema,
                config,
                images,
                states,
                actions,
                episode_index=episode_index,
                global_start=global_index,
                task_index=source.task_index,
            )
            relative = Path(f"data/chunk-000/episode_{episode_index:06d}.parquet")
            parquet_path = staging / relative
            pq.write_table(table, parquet_path, compression="zstd")

            frame_values = np.arange(length, dtype=np.int64)
            numeric[columns.state].update(states)
            numeric[columns.action].update(actions)
            numeric[columns.timestamp].update(frame_values.astype(np.float32) / config.fps)
            numeric[columns.frame_index].update(frame_values)
            numeric[columns.episode_index].update(np.full(length, episode_index))
            numeric[columns.global_index].update(np.arange(global_index, global_index + length))
            numeric[columns.task_index].update(np.full(length, source.task_index))
            if len(prefix_actions):
                action_groups[PREFIX_PHASE].append(prefix_actions)
            action_groups[REPAIR_PHASE].append(repair_actions)

            pre_guard_start = max(0, restart - config.pre_guard_frames) if not repair_only else 0
            post_guard_stop = (
                min(length, restart + config.post_guard_frames) if not repair_only else 0
            )
            for frame_index in range(length):
                in_prefix = not repair_only and frame_index < restart
                in_pre = not repair_only and pre_guard_start <= frame_index < restart
                in_post = not repair_only and restart <= frame_index < post_guard_stop
                guarded = in_pre or in_post
                terminal = frame_index == length - 1
                source_phase = PREFIX_PHASE if in_prefix else REPAIR_PHASE
                trainable_frames.append(
                    {
                        "index": global_index + frame_index,
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "phase": GUARD_PHASE if guarded else source_phase,
                        "source_phase": source_phase,
                        "segment_role": (
                            "vla_prefix" if in_prefix else "repair_only" if repair_only else "repair_suffix"
                        ),
                        "trainable": not guarded and not terminal,
                        "continuous_segment_id": (
                            f"{episode_index}:native_episode"
                            if not repair_only and config.allow_splice_crossing_action_chunks
                            else f"{episode_index}:"
                            f"{'vla_prefix' if in_prefix else 'repair_only' if repair_only else 'repair_suffix'}"
                        ),
                        "terminal_guard": terminal,
                        "intervention_guard": guarded,
                        "intervention_guard_role": (
                            "pre_intervention_error_action"
                            if in_pre
                            else "post_intervention_handoff"
                            if in_post
                            else None
                        ),
                        "splice_boundary_after": not repair_only and frame_index == restart - 1,
                        "splice_boundary_before": not repair_only and frame_index == restart,
                        "dataset_source": config.dataset_source,
                        "source_episode_id": source.source_episode_id,
                        "source_frame_index": frame_index if repair_only or in_prefix else frame_index - restart,
                    }
                )

            entry = {
                "episode_index": episode_index,
                "source_episode_id": source.source_episode_id,
                "episode_mode": source.mode,
                "task_index": source.task_index,
                "task": source.task,
                "length": length,
                "vla_prefix_length": restart,
                "repair_suffix_length": len(repair_states),
                "splice_frame_index": None if repair_only else restart,
                "splice_state_max_abs_error": splice_error,
                "source_prefix_payload": (
                    str(_segment_payload_path(source.prefix)) if source.prefix is not None else None
                ),
                "source_prefix_payload_sha256": (
                    sha256_file(_segment_payload_path(source.prefix))
                    if source.prefix is not None
                    else None
                ),
                "source_prefix_parquet": (
                    str(source.prefix.parquet)
                    if source.prefix is not None and source.prefix.parquet is not None
                    else None
                ),
                "source_prefix_parquet_sha256": (
                    sha256_file(source.prefix.parquet)
                    if source.prefix is not None and source.prefix.parquet is not None
                    else None
                ),
                "source_repair_payload": str(_segment_payload_path(source.repair)),
                "source_repair_payload_sha256": sha256_file(
                    _segment_payload_path(source.repair)
                ),
                "source_repair_parquet": (
                    str(source.repair.parquet)
                    if source.repair.parquet is not None
                    else None
                ),
                "source_repair_parquet_sha256": (
                    sha256_file(source.repair.parquet)
                    if source.repair.parquet is not None
                    else None
                ),
                "simulator_continuity": simulator_evidence,
                "visual_continuity": visual,
                "repair_transition_evidence": transition,
                "source_metadata": source.metadata,
                "parquet": str(relative),
                "parquet_sha256": sha256_file(parquet_path),
            }
            manifest_episodes.append(entry)
            episode_rows.append({"episode_index": episode_index, "tasks": [source.task], "length": length})
            episode_stats.append(
                {
                    "episode_index": episode_index,
                    "stats": {
                        columns.state: _array_stats(states),
                        columns.action: _array_stats(actions),
                    },
                }
            )
            total_prefix += restart
            total_repair += len(repair_states)
            if repair_only:
                repair_only_episode_count += 1
                repair_only_frame_count += len(repair_states)
            global_index += length

        valid_indices = valid_start_indices(trainable_frames, config.action_horizon)
        trainable = {
            "schema_version": 1,
            "created_at": utc_now(),
            "action_horizon": config.action_horizon,
            "pre_intervention_guard_frames": config.pre_guard_frames,
            "post_intervention_guard_frames": config.post_guard_frames,
            "frame_count": global_index,
            "valid_start_count": len(valid_indices),
            "frames": trainable_frames,
        }
        info = copy.deepcopy(reference_info)
        info.update(
            {
                "total_episodes": len(episodes),
                "total_frames": global_index,
                "total_tasks": len(task_catalog),
                "total_videos": 0,
                "total_chunks": math.ceil(len(episodes) / int(info.get("chunks_size", 1000))),
                "fps": config.fps,
                "splits": {"train": f"0:{len(episodes)}"},
            }
        )
        write_json(meta_dir / "info.json", info)
        write_jsonl(meta_dir / "tasks.jsonl", task_catalog)
        write_jsonl(meta_dir / "episodes.jsonl", episode_rows)
        write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stats)
        write_json(
            meta_dir / "stats.json",
            {
                **{name: value.value() for name, value in image_stats.items()},
                **{name: value.value() for name, value in numeric.items()},
            },
        )
        write_json(meta_dir / "trainable_index_manifest.json", trainable)
        write_json(
            meta_dir / "action_distribution_report.json",
            {
                "schema_version": 1,
                "created_at": utc_now(),
                "by_phase": {
                    phase: _action_summary(np.concatenate(group, axis=0))
                    for phase, group in action_groups.items()
                },
            },
        )
        manifest = {
            "schema_version": 2,
            "created_at": utc_now(),
            "dataset_source": config.dataset_source,
            "reference_dataset": str(config.reference_dataset),
            "reference_info_sha256": sha256_file(config.reference_dataset / "meta/info.json"),
            "input_manifest": str(config.episodes_manifest),
            "input_manifest_sha256": sha256_file(config.episodes_manifest),
            "config": str(config.config_path),
            "config_sha256": sha256_file(config.config_path),
            "episode_count": len(episodes),
            "frame_count": global_index,
            "vla_prefix_frame_count": total_prefix,
            "repair_suffix_frame_count": total_repair,
            "spliced_episode_count": len(episodes) - repair_only_episode_count,
            "repair_only_episode_count": repair_only_episode_count,
            "repair_only_frame_count": repair_only_frame_count,
            "valid_start_count": len(valid_indices),
            "max_splice_state_abs_error": maximum_splice_error,
            "policies": {
                "prefix_range": "[0, restart_frame)",
                "duplicate_restart_state_avoided": True,
                "splice_crossing_action_chunks_trainable": (
                    config.allow_splice_crossing_action_chunks
                ),
                "pre_guard_frames": config.pre_guard_frames,
                "post_guard_frames": config.post_guard_frames,
                "repair_only_intervention_guard_frames": 0,
                "terminal_row_trainable": False,
                "simulator_continuity": vars(config.continuity),
            },
            "episodes": manifest_episodes,
        }
        write_json(meta_dir / "build_manifest.json", manifest)

        errors = _validate_output(
            staging,
            reference_schema,
            manifest_episodes,
            global_index,
            config,
        )
        demo_manifest: dict[str, Any] | None = None
        if config.demo_videos.enabled:
            from .demo_videos import render_demo_videos

            demo_manifest = render_demo_videos(
                staging,
                meta_dir / "visualization/trajectory_demos",
                camera_columns=tuple(config.cameras),
                fps=config.fps,
                count=config.demo_videos.count,
                dataset_identity=config.output,
            )
        report = {
            "schema_version": 1,
            "created_at": utc_now(),
            "valid": not errors,
            "errors": errors,
            "dataset": str(output),
            "episode_count": len(episodes),
            "frame_count": global_index,
            "vla_prefix_frame_count": total_prefix,
            "repair_suffix_frame_count": total_repair,
            "spliced_episode_count": len(episodes) - repair_only_episode_count,
            "repair_only_episode_count": repair_only_episode_count,
            "repair_only_frame_count": repair_only_frame_count,
            "valid_start_count": len(valid_indices),
            "max_splice_state_abs_error": maximum_splice_error,
            "all_simulator_continuity_verified": config.continuity.require_simulator_evidence,
            "all_visual_continuity_verified": True,
            "demo_video_count": (
                int(demo_manifest["demo_count"]) if demo_manifest is not None else 0
            ),
            "demo_video_manifest": (
                "meta/visualization/trajectory_demos/manifest.json"
                if demo_manifest is not None
                else None
            ),
        }
        write_json(meta_dir / "validation_report.json", report)
        require(report["valid"], f"output validation failed: {errors}")

        provenance = meta_dir / "provenance"
        provenance.mkdir(parents=True)
        shutil.copy2(config.config_path, provenance / "build_config.yaml")
        shutil.copy2(config.episodes_manifest, provenance / "episodes_manifest.json")
        programs = provenance / "programs"
        programs.mkdir()
        for module_name in (
            "builder.py",
            "config.py",
            "continuity.py",
            "demo_videos.py",
            "manifest.py",
            "media.py",
            "research_manifest.py",
            "run.py",
        ):
            shutil.copy2(Path(__file__).with_name(module_name), programs / module_name)
        copied: list[dict[str, str]] = []
        for path in config.provenance_files:
            require(path is not None and path.is_file(), f"missing provenance file: {path}")
            destination = provenance / "files" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination = destination.with_name(f"{sha256_file(path)[:12]}-{path.name}")
            shutil.copy2(path, destination)
            copied.append(
                {
                    "source": str(path),
                    "copy": str(destination.relative_to(staging)),
                    "sha256": sha256_file(path),
                }
            )
        write_json(provenance / "manifest.json", {"files": copied})

        os.replace(staging, output)
        complete = True
        return report
    finally:
        if not complete and staging.exists():
            shutil.rmtree(staging)


def _array_stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "min": array.min(axis=0).tolist(),
    }


def _validate_output(
    staging: Path,
    schema: pa.Schema,
    episodes: list[dict[str, Any]],
    expected_frames: int,
    config: DatasetBuildConfig,
) -> list[str]:
    errors: list[str] = []
    global_index = 0
    for episode_index, entry in enumerate(episodes):
        table = pq.read_table(staging / entry["parquet"])
        if not table.schema.equals(schema, check_metadata=True):
            errors.append(f"schema:{episode_index}")
        if len(table) != int(entry["length"]):
            errors.append(f"length:{episode_index}")
        length = len(table)
        columns = config.columns
        frame = np.asarray(table[columns.frame_index].to_pylist(), dtype=np.int64)
        episode = np.asarray(table[columns.episode_index].to_pylist(), dtype=np.int64)
        index = np.asarray(table[columns.global_index].to_pylist(), dtype=np.int64)
        task = np.asarray(table[columns.task_index].to_pylist(), dtype=np.int64)
        timestamp = np.asarray(table[columns.timestamp].to_pylist(), dtype=np.float32)
        if not np.array_equal(frame, np.arange(length)):
            errors.append(f"frame_index:{episode_index}")
        if not np.array_equal(episode, np.full(length, episode_index)):
            errors.append(f"episode_index:{episode_index}")
        if not np.array_equal(index, np.arange(global_index, global_index + length)):
            errors.append(f"global_index:{episode_index}")
        if not np.array_equal(task, np.full(length, int(entry["task_index"]))):
            errors.append(f"task_index:{episode_index}")
        if not np.allclose(timestamp, frame.astype(np.float32) / config.fps, atol=1e-7):
            errors.append(f"timestamp:{episode_index}")
        if sha256_file(staging / entry["parquet"]) != entry["parquet_sha256"]:
            errors.append(f"parquet_sha256:{episode_index}")
        global_index += len(table)
    if global_index != expected_frames:
        errors.append("frame_count")
    return errors
