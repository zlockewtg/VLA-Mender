"""Transition-aware sample filtering for LeRobot action chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, SupportsIndex

import numpy as np


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported trainable-index manifest: {manifest_path}")
    if not isinstance(value.get("frames"), list):
        raise ValueError(f"trainable-index manifest has no frames list: {manifest_path}")
    return value


def valid_global_indices(manifest: Mapping[str, Any], action_horizon: int) -> tuple[int, ...]:
    """Recompute action-chunk starts; never trust only the declared count."""

    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    frames = sorted(
        (dict(frame) for frame in manifest["frames"]),
        key=lambda frame: (int(frame["episode_index"]), int(frame["frame_index"])),
    )
    by_episode: dict[int, list[dict[str, Any]]] = {}
    seen: set[int] = set()
    for frame in frames:
        global_index = int(frame["index"])
        if global_index in seen:
            raise ValueError(f"duplicate global frame index {global_index}")
        seen.add(global_index)
        by_episode.setdefault(int(frame["episode_index"]), []).append(frame)
    valid: list[int] = []
    for episode_index in sorted(by_episode):
        episode = by_episode[episode_index]
        actual = [int(frame["frame_index"]) for frame in episode]
        if actual != list(range(len(episode))):
            raise ValueError(f"episode {episode_index} frame indices are not contiguous")
        for start, frame in enumerate(episode):
            if not bool(frame.get("trainable", False)):
                continue
            segment = frame.get("continuous_segment_id")
            targets = episode[start : min(len(episode), start + action_horizon)]
            if all(
                bool(target.get("trainable", False))
                and target.get("continuous_segment_id") == segment
                and str(target.get("phase")) != "transition"
                for target in targets
            ):
                valid.append(int(frame["index"]))
    return tuple(valid)


class TrainableIndexDataset:
    """Map compact sampler indices to manifest-approved LeRobot rows."""

    def __init__(self, dataset: Any, manifest: Mapping[str, Any], action_horizon: int) -> None:
        self._dataset = dataset
        approved = valid_global_indices(manifest, action_horizon)
        declared = int(manifest.get("valid_start_count", len(approved)))
        if declared != len(approved):
            raise ValueError(
                f"trainable-index valid_start_count is {declared}, recomputed {len(approved)}"
            )
        hf_dataset = getattr(dataset, "hf_dataset", None)
        if hf_dataset is None:
            raise TypeError("TrainableIndexDataset requires a LeRobotDataset with hf_dataset")
        columns = set(getattr(hf_dataset, "column_names", ()))
        source = (
            np.asarray(hf_dataset["index"], dtype=np.int64).reshape(-1)
            if "index" in columns
            else np.arange(len(dataset), dtype=np.int64)
        )
        global_to_local = {int(global_index): local for local, global_index in enumerate(source)}
        missing = [index for index in approved if index not in global_to_local]
        if missing:
            raise ValueError(
                f"{len(missing)} approved indices are absent from LeRobot; first: {missing[:10]}"
            )
        self.valid_global_indices = approved
        self.valid_local_indices = tuple(global_to_local[index] for index in approved)
        self.action_horizon = int(action_horizon)

    def __getitem__(self, index: SupportsIndex) -> Any:
        return self._dataset[self.valid_local_indices[index.__index__()]]

    def __len__(self) -> int:
        return len(self.valid_local_indices)
