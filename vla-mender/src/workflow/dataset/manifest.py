"""Input episode-manifest parser with explicit path and lineage handling."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _resolve(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass(frozen=True)
class ImageSource:
    video: Path | None = None
    column: str | None = None

    @classmethod
    def parse(cls, base: Path, value: Mapping[str, Any], label: str) -> "ImageSource":
        unknown = set(value) - {"video", "column"}
        if unknown:
            raise ValueError(f"unknown {label} image-source keys: {sorted(unknown)}")
        video = _resolve(base, value["video"], f"{label}.video") if value.get("video") else None
        column = str(value["column"]) if value.get("column") else None
        if (video is None) == (column is None):
            raise ValueError(f"{label} requires exactly one of video or column")
        return cls(video=video, column=column)


@dataclass(frozen=True)
class SegmentSource:
    parquet: Path
    images: dict[str, ImageSource]

    @classmethod
    def parse(cls, base: Path, value: Mapping[str, Any], label: str) -> "SegmentSource":
        unknown = set(value) - {"parquet", "images"}
        if unknown:
            raise ValueError(f"unknown {label} keys: {sorted(unknown)}")
        images = value.get("images") or {}
        if not isinstance(images, Mapping):
            raise ValueError(f"{label}.images must be a mapping")
        return cls(
            parquet=_resolve(base, value.get("parquet"), f"{label}.parquet"),
            images={
                str(name): ImageSource.parse(base, source, f"{label}.images.{name}")
                for name, source in images.items()
            },
        )


@dataclass(frozen=True)
class EpisodeSource:
    source_episode_id: int | str
    restart_frame: int
    task_index: int
    task: str
    prefix: SegmentSource
    repair: SegmentSource
    continuity: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve_continuity(base: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "reset_descriptor", "attempt_manifest", "result", "reset_state_key",
        "repair_state_key", "expected_source_python",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown continuity keys: {sorted(unknown)}")
    result = dict(value)
    for key in ("reset_descriptor", "attempt_manifest", "result"):
        if key in result:
            result[key] = str(_resolve(base, result[key], f"continuity.{key}"))
    return result


def load_episode_manifest(path: Path) -> list[EpisodeSource]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("episodes") if isinstance(raw, Mapping) else raw
    if not isinstance(rows, list) or not rows:
        raise ValueError("episode manifest must contain a non-empty episodes list")
    base = path.parent
    episodes: list[EpisodeSource] = []
    for position, item in enumerate(rows):
        if not isinstance(item, Mapping):
            raise ValueError(f"episode {position} is not an object")
        required = {"source_episode_id", "restart_frame", "task_index", "task", "prefix", "repair"}
        missing = required - set(item)
        if missing:
            raise ValueError(f"episode {position} missing keys: {sorted(missing)}")
        unknown = set(item) - required - {"continuity", "metadata"}
        if unknown:
            raise ValueError(f"episode {position} has unknown keys: {sorted(unknown)}")
        episode = EpisodeSource(
            source_episode_id=item["source_episode_id"],
            restart_frame=int(item["restart_frame"]),
            task_index=int(item["task_index"]),
            task=str(item["task"]),
            prefix=SegmentSource.parse(base, item["prefix"], f"episode[{position}].prefix"),
            repair=SegmentSource.parse(base, item["repair"], f"episode[{position}].repair"),
            continuity=_resolve_continuity(base, item.get("continuity") or {}),
            metadata=dict(item.get("metadata") or {}),
        )
        if episode.restart_frame <= 0:
            raise ValueError(f"episode {position} restart_frame must be positive")
        episodes.append(episode)
    identities = [str(item.source_episode_id) for item in episodes]
    if len(identities) != len(set(identities)):
        raise ValueError("episode manifest contains duplicate source_episode_id values")
    return episodes
