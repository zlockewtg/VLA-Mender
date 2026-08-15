"""Render annotated full and boundary-clip videos for spliced dataset episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import cv2
import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq

from .continuity import require
from .media import decode_embedded_image


HEADER_HEIGHT = 48


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _writer(path: Path, *, width: int, height: int, fps: int) -> Iterator[Any]:
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "20", "-preset", "medium", "-movflags", "+faststart"],
    )
    writer.send(None)
    return writer


def _annotated_frame(
    agent: np.ndarray,
    wrist: np.ndarray,
    *,
    episode_index: int,
    source_episode: int | str,
    frame_index: int,
    length: int,
    splice: int,
) -> np.ndarray:
    require(agent.shape == wrist.shape, "agent and wrist images have different shapes")
    height, width, channels = agent.shape
    require(channels == 3, "visualization expects RGB images")
    canvas = np.zeros((height + HEADER_HEIGHT, width * 2, 3), dtype=np.uint8)
    canvas[:HEADER_HEIGHT] = (22, 24, 29)
    canvas[HEADER_HEIGHT:, :width] = agent
    canvas[HEADER_HEIGHT:, width:] = wrist
    phase = "VLA PREFIX" if frame_index < splice else "REPAIR SUFFIX"
    phase_color = (75, 220, 120) if frame_index < splice else (255, 170, 55)
    title = (
        f"ep {episode_index:03d} | source {source_episode} | "
        f"frame {frame_index:03d}/{length - 1:03d} | {phase} | splice {splice}"
    )
    cv2.putText(
        canvas,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        phase_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "AGENT VIEW",
        (8, HEADER_HEIGHT + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "WRIST VIEW",
        (width + 8, HEADER_HEIGHT + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if frame_index == splice - 1:
        boundary = "LAST PREFIX FRAME"
    elif frame_index == splice:
        boundary = "FIRST REPAIR FRAME"
    else:
        boundary = ""
    if abs(frame_index - splice) <= 2:
        canvas[:4] = (255, 55, 55)
        canvas[-4:] = (255, 55, 55)
        canvas[:, :4] = (255, 55, 55)
        canvas[:, -4:] = (255, 55, 55)
    if boundary:
        cv2.putText(
            canvas,
            boundary,
            (width - 92, height + HEADER_HEIGHT - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 55, 55),
            2,
            cv2.LINE_AA,
        )
    return np.ascontiguousarray(canvas)


def render_splice_videos(
    dataset: Path,
    output: Path,
    *,
    fps: int = 20,
    clip_pre_frames: int = 40,
    clip_post_frames: int = 60,
) -> dict[str, Any]:
    """Render every prefix_plus_repair episode and write a hashed index."""

    require(dataset.is_dir(), f"missing dataset: {dataset}")
    require(not output.exists(), f"refusing to overwrite visualization output: {output}")
    build = json.loads((dataset / "meta/build_manifest.json").read_text(encoding="utf-8"))
    selected = [
        row for row in build["episodes"] if row.get("episode_mode") == "prefix_plus_repair"
    ]
    require(bool(selected), "dataset has no prefix_plus_repair episodes")
    output.mkdir(parents=True)
    full_dir = output / "full"
    clip_dir = output / "clips"
    full_dir.mkdir()
    clip_dir.mkdir()
    records: list[dict[str, Any]] = []
    try:
        for entry in selected:
            episode_index = int(entry["episode_index"])
            source_episode = (entry.get("source_metadata") or {}).get(
                "source_rollout_episode_id", entry["source_episode_id"]
            )
            splice = int(entry["splice_frame_index"])
            table = pq.read_table(
                dataset / entry["parquet"], columns=["image", "wrist_image"]
            )
            length = len(table)
            stem = (
                f"episode_{episode_index:03d}_source_{int(source_episode):03d}_"
                f"splice_{splice:03d}"
            )
            full_path = full_dir / f"{stem}_full.mp4"
            clip_path = clip_dir / f"{stem}_clip.mp4"
            temporary_full = full_path.with_name(f".{full_path.stem}.incomplete.mp4")
            temporary_clip = clip_path.with_name(f".{clip_path.stem}.incomplete.mp4")
            full_writer = _writer(temporary_full, width=512, height=304, fps=fps)
            clip_writer = _writer(temporary_clip, width=512, height=304, fps=fps)
            clip_start = max(0, splice - clip_pre_frames)
            clip_stop = min(length, splice + clip_post_frames)
            try:
                for frame_index in range(length):
                    frame = _annotated_frame(
                        decode_embedded_image(table["image"][frame_index].as_py()),
                        decode_embedded_image(table["wrist_image"][frame_index].as_py()),
                        episode_index=episode_index,
                        source_episode=source_episode,
                        frame_index=frame_index,
                        length=length,
                        splice=splice,
                    )
                    full_writer.send(frame)
                    if clip_start <= frame_index < clip_stop:
                        clip_writer.send(frame)
            finally:
                full_writer.close()
                clip_writer.close()
            os.replace(temporary_full, full_path)
            os.replace(temporary_clip, clip_path)
            records.append(
                {
                    "episode_index": episode_index,
                    "source_episode_id": source_episode,
                    "splice_frame_index": splice,
                    "full_frame_count": length,
                    "clip_range": [clip_start, clip_stop],
                    "full_video": str(full_path.relative_to(output)),
                    "full_video_sha256": _sha256(full_path),
                    "clip_video": str(clip_path.relative_to(output)),
                    "clip_video_sha256": _sha256(clip_path),
                }
            )
    except Exception:
        for path in output.rglob("*.incomplete.mp4"):
            path.unlink()
        raise
    manifest = {
        "schema_version": 1,
        "dataset": str(dataset.resolve()),
        "fps": fps,
        "resolution": [512, 304],
        "layout": "agent_view_left__wrist_view_right",
        "clip_pre_frames": clip_pre_frames,
        "clip_post_frames": clip_post_frames,
        "episode_count": len(records),
        "episodes": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--clip-pre-frames", type=int, default=40)
    parser.add_argument("--clip-post-frames", type=int, default=60)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else dataset / "meta/visualization/splice_videos"
    )
    manifest = render_splice_videos(
        dataset,
        output,
        fps=args.fps,
        clip_pre_frames=args.clip_pre_frames,
        clip_post_frames=args.clip_post_frames,
    )
    print(
        json.dumps(
            {"output": str(output), "episode_count": manifest["episode_count"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
