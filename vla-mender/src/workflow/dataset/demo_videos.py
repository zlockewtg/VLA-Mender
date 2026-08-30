"""Render a deterministic small set of complete trajectory demo videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq

from .continuity import require
from .media import decode_embedded_image


HEADER_HEIGHT = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _writer(path: Path, *, width: int, height: int, fps: int) -> Iterator[Any]:
    require(width % 2 == 0 and height % 2 == 0, "demo video dimensions must be even")
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


def _evenly_spaced(entries: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    require(count > 0, "demo video count must be positive")
    if len(entries) <= count:
        return list(entries)
    if count == 1:
        return [entries[0]]
    positions = [round(index * (len(entries) - 1) / (count - 1)) for index in range(count)]
    return [entries[position] for position in positions]


def _frame(
    images: Sequence[np.ndarray],
    camera_columns: Sequence[str],
    *,
    episode_index: int,
    source_episode: int | str,
    frame_index: int,
    length: int,
    splice: int | None,
) -> np.ndarray:
    require(bool(images), "demo requires at least one camera")
    height = images[0].shape[0]
    require(
        all(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3 for image in images),
        "demo images must be uint8 HWC RGB",
    )
    require(all(image.shape[0] == height for image in images), "demo camera heights differ")
    widths = [image.shape[1] for image in images]
    canvas = np.zeros((height + HEADER_HEIGHT, sum(widths), 3), dtype=np.uint8)
    canvas[:HEADER_HEIGHT] = (22, 24, 29)
    offset = 0
    for image, column, width in zip(images, camera_columns, widths, strict=True):
        canvas[HEADER_HEIGHT:, offset : offset + width] = image
        cv2.putText(
            canvas,
            str(column).upper(),
            (offset + 8, HEADER_HEIGHT + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        offset += width
    if splice is None:
        phase = "REPAIR ONLY"
        phase_color = (255, 170, 55)
        splice_label = "none"
    elif frame_index < splice:
        phase = "VLA PREFIX"
        phase_color = (75, 220, 120)
        splice_label = str(splice)
    else:
        phase = "REPAIR SUFFIX"
        phase_color = (255, 170, 55)
        splice_label = str(splice)
    step_title = f"STEP {frame_index:03d} / {length - 1:03d}"
    cv2.putText(
        canvas,
        step_title,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        phase_color,
        2,
        cv2.LINE_AA,
    )
    metadata = (
        f"EP {episode_index:03d} | {phase} | splice {splice_label} | "
        f"source {source_episode}"
    )
    cv2.putText(
        canvas,
        metadata,
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (225, 228, 235),
        1,
        cv2.LINE_AA,
    )
    if splice is not None and abs(frame_index - splice) <= 2:
        canvas[:4] = (255, 55, 55)
        canvas[-4:] = (255, 55, 55)
        canvas[:, :4] = (255, 55, 55)
        canvas[:, -4:] = (255, 55, 55)
    return np.ascontiguousarray(canvas)


def render_demo_videos(
    dataset: Path,
    output: Path,
    *,
    camera_columns: Sequence[str] = ("image", "wrist_image"),
    fps: int = 20,
    count: int = 3,
    dataset_identity: Path | None = None,
) -> dict[str, Any]:
    """Render complete demos selected uniformly over final episode order."""

    dataset = dataset.resolve()
    output = output.resolve()
    require(dataset.is_dir(), f"missing dataset: {dataset}")
    require(not output.exists(), f"refusing to overwrite demo output: {output}")
    require(bool(camera_columns), "demo videos require at least one camera column")
    build = json.loads((dataset / "meta/build_manifest.json").read_text(encoding="utf-8"))
    entries = list(build.get("episodes") or [])
    require(bool(entries), "dataset build manifest has no episodes")
    selected = _evenly_spaced(entries, min(count, len(entries)))
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for selection_index, entry in enumerate(selected):
            episode_index = int(entry["episode_index"])
            source_episode = (entry.get("source_metadata") or {}).get(
                "source_rollout_episode_id", entry["source_episode_id"]
            )
            raw_splice = entry.get("splice_frame_index")
            splice = None if raw_splice is None else int(raw_splice)
            table = pq.read_table(dataset / entry["parquet"], columns=list(camera_columns))
            length = len(table)
            first_images = [
                decode_embedded_image(table[column][0].as_py()) for column in camera_columns
            ]
            width = sum(image.shape[1] for image in first_images)
            height = first_images[0].shape[0] + HEADER_HEIGHT
            stem = f"demo_{selection_index + 1:02d}_episode_{episode_index:03d}"
            final_path = output / f"{stem}.mp4"
            temporary = output / f".{stem}.incomplete.mp4"
            writer = _writer(temporary, width=width, height=height, fps=fps)
            try:
                for frame_index in range(length):
                    images = [
                        decode_embedded_image(table[column][frame_index].as_py())
                        for column in camera_columns
                    ]
                    writer.send(
                        _frame(
                            images,
                            camera_columns,
                            episode_index=episode_index,
                            source_episode=source_episode,
                            frame_index=frame_index,
                            length=length,
                            splice=splice,
                        )
                    )
            finally:
                writer.close()
            os.replace(temporary, final_path)
            records.append(
                {
                    "selection_index": selection_index,
                    "episode_index": episode_index,
                    "source_episode_id": source_episode,
                    "episode_mode": entry.get("episode_mode"),
                    "splice_frame_index": splice,
                    "frame_count": length,
                    "video": final_path.name,
                    "video_sha256": _sha256(final_path),
                }
            )
    except Exception:
        for path in output.glob("*.incomplete.mp4"):
            path.unlink()
        raise
    manifest = {
        "schema_version": 1,
        "dataset": str((dataset_identity or dataset).resolve()),
        "selection": "evenly_spaced_final_episode_order",
        "requested_count": count,
        "demo_count": len(records),
        "fps": fps,
        "camera_columns": list(camera_columns),
        "layout": "camera_columns_left_to_right_with_phase_header",
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
    parser.add_argument("--camera", action="append", dest="cameras")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else dataset / "meta/visualization/trajectory_demos"
    )
    manifest = render_demo_videos(
        dataset,
        output,
        camera_columns=tuple(args.cameras or ("image", "wrist_image")),
        fps=args.fps,
        count=args.count,
    )
    print(json.dumps({"output": str(output), "demo_count": manifest["demo_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
