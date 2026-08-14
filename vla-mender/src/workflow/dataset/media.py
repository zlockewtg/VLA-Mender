"""Image and video adapters for embedded-image LeRobot datasets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np

from .continuity import require


def decode_embedded_image(item: dict[str, Any]) -> np.ndarray:
    payload = item.get("bytes")
    if payload is None and item.get("path"):
        payload = Path(str(item["path"])).read_bytes()
    require(isinstance(payload, bytes), "embedded image contains neither bytes nor a readable path")
    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    require(bgr is not None, "failed to decode embedded image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_png(rgb: np.ndarray) -> dict[str, Any]:
    require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, "image must be uint8 HWC RGB")
    ok, payload = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    require(bool(ok), "failed to encode PNG")
    return {"bytes": payload.tobytes(), "path": None}


def transform_image(rgb: np.ndarray, *, horizontal_flip: bool) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, ::-1] if horizontal_flip else rgb)


def validate_shape(rgb: np.ndarray, *, width: int, height: int, label: str) -> None:
    require(rgb.shape == (height, width, 3), f"{label} has shape {rgb.shape}, expected {(height, width, 3)}")


def iter_video_frames(path: Path) -> Iterator[np.ndarray]:
    require(path.is_file(), f"missing video: {path}")
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        width, height = (int(value) for value in metadata["size"])
        for payload in reader:
            yield np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
    finally:
        reader.close()


def video_frame(path: Path, index: int) -> np.ndarray:
    require(index >= 0, "video frame index must be non-negative")
    for frame_index, frame in enumerate(iter_video_frames(path)):
        if frame_index == index:
            return frame
    raise RuntimeError(f"video ended before frame {index}: {path}")


def video_png_frames(
    path: Path,
    length: int,
    *,
    width: int,
    height: int,
    horizontal_flip: bool,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    encoded: list[dict[str, Any]] = []
    decoded: list[np.ndarray] = []
    for frame_index, rgb in enumerate(iter_video_frames(path)):
        if frame_index >= length:
            break
        rgb = transform_image(rgb, horizontal_flip=horizontal_flip)
        validate_shape(rgb, width=width, height=height, label=f"{path} frame {frame_index}")
        decoded.append(rgb)
        encoded.append(encode_png(rgb))
    require(len(encoded) == length, f"video {path} yielded {len(encoded)} frames, expected {length}")
    return encoded, decoded


def embedded_png_frames(
    items: Sequence[dict[str, Any]],
    *,
    width: int,
    height: int,
    horizontal_flip: bool = False,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    encoded: list[dict[str, Any]] = []
    decoded: list[np.ndarray] = []
    for index, item in enumerate(items):
        rgb = transform_image(decode_embedded_image(item), horizontal_flip=horizontal_flip)
        validate_shape(rgb, width=width, height=height, label=f"embedded image {index}")
        decoded.append(rgb)
        encoded.append(encode_png(rgb))
    return encoded, decoded
