"""Render complete trajectories with zero-arm loss-masked frames highlighted."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
import pyarrow.parquet as pq

from .continuity import require
from .demo_videos import _sha256, _writer
from .media import decode_embedded_image


HEADER_HEIGHT = 64
MASK_COLOR = np.asarray((255, 0, 80), dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class ZeroArmMaskSettings:
    """Thresholds used by the OpenPI zero-arm loss mask."""

    mode: str = "command"
    arm_dims: int = 6
    position_threshold_m: float = 0.002
    orientation_threshold_rad: float = 0.02
    position_action_scale_m: float = 0.05
    orientation_action_scale_rad: float = 0.5
    gripper_change_eps: float = 1.0e-4
    gripper_state_change_threshold: float = 5.0e-5
    keep_chunk_start: bool = True


@dataclasses.dataclass(frozen=True)
class ZeroArmMaskTimeline:
    """Trajectory-wide base mask and the signals that preserve supervision."""

    zero_masked: np.ndarray
    supervised: np.ndarray
    arm_motion: np.ndarray
    gripper_command_change: np.ndarray
    gripper_state_change: np.ndarray
    arm_norm: np.ndarray
    position_norm_m: np.ndarray
    orientation_norm_rad: np.ndarray


def compute_base_zero_arm_mask(
    actions: np.ndarray,
    states: np.ndarray,
    settings: ZeroArmMaskSettings,
) -> ZeroArmMaskTimeline:
    """Compute a stable per-frame view of the chunk-local training mask.

    OpenPI always supervises chunk slot 0 and the final chunk slot. Those
    context-dependent exceptions are excluded from this stable per-frame view.
    """

    actions = np.asarray(actions)
    states = np.asarray(states)
    require(actions.ndim == 2, f"expected [time, action_dim], got {actions.shape}")
    require(states.ndim == 2, f"expected [time, state_dim], got {states.shape}")
    require(len(actions) == len(states), "actions and states must have the same length")
    require(actions.shape[1] >= settings.arm_dims, "action dimension is too small")
    require(settings.arm_dims >= 6, "zero-arm visualization requires six arm dims")

    require(settings.mode in {"command", "state_delta"}, "mode must be command or state_delta")
    if settings.mode == "state_delta":
        require(states.shape[1] >= 6, "state_delta mode requires six end-effector state dims")
        position_norm_m = np.zeros(len(actions), dtype=np.float32)
        orientation_norm_rad = np.zeros(len(actions), dtype=np.float32)
        if len(actions) > 1:
            position_norm_m[:-1] = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=-1)
            orientation_norm_rad[:-1] = np.linalg.norm(
                np.diff(states[:, 3:6], axis=0), axis=-1
            )
        arm_norm = np.hypot(position_norm_m, orientation_norm_rad)
    else:
        arm_norm = np.linalg.norm(actions[:, : settings.arm_dims], axis=-1)
        position_norm_m = np.linalg.norm(
            actions[:, :3] * settings.position_action_scale_m, axis=-1
        )
        orientation_norm_rad = np.linalg.norm(
            actions[:, 3:6] * settings.orientation_action_scale_rad, axis=-1
        )
    arm_motion = (position_norm_m > settings.position_threshold_m) | (
        orientation_norm_rad > settings.orientation_threshold_rad
    )

    gripper_command_change = np.zeros(len(actions), dtype=np.bool_)
    if actions.shape[1] >= 7 and len(actions) > 1:
        gripper_command_change[1:] = np.abs(np.diff(actions[:, 6])) > settings.gripper_change_eps

    gripper_state_change = np.zeros(len(actions), dtype=np.bool_)
    if states.shape[1] > 6 and len(states) > 1:
        physical_delta = np.max(np.abs(np.diff(states[:, 6:], axis=0)), axis=-1)
        gripper_state_change[:-1] = physical_delta > settings.gripper_state_change_threshold

    supervised = arm_motion | gripper_command_change | gripper_state_change
    return ZeroArmMaskTimeline(
        zero_masked=~supervised,
        supervised=supervised,
        arm_motion=arm_motion,
        gripper_command_change=gripper_command_change,
        gripper_state_change=gripper_state_change,
        arm_norm=arm_norm,
        position_norm_m=position_norm_m,
        orientation_norm_rad=orientation_norm_rad,
    )


def _mask_runs(mask: np.ndarray) -> list[dict[str, int]]:
    runs: list[dict[str, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append({"start_frame": start, "end_frame": index - 1, "length": index - start})
            start = None
    return runs


def _phase(frame_index: int, splice: int | None, trajectory_kind: str | None) -> str:
    if splice is None:
        if trajectory_kind == "evaluator_confirmed_vla_success":
            return "VLA SUCCESS"
        return "REPAIR ONLY"
    return "VLA PREFIX" if frame_index < splice else "REPAIR SUFFIX"


def _frame(
    images: Sequence[np.ndarray],
    camera_columns: Sequence[str],
    *,
    episode_index: int,
    frame_index: int,
    length: int,
    splice: int | None,
    trajectory_kind: str | None,
    timeline: ZeroArmMaskTimeline,
    settings: ZeroArmMaskSettings,
) -> np.ndarray:
    require(bool(images), "video requires at least one camera")
    require(
        all(
            image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3 for image in images
        ),
        "video images must be uint8 HWC RGB",
    )
    height = images[0].shape[0]
    require(all(image.shape[0] == height for image in images), "camera heights differ")
    widths = [image.shape[1] for image in images]
    masked = bool(timeline.zero_masked[frame_index])
    header_color = (82, 5, 30) if masked else (18, 43, 27)
    status_color = (255, 95, 130) if masked else (100, 235, 140)

    canvas = np.zeros((height + HEADER_HEIGHT, sum(widths), 3), dtype=np.uint8)
    canvas[:HEADER_HEIGHT] = header_color
    offset = 0
    for image, column, width in zip(images, camera_columns, widths, strict=True):
        rendered = image
        if masked:
            rendered = np.clip(image.astype(np.float32) * 0.62 + MASK_COLOR * 0.38, 0, 255).astype(
                np.uint8
            )
        canvas[HEADER_HEIGHT:, offset : offset + width] = rendered
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

    status = "ZERO-ARM LOSS MASKED" if masked else "SUPERVISED"
    title = (
        f"{status} | ep {episode_index:03d} | frame {frame_index:03d}/{length - 1:03d}"
        f" | {_phase(frame_index, splice, trajectory_kind)}"
    )
    detail = (
        f"{settings.mode} pos {timeline.position_norm_m[frame_index]:.5f}/"
        f"{settings.position_threshold_m:.4f}m"
        f" | rot {timeline.orientation_norm_rad[frame_index]:.5f}/"
        f"{settings.orientation_threshold_rad:.3f}rad"
        f" | grip_cmd {int(timeline.gripper_command_change[frame_index])}"
        f" | grip_state {int(timeline.gripper_state_change[frame_index])}"
    )
    cv2.putText(
        canvas,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        status_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        detail,
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    if masked:
        border = 7
        canvas[HEADER_HEIGHT : HEADER_HEIGHT + border] = MASK_COLOR
        canvas[-border:] = MASK_COLOR
        canvas[HEADER_HEIGHT:, :border] = MASK_COLOR
        canvas[HEADER_HEIGHT:, -border:] = MASK_COLOR
    return np.ascontiguousarray(canvas)


def _settings_dict(settings: ZeroArmMaskSettings) -> dict[str, Any]:
    return dataclasses.asdict(settings)


def render_zero_mask_videos(
    dataset: Path,
    output: Path,
    *,
    episode_indices: Sequence[int],
    camera_columns: Sequence[str] = ("image", "wrist_image"),
    fps: int = 20,
    settings: ZeroArmMaskSettings = ZeroArmMaskSettings(),
) -> dict[str, Any]:
    """Render selected complete trajectories and write a diagnostic manifest."""

    dataset = dataset.resolve()
    output = output.resolve()
    require(dataset.is_dir(), f"missing dataset: {dataset}")
    require(not output.exists(), f"refusing to overwrite video output: {output}")
    require(bool(episode_indices), "at least one episode must be selected")
    require(len(set(episode_indices)) == len(episode_indices), "episode indices must be unique")
    require(bool(camera_columns), "at least one camera is required")
    build = json.loads((dataset / "meta/build_manifest.json").read_text(encoding="utf-8"))
    by_episode = {int(entry["episode_index"]): entry for entry in build.get("episodes") or []}
    missing = [index for index in episode_indices if index not in by_episode]
    require(not missing, f"episodes missing from build manifest: {missing}")

    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for selection_index, episode_index in enumerate(episode_indices):
            entry = by_episode[episode_index]
            source_metadata = entry.get("source_metadata") or {}
            trajectory_kind = source_metadata.get("trajectory_kind")
            raw_splice = entry.get("splice_frame_index")
            splice = None if raw_splice is None else int(raw_splice)
            columns = [*camera_columns, "actions", "state"]
            table = pq.read_table(dataset / entry["parquet"], columns=columns)
            actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
            states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
            timeline = compute_base_zero_arm_mask(actions, states, settings)
            length = len(table)
            first_images = [
                decode_embedded_image(table[column][0].as_py()) for column in camera_columns
            ]
            width = sum(image.shape[1] for image in first_images)
            height = first_images[0].shape[0] + HEADER_HEIGHT
            stem = f"zero_mask_demo_{selection_index + 1:02d}_episode_{episode_index:03d}"
            final_path = output / f"{stem}.mp4"
            temporary = output / f".{stem}.incomplete.mp4"
            writer: Iterator[Any] = _writer(temporary, width=width, height=height, fps=fps)
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
                            frame_index=frame_index,
                            length=length,
                            splice=splice,
                            trajectory_kind=trajectory_kind,
                            timeline=timeline,
                            settings=settings,
                        )
                    )
            finally:
                writer.close()
            os.replace(temporary, final_path)
            masked_count = int(timeline.zero_masked.sum())
            source_episode = (entry.get("source_metadata") or {}).get(
                "source_rollout_episode_id", entry.get("source_episode_id")
            )
            records.append(
                {
                    "selection_index": selection_index,
                    "episode_index": episode_index,
                    "source_episode_id": source_episode,
                    "episode_mode": entry.get("episode_mode"),
                    "trajectory_kind": trajectory_kind,
                    "splice_frame_index": splice,
                    "frame_count": length,
                    "zero_masked_frame_count": masked_count,
                    "zero_masked_frame_fraction": masked_count / length,
                    "supervised_frame_count": int(timeline.supervised.sum()),
                    "arm_motion_frame_count": int(timeline.arm_motion.sum()),
                    "gripper_command_change_frame_count": int(
                        timeline.gripper_command_change.sum()
                    ),
                    "gripper_state_change_frame_count": int(timeline.gripper_state_change.sum()),
                    "zero_mask_runs": _mask_runs(timeline.zero_masked),
                    "video": final_path.name,
                    "video_sha256": _sha256(final_path),
                }
            )
            print(
                f"rendered episode {episode_index}: {masked_count}/{length} zero-masked frames",
                flush=True,
            )
    except Exception:
        for path in output.glob("*.incomplete.mp4"):
            path.unlink()
        raise

    manifest = {
        "schema_version": 1,
        "dataset": str(dataset),
        "selection": "explicit_episode_indices",
        "episode_indices": list(episode_indices),
        "video_count": len(records),
        "fps": fps,
        "camera_columns": list(camera_columns),
        "highlight": {
            "meaning": "red-magenta tint and border means zero-arm loss masked",
            "mask_semantics": (
                "base per-frame mask for an ordinary horizon slot; excludes OpenPI's "
                "chunk-local keep_chunk_start and final-slot forced-supervision exceptions"
            ),
            "supervised_when": (
                f"{settings.mode} arm physical motion exceeds either threshold, gripper command changes, "
                "or next-frame physical gripper state changes"
            ),
        },
        "training_mask_settings": _settings_dict(settings),
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
    parser.add_argument("--episode", action="append", type=int, dest="episodes", required=True)
    parser.add_argument("--camera", action="append", dest="cameras")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--mode", choices=("command", "state_delta"), default="command")
    parser.add_argument("--position-threshold-m", type=float, default=0.002)
    parser.add_argument("--orientation-threshold-rad", type=float, default=0.02)
    parser.add_argument("--position-action-scale-m", type=float, default=0.05)
    parser.add_argument("--orientation-action-scale-rad", type=float, default=0.5)
    parser.add_argument("--gripper-change-eps", type=float, default=1.0e-4)
    parser.add_argument("--gripper-state-change-threshold", type=float, default=5.0e-5)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else dataset / "meta/visualization/zero_arm_mask_demos"
    )
    settings = ZeroArmMaskSettings(
        mode=args.mode,
        position_threshold_m=args.position_threshold_m,
        orientation_threshold_rad=args.orientation_threshold_rad,
        position_action_scale_m=args.position_action_scale_m,
        orientation_action_scale_rad=args.orientation_action_scale_rad,
        gripper_change_eps=args.gripper_change_eps,
        gripper_state_change_threshold=args.gripper_state_change_threshold,
    )
    manifest = render_zero_mask_videos(
        dataset,
        output,
        episode_indices=tuple(args.episodes),
        camera_columns=tuple(args.cameras or ("image", "wrist_image")),
        fps=args.fps,
        settings=settings,
    )
    print(json.dumps({"output": str(output), "video_count": manifest["video_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
