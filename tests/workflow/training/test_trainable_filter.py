from __future__ import annotations

import pytest

from workflow.training.trainable_filter import TrainableIndexDataset, valid_global_indices


def _manifest() -> dict[str, object]:
    frames = []
    for index in range(6):
        prefix = index < 3
        frames.append(
            {
                "index": index,
                "episode_index": 0,
                "frame_index": index,
                "phase": "original_vla_prefix" if prefix else "repair_suffix",
                "trainable": index not in {2, 5},
                "continuous_segment_id": "0:prefix" if prefix else "0:repair",
            }
        )
    return {
        "schema_version": 1,
        "action_horizon": 2,
        "valid_start_count": 2,
        "frames": frames,
    }


def test_valid_starts_do_not_cross_splice_or_terminal() -> None:
    assert valid_global_indices(_manifest(), 2) == (0, 3)


def test_declared_valid_count_is_rechecked() -> None:
    class HfDataset:
        column_names = ["index"]

        def __getitem__(self, key: str) -> list[int]:
            assert key == "index"
            return list(range(6))

    class Dataset:
        hf_dataset = HfDataset()

        def __len__(self) -> int:
            return 6

        def __getitem__(self, index: int) -> int:
            return index

    manifest = _manifest()
    manifest["valid_start_count"] = 3
    with pytest.raises(ValueError, match="recomputed 2"):
        TrainableIndexDataset(Dataset(), manifest, 2)


def test_non_contiguous_episode_is_rejected() -> None:
    manifest = _manifest()
    manifest["frames"][1]["frame_index"] = 7  # type: ignore[index]
    with pytest.raises(ValueError, match="not contiguous"):
        valid_global_indices(manifest, 2)
