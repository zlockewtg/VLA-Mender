from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from workflow.training.config import load_training_config


def _settings(tmp_path: Path) -> dict[str, object]:
    return {
        "openpi_source": str(tmp_path / "openpi"),
        "openpi_environment": str(tmp_path / "venv"),
        "openpi_commit": "deadbeef",
        "base_config": "pi0_libero",
        "run_name": "run",
        "experiment_name": "experiment",
        "checkpoint_base_dir": str(tmp_path / "checkpoints"),
        "initialization_checkpoint": str(tmp_path / "initialization"),
        "normalization_asset_id": "physical-intelligence/libero",
        "dataset": str(tmp_path / "dataset"),
        "sampling_mode": "native",
        "gpus": [0, 1, 2, 3],
    }


def test_parses_reference_zero_arm_loss_mask(tmp_path: Path) -> None:
    raw = _settings(tmp_path)
    raw.update(
        mask_zero_arm_action_loss=True,
        zero_arm_mask_mode="state_delta",
        zero_arm_action_dims=6,
        zero_arm_position_threshold_m=0.002,
        zero_arm_orientation_threshold_rad=0.02,
        zero_arm_position_action_scale_m=0.05,
        zero_arm_orientation_action_scale_rad=0.5,
        zero_arm_gripper_change_eps=1.0e-4,
        zero_arm_gripper_state_change_threshold=5.0e-5,
        zero_arm_keep_chunk_start=True,
    )
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_training_config(path)

    assert config.mask_zero_arm_action_loss is True
    assert config.zero_arm_mask_mode == "state_delta"
    assert config.zero_arm_position_threshold_m == 0.002
    assert config.zero_arm_orientation_threshold_rad == 0.02
    assert config.zero_arm_gripper_change_eps == 1.0e-4
    assert config.zero_arm_keep_chunk_start is True


def test_physical_zero_arm_thresholds_are_defaults(tmp_path: Path) -> None:
    raw = _settings(tmp_path)
    raw.update(mask_zero_arm_action_loss=True)
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_training_config(path)

    assert config.zero_arm_position_threshold_m == 0.002
    assert config.zero_arm_orientation_threshold_rad == 0.02
    assert config.zero_arm_gripper_change_eps == 1.0e-4


def test_rejects_removed_norm_only_mode(tmp_path: Path) -> None:
    raw = _settings(tmp_path)
    raw.update(
        mask_zero_arm_action_loss=True,
        zero_arm_action_norm_threshold=0.02,
        zero_arm_position_threshold_m=None,
        zero_arm_orientation_threshold_rad=None,
    )
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="norm masking was removed"):
        load_training_config(path)
