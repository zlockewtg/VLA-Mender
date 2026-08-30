"""Torchrun worker that injects one resolved YAML config into pinned OpenPI."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import logging
from pathlib import Path
import sys
from typing import Any

from .config import load_training_config, validate_training_inputs
from .trainable_filter import TrainableIndexDataset, load_manifest


def _load_trainer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("vla_mender_openpi_train_pytorch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import OpenPI PyTorch trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(settings_path: Path) -> None:
    settings = load_training_config(settings_path)
    # The parent launcher checks checkpoint state once. Repeating that check in
    # every torchrun rank races with rank 0 creating the directory.
    validate_training_inputs(settings, check_checkpoint_state=False)
    openpi_src = str(settings.openpi_source / "src")
    if openpi_src not in sys.path:
        sys.path.insert(0, openpi_src)

    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    import openpi.training.config as openpi_config
    import openpi.training.data_loader as openpi_data
    import openpi.training.optimizer as openpi_optimizer
    import openpi.transforms as openpi_transforms

    base = openpi_config.get_config(settings.base_config)
    if not hasattr(base.data, "extra_delta_transform"):
        raise TypeError(f"OpenPI base config is not a LIBERO data config: {settings.base_config}")
    model = dataclasses.replace(base.model, action_horizon=settings.action_horizon)
    data_base = dataclasses.replace(
        base.data.base_config or openpi_config.DataConfig(),
        mask_zero_arm_action_loss=settings.mask_zero_arm_action_loss,
        zero_arm_mask_mode=settings.zero_arm_mask_mode,
        zero_arm_action_dims=settings.zero_arm_action_dims,
        zero_arm_position_threshold_m=settings.zero_arm_position_threshold_m,
        zero_arm_orientation_threshold_rad=settings.zero_arm_orientation_threshold_rad,
        zero_arm_position_action_scale_m=settings.zero_arm_position_action_scale_m,
        zero_arm_orientation_action_scale_rad=settings.zero_arm_orientation_action_scale_rad,
        zero_arm_gripper_change_eps=settings.zero_arm_gripper_change_eps,
        zero_arm_gripper_state_change_threshold=settings.zero_arm_gripper_state_change_threshold,
        zero_arm_keep_chunk_start=settings.zero_arm_keep_chunk_start,
    )
    data = dataclasses.replace(
        base.data,
        repo_id=str(settings.dataset),
        assets=openpi_config.AssetsConfig(
            assets_dir=str(settings.initialization_checkpoint),
            asset_id=settings.normalization_asset_id,
        ),
        base_config=data_base,
        extra_delta_transform=settings.extra_delta_transform,
    )
    lr = openpi_optimizer.CosineDecaySchedule(
        warmup_steps=settings.learning_rate.warmup_steps,
        peak_lr=settings.learning_rate.peak_lr,
        decay_steps=settings.learning_rate.decay_steps,
        decay_lr=settings.learning_rate.decay_lr,
    )
    config = dataclasses.replace(
        base,
        name=settings.run_name,
        project_name=settings.project_name,
        exp_name=settings.experiment_name,
        model=model,
        data=data,
        pytorch_weight_path=str(settings.initialization_checkpoint),
        checkpoint_base_dir=str(settings.checkpoint_base_dir),
        lr_schedule=lr,
        seed=settings.seed,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        num_train_steps=settings.num_train_steps,
        log_interval=settings.log_interval,
        save_interval=settings.save_interval,
        keep_period=settings.keep_period,
        overwrite=settings.overwrite,
        resume=settings.resume,
        wandb_enabled=settings.wandb_enabled,
        pytorch_training_precision=settings.pytorch_training_precision,
    )
    manifest = (
        load_manifest(settings.trainable_index_manifest)
        if settings.sampling_mode == "transition_aware"
        else None
    )

    def create_filtered_dataset(data_config: Any, action_horizon: int, model_config: Any) -> Any:
        repo_id = data_config.repo_id
        if repo_id is None or repo_id == "fake":
            raise ValueError("post-training requires a concrete LeRobot dataset")
        metadata = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        delta_timestamps = {
            key: [step / metadata.fps for step in range(action_horizon)]
            for key in data_config.action_sequence_keys
        }
        use_zero_arm_mask = data_config.mask_zero_arm_action_loss
        if use_zero_arm_mask:
            delta_timestamps["state"] = [step / metadata.fps for step in range(action_horizon)]
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            delta_timestamps=delta_timestamps,
        )
        assert manifest is not None
        dataset = TrainableIndexDataset(dataset, manifest, action_horizon)
        logging.info(
            "Applied VLA-Mender trainable-index filter: %d starts, horizon=%d",
            len(dataset),
            action_horizon,
        )
        if use_zero_arm_mask:
            dataset = openpi_data.GripperStateChunkDataset(dataset, action_horizon)
        if data_config.mask_zero_arm_action_loss:
            dataset = openpi_data.TransformedDataset(
                dataset,
                [
                    openpi_transforms.ZeroArmActionLossMask(
                        mode=data_config.zero_arm_mask_mode,
                        arm_dims=data_config.zero_arm_action_dims,
                        position_threshold_m=data_config.zero_arm_position_threshold_m,
                        orientation_threshold_rad=data_config.zero_arm_orientation_threshold_rad,
                        position_action_scale_m=data_config.zero_arm_position_action_scale_m,
                        orientation_action_scale_rad=data_config.zero_arm_orientation_action_scale_rad,
                        gripper_change_eps=data_config.zero_arm_gripper_change_eps,
                        gripper_state_change_threshold=data_config.zero_arm_gripper_state_change_threshold,
                        keep_chunk_start=data_config.zero_arm_keep_chunk_start,
                    )
                ],
            )
        if data_config.prompt_from_task:
            dataset = openpi_data.TransformedDataset(
                dataset, [openpi_transforms.PromptFromLeRobotTask(metadata.tasks)]
            )
        return dataset

    if settings.sampling_mode == "transition_aware":
        openpi_data.create_torch_dataset = create_filtered_dataset
    else:
        logging.info(
            "Using native LeRobot sampling: every dataset row is a start; "
            "action chunks may cross splice boundaries and use tail padding"
        )
    trainer = _load_trainer(settings.openpi_source / "scripts/train_pytorch.py")
    trainer.init_logging()
    trainer.train_loop(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args(argv)
    _run(args.settings.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
