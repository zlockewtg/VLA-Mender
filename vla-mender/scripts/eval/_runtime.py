#!/usr/bin/env python3
"""Internal process adapters for LIBERO eval workers and scene generation."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from workflow.libero_runtime import LiberoRuntime  # noqa: E402
from workflow.openpi_backend import OpenPIBackend  # noqa: E402
from workflow.parameters import (  # noqa: E402
    BackendSettings,
    ExperimentSettings,
    InitialStateSettings,
    RolloutSettings,
    TaskSettings,
)
from workflow.rollout.evaluator import (  # noqa: E402
    EvaluationConfig,
    EvaluationFrame,
    aggregate_episode_results,
)
from workflow.rollout.runner import (  # noqa: E402
    EpisodeRun,
    EpisodeSpec,
    build_episode_specs,
    run_evaluation_batch,
)
from workflow.rollout.state_provider import (  # noqa: E402
    generate_randomized_states,
    resolve_evaluation_initial_states,
)


ALL_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)
COMPILE_MODES = (
    "none",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)


def parse_worker_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-name", default="pi0_libero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--suites", nargs="+", choices=ALL_SUITES, required=True)
    parser.add_argument("--task-ids", nargs="+", type=int, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--initial-state-indices", nargs="+", type=int)
    parser.add_argument("--custom-initial-state-manifest", type=Path)
    parser.add_argument("--trials-per-initial-state", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-seed-offset", type=int, default=0)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--post-success-steps", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--action-chunk", type=int, default=5)
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--compile-mode", choices=COMPILE_MODES, default="none")
    parser.add_argument("--binary-gripper", action="store_true")
    parser.add_argument("--gripper-hysteresis-threshold", type=float, default=0.2)
    parser.add_argument("--training-schema-only", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--videos-only", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--openpi-source", type=Path)
    parser.add_argument("--openpi-commit")
    parser.add_argument("--openpi-norm-stats", type=Path)
    args = parser.parse_args(argv)

    if len(args.suites) != 1 or len(args.task_ids) != 1:
        parser.error("a worker requires exactly one suite and one task ID")
    if args.task_ids[0] < 0:
        parser.error("--task-ids must be non-negative")
    if args.episodes_per_task <= 0 or args.trials_per_initial_state <= 0:
        parser.error("episode and trial counts must be positive")
    if args.control_freq <= 0 or args.fps != args.control_freq:
        parser.error("--control-freq must be positive and equal --fps")
    if args.max_steps <= 0 or args.action_chunk <= 0 or args.num_inference_steps <= 0:
        parser.error("step, chunk, and inference-step counts must be positive")
    if args.num_steps_wait < 0:
        parser.error("--num-steps-wait must be non-negative")
    if args.post_success_steps != 0:
        parser.error(
            "only native-done evaluation (--post-success-steps 0) is supported"
        )
    if not 0.0 <= args.gripper_hysteresis_threshold <= 1.0:
        parser.error("--gripper-hysteresis-threshold must be in [0, 1]")
    if args.image_writer_threads <= 0:
        parser.error("--image-writer-threads must be positive")
    if args.initial_state_indices is not None and (
        any(index < 0 for index in args.initial_state_indices)
        or len(args.initial_state_indices) != len(set(args.initial_state_indices))
    ):
        parser.error("--initial-state-indices must be unique non-negative integers")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.videos_only and args.resume:
        parser.error("--videos-only does not support --resume")
    if args.videos_only and args.save_images:
        parser.error("--videos-only and --save-images are mutually exclusive")
    return args


def dataset_features(
    *, save_images: bool, training_schema_only: bool
) -> dict[str, Any]:
    visual_dtype = "image" if save_images else "video"
    features: dict[str, Any] = {
        "image": {
            "dtype": visual_dtype,
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": visual_dtype,
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }
    if not training_schema_only:
        features.update(
            {
                "next.reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
                "next.done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
                "next.success": {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": ["success"],
                },
                "next.truncated": {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": ["truncated"],
                },
            }
        )
    return features


def _add_frame(dataset: Any, frame: dict[str, Any], task: str) -> None:
    if "task" in inspect.signature(dataset.add_frame).parameters:
        dataset.add_frame(frame, task=task)
    else:
        dataset.add_frame({**frame, "task": task})


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _safe_remove(output: Path) -> None:
    if output == Path(output.anchor) or len(output.parts) < 3:
        raise ValueError(f"refusing to recursively replace broad output path: {output}")
    shutil.rmtree(output)


def _quarantine_uncommitted_files(output: Path, *, first_episode: int) -> None:
    pattern = re.compile(r"episode_(\d+)\.(?:parquet|mp4)$")
    stale: list[Path] = []
    for directory in (output / "data", output / "videos"):
        if directory.exists():
            for path in directory.rglob("episode_*.*"):
                match = pattern.fullmatch(path.name)
                if match and int(match.group(1)) >= first_episode:
                    stale.append(path)
    if not stale:
        return
    recovery = (
        output.parent
        / f"{output.name}_recovery"
        / f"before_episode_{first_episode:06d}"
    )
    for path in stale:
        target = recovery / path.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target = target.with_name(
                f"{target.stem}.{path.stat().st_mtime_ns}{target.suffix}"
            )
        path.replace(target)


def _settings(args: argparse.Namespace) -> ExperimentSettings:
    selected = args.initial_state_indices or list(range(args.episodes_per_task))
    return ExperimentSettings(
        task=TaskSettings(
            suite=args.suites[0],
            task_id=args.task_ids[0],
            checkpoint=args.checkpoint.resolve(),
            policy_config=args.config_name,
        ),
        initial_states=InitialStateSettings(
            provider="official",
            count=len(selected),
            seed_start=args.seed,
        ),
        rollout=RolloutSettings(
            control_frequency_hz=args.control_freq,
            max_steps=args.max_steps,
            policy_seed=args.seed + args.policy_seed_offset,
            gpus=(0,),
            workers_per_gpu=1,
            action_chunk=args.action_chunk,
            inference_steps=args.num_inference_steps,
            num_steps_wait=args.num_steps_wait,
            binary_gripper=args.binary_gripper,
            gripper_hysteresis_threshold=args.gripper_hysteresis_threshold,
        ),
        backend=BackendSettings(
            name="openpi",
            openpi_source=args.openpi_source.resolve() if args.openpi_source else None,
            openpi_commit=args.openpi_commit,
            openpi_norm_stats=(
                args.openpi_norm_stats.resolve() if args.openpi_norm_stats else None
            ),
        ),
    )


def _result_contract(
    args: argparse.Namespace,
    *,
    task_description: str,
    selected_indices: list[int],
    state_kind: str,
    state_manifest: Path | None,
    state_array_hash: str,
) -> dict[str, Any]:
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "config_name": args.config_name,
        "device": args.device,
        "openpi_source": (
            str(args.openpi_source.expanduser().resolve())
            if args.openpi_source
            else None
        ),
        "openpi_commit": args.openpi_commit,
        "openpi_norm_stats": (
            str(args.openpi_norm_stats.expanduser().resolve())
            if args.openpi_norm_stats
            else None
        ),
        "suite": args.suites[0],
        "task_id": args.task_ids[0],
        "task": task_description,
        "initial_state_indices": selected_indices,
        "initial_state_kind": state_kind,
        "custom_initial_state_manifest": (
            str(state_manifest) if state_manifest else None
        ),
        "state_array_sha256": state_array_hash,
        "trials_per_initial_state": args.trials_per_initial_state,
        "seed": args.seed,
        "policy_seed_offset": args.policy_seed_offset,
        "control_freq": args.control_freq,
        "fps": args.fps,
        "max_steps": args.max_steps,
        "num_steps_wait": args.num_steps_wait,
        "action_chunk": args.action_chunk,
        "num_inference_steps": args.num_inference_steps,
        "compile_mode": args.compile_mode,
        "binary_gripper": args.binary_gripper,
        "gripper_hysteresis_threshold": (
            args.gripper_hysteresis_threshold if args.binary_gripper else None
        ),
        "training_schema_only": args.training_schema_only,
        "save_images": args.save_images,
        "videos_only": args.videos_only,
        "repo_id": args.repo_id,
    }


def _new_results(
    contract: dict[str, Any], *, task_description: str, output: Path
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract": contract,
        **contract,
        "task": task_description,
        "dataset_root": None if contract["videos_only"] else str(output),
        "video_root": str(output / "videos"),
        "episodes": [],
        "overall": aggregate_episode_results([]),
    }


def validate_resume_prefix(
    episodes: list[dict[str, Any]], specs: list[tuple[int, int]]
) -> None:
    actual = [
        (int(item.get("trial_index", -1)), int(item.get("initial_state_index", -1)))
        for item in episodes
    ]
    if actual != specs[: len(actual)]:
        raise ValueError("saved results are not a unique contiguous rollout prefix")


def _open_output(
    args: argparse.Namespace,
    *,
    contract: dict[str, Any],
    task_description: str,
    specs: list[tuple[int, int]],
) -> tuple[Any | None, dict[str, Any]]:
    output = args.output.expanduser().resolve()
    results_path = output / "results.json"
    if args.overwrite and output.exists():
        _safe_remove(output)

    if args.videos_only:
        if output.exists():
            raise FileExistsError(f"output already exists: {output}; pass --overwrite")
        (output / "videos" / "agentview").mkdir(parents=True)
        (output / "videos" / "wrist").mkdir(parents=True)
        results = _new_results(
            contract, task_description=task_description, output=output
        )
        _write_json(results_path, results)
        return None, results

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if args.resume:
        if not results_path.is_file():
            raise FileNotFoundError(f"cannot resume without {results_path}")
        results = json.loads(results_path.read_text(encoding="utf-8"))
        if results.get("contract") != contract:
            raise ValueError("resume arguments differ from the saved worker contract")
        episodes = results.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError("saved results.json has no episodes list")
        validate_resume_prefix(episodes, specs)
        info_path = output / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"cannot resume without {info_path}")
        metadata_episodes = int(
            json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"]
        )
        if metadata_episodes != len(episodes):
            raise RuntimeError(
                "cannot safely resume: LeRobot metadata/results episode counts differ "
                f"({metadata_episodes} != {len(episodes)})"
            )
        _quarantine_uncommitted_files(output, first_episode=metadata_episodes)
        shutil.rmtree(output / "images", ignore_errors=True)
        dataset = LeRobotDataset(repo_id=args.repo_id, root=output)
        if int(dataset.meta.total_episodes) != len(episodes):
            raise RuntimeError("LeRobot loader episode count differs from results.json")
        return dataset, results

    if output.exists():
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite or --resume"
        )
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output,
        robot_type="panda",
        fps=args.fps,
        features=dataset_features(
            save_images=args.save_images,
            training_schema_only=args.training_schema_only,
        ),
        use_videos=not args.save_images,
        image_writer_threads=args.image_writer_threads,
    )
    results = _new_results(contract, task_description=task_description, output=output)
    _write_json(results_path, results)
    return dataset, results


@contextlib.contextmanager
def _video_frame_context(output: Path, episode_index: int, fps: int):
    import imageio.v2 as imageio

    wide = imageio.get_writer(
        output / "videos" / "agentview" / f"episode_{episode_index:06d}.mp4",
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=1,
    )
    wrist = imageio.get_writer(
        output / "videos" / "wrist" / f"episode_{episode_index:06d}.mp4",
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=1,
    )

    def persist(frame: EvaluationFrame) -> None:
        wide.append_data(frame.image)
        wrist.append_data(frame.wrist_image)

    try:
        yield persist
    finally:
        wide.close()
        wrist.close()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Adapt one campaign worker contract to the shared rollout core."""

    output = args.output.expanduser().resolve()
    settings = _settings(args)
    settings.validate()
    runtime = LiberoRuntime(
        settings.task.suite,
        settings.task.task_id,
        settings.controller.source_control_space,
        settings.rollout.control_frequency_hz,
    )
    task_description = runtime.task_description()
    bundle = resolve_evaluation_initial_states(
        runtime=runtime,
        manifest_path=args.custom_initial_state_manifest,
        suite=settings.task.suite,
        task_id=settings.task.task_id,
        task_description=task_description,
        control_frequency=args.control_freq,
    )

    selected_indices = args.initial_state_indices or list(range(args.episodes_per_task))
    invalid = [
        index for index in selected_indices if not 0 <= index < len(bundle.states)
    ]
    if invalid:
        raise ValueError(
            f"invalid initial-state indices {invalid}; available={len(bundle.states)}"
        )
    scene_seeds = (
        {
            index: int(entry["scene_model_seed"])
            for index, entry in bundle.entries.items()
        }
        if bundle.entries is not None
        else None
    )
    specs = build_episode_specs(
        selected_indices,
        trials_per_initial_state=args.trials_per_initial_state,
        state_count=len(bundle.states),
        seed=args.seed,
        policy_seed_offset=args.policy_seed_offset,
        scene_seeds=scene_seeds,
    )
    spec_keys = [(spec.trial_index, spec.initial_state_index) for spec in specs]
    contract = _result_contract(
        args,
        task_description=task_description,
        selected_indices=selected_indices,
        state_kind=bundle.kind,
        state_manifest=bundle.manifest_path,
        state_array_hash=bundle.array_sha256,
    )
    dataset, results = _open_output(
        args,
        contract=contract,
        task_description=task_description,
        specs=spec_keys,
    )
    results_path = output / "results.json"
    completed = len(results["episodes"])
    pending_specs = specs[completed:]
    if not pending_specs:
        if dataset is not None:
            dataset.stop_image_writer()
        return results

    policy = OpenPIBackend(settings, args.device, compile_mode=args.compile_mode).policy
    env = runtime.new_env(args.seed)
    dataset_indices: dict[int, int] = {}

    def frame_context(spec: EpisodeSpec):
        dataset_episode_index = (
            int(dataset.meta.total_episodes)
            if dataset is not None
            else len(results["episodes"])
        )
        dataset_indices[spec.episode_index] = dataset_episode_index
        if dataset is None:
            return _video_frame_context(output, dataset_episode_index, args.fps)

        def persist_frame(frame: EvaluationFrame) -> None:
            value: dict[str, Any] = {
                "image": frame.image,
                "wrist_image": frame.wrist_image,
                "state": frame.state,
                "actions": frame.action,
            }
            if not args.training_schema_only:
                value.update(
                    {
                        "next.reward": np.asarray([frame.reward], dtype=np.float32),
                        "next.done": np.asarray([frame.success], dtype=np.bool_),
                        "next.success": np.asarray([frame.success], dtype=np.bool_),
                        "next.truncated": np.asarray([frame.truncated], dtype=np.bool_),
                    }
                )
            _add_frame(dataset, value, task_description)

        return contextlib.nullcontext(persist_frame)

    def persist_episode(run: EpisodeRun) -> None:
        if dataset is not None:
            dataset.save_episode()
        spec = run.spec
        entry = (
            bundle.entries[spec.initial_state_index]
            if bundle.entries is not None
            else None
        )
        results["episodes"].append(
            {
                "dataset_episode_index": dataset_indices[spec.episode_index],
                "suite": settings.task.suite,
                "task_id": settings.task.task_id,
                "task": task_description,
                **run.as_record(),
                "initial_state_kind": bundle.kind,
                "custom_initial_state_manifest": (
                    str(bundle.manifest_path) if bundle.manifest_path else None
                ),
                "initial_state_sha256": (
                    entry["simulator_state_sha256"] if entry is not None else None
                ),
                "placement_seed": (
                    int(entry["placement_seed"]) if entry is not None else None
                ),
                "checkpoint": str(settings.task.checkpoint),
                "control_freq": args.control_freq,
            }
        )
        results["overall"] = aggregate_episode_results(results["episodes"])
        _write_json(results_path, results)

    try:
        run_evaluation_batch(
            runtime=runtime,
            env=env,
            policy=policy,
            initial_states=bundle.states,
            specs=pending_specs,
            task_description=task_description,
            config=EvaluationConfig(
                max_steps=args.max_steps,
                action_chunk=args.action_chunk,
                num_steps_wait=args.num_steps_wait,
                binary_gripper=args.binary_gripper,
                gripper_hysteresis_threshold=args.gripper_hysteresis_threshold,
            ),
            frame_callback_context=frame_context,
            on_episode_complete=persist_episode,
        )
    finally:
        env.close()
        if dataset is not None:
            dataset.stop_image_writer()
    return results


def worker_main(argv: list[str] | None = None) -> int:
    args = parse_worker_args(argv)
    results = evaluate(args)
    print(json.dumps(results["overall"], sort_keys=True))
    return 0


def parse_generator_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or reuse validated deterministic BDDL-randomized LIBERO states."
        )
    )
    parser.add_argument("--suite", choices=ALL_SUITES, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--validation-hold-steps", type=int, default=5)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--maximum-stabilization-drift", type=float, default=0.05)
    parser.add_argument("--max-attempts", type=int)
    args = parser.parse_args(argv)
    if args.task_id < 0 or args.count <= 0 or args.control_freq <= 0:
        parser.error("task ID must be non-negative; count/frequency must be positive")
    if args.wait_steps < 0 or args.validation_hold_steps < 0:
        parser.error("wait steps must be non-negative")
    if args.maximum_stabilization_drift <= 0:
        parser.error("maximum stabilization drift must be positive")
    args.max_attempts = args.max_attempts or args.count * 20
    if args.max_attempts < args.count:
        parser.error("--max-attempts must be at least --count")
    return args


def generator_main(argv: list[str] | None = None) -> int:
    args = parse_generator_args(argv)
    manifest = generate_randomized_states(
        output=args.output.expanduser().resolve(),
        suite_name=args.suite,
        task_id=args.task_id,
        count=args.count,
        seed_start=args.seed_start,
        wait_steps=args.wait_steps,
        validation_hold_steps=args.validation_hold_steps,
        control_freq=args.control_freq,
        maximum_stabilization_drift=args.maximum_stabilization_drift,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] not in {"worker", "generate"}:
        raise SystemExit("usage: _runtime.py {worker|generate} ...")
    command = values.pop(0)
    return worker_main(values) if command == "worker" else generator_main(values)


if __name__ == "__main__":
    raise SystemExit(main())
