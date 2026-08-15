"""Pre-repair adapter around the shared LIBERO evaluation core."""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np

from ..libero_runtime import LiberoRuntime
from ..openpi_backend import OpenPIBackend, openpi_runtime_preflight
from ..parameters import ExperimentSettings
from ..trajectory_protocol import protocol_metadata
from .evaluator import EvaluationConfig, aggregate_episode_results
from .runner import EpisodeRun, EpisodeSpec, build_episode_specs, run_evaluation_batch
from .state_provider import array_hash, generate_initial_states


def _policy_from_openpi(settings: ExperimentSettings, device: str) -> Any:
    """Compatibility wrapper for callers that still use the old helper."""

    return OpenPIBackend(settings, device).policy


def _episode_record(
    *,
    settings: ExperimentSettings,
    run: EpisodeRun,
    task_description: str,
    device: str,
) -> dict[str, Any]:
    shared_record = run.as_record(include_trajectory=True)
    # The pre-repair public schema predates multi-trial standalone evaluation.
    # Keep it field-compatible while the shared core still tracks trials.
    shared_record.pop("trial_index")
    return {
        **shared_record,
        "control_frequency_hz": settings.rollout.control_frequency_hz,
        "num_steps_wait": settings.rollout.num_steps_wait,
        "binary_gripper": settings.rollout.binary_gripper,
        "gripper_hysteresis_threshold": (
            settings.rollout.gripper_hysteresis_threshold
            if settings.rollout.binary_gripper
            else None
        ),
        "source_control_space": settings.controller.source_control_space,
        "device": device,
        "task": {
            "suite": settings.task.suite,
            "task_id": settings.task.task_id,
            "description": task_description,
        },
        "settings_fingerprint": settings.fingerprint(),
        "trajectory_protocol": protocol_metadata(),
    }


def _write_episode_videos(
    output: Path, run: EpisodeRun, *, control_frequency_hz: int
) -> None:
    import imageio.v2 as imageio

    video_dir = output / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    for name, frames in (
        ("wide", [frame.image for frame in run.outcome.frames]),
        ("wrist", [frame.wrist_image for frame in run.outcome.frames]),
    ):
        imageio.mimwrite(
            video_dir / f"episode_{run.spec.episode_index:06d}_{name}.mp4",
            frames,
            fps=control_frequency_hz,
            codec="libx264",
            quality=7,
            macro_block_size=1,
        )


def _run_worker(
    settings: ExperimentSettings,
    output: Path,
    gpu: int,
    items: list[tuple[EpisodeSpec, np.ndarray]],
) -> list[dict[str, Any]]:
    """Load one policy and execute its deterministic shared-core shard."""

    if not items:
        return []
    device = f"cuda:{gpu}"
    policy = OpenPIBackend(settings, device).policy
    runtime = LiberoRuntime(
        settings.task.suite,
        settings.task.task_id,
        settings.controller.source_control_space,
        settings.rollout.control_frequency_hz,
        libero_root=settings.backend.libero_root,
    )
    task_description = settings.task.task_description or runtime.task_description()
    states = {spec.initial_state_index: state for spec, state in items}
    specs = [item[0] for item in items]
    summaries: list[dict[str, Any]] = []

    def persist_episode(run: EpisodeRun) -> None:
        _write_episode_videos(
            output,
            run,
            control_frequency_hz=settings.rollout.control_frequency_hz,
        )
        result = _episode_record(
            settings=settings,
            run=run,
            task_description=task_description,
            device=device,
        )
        episode_path = (
            output / "episodes" / f"episode_{run.spec.episode_index:06d}.json"
        )
        LiberoRuntime.write_json(episode_path, result)
        summaries.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"states", "actions", "successes", "rewards"}
            }
        )

    env = runtime.new_env(specs[0].scene_seed)
    try:
        run_evaluation_batch(
            runtime=runtime,
            env=env,
            policy=policy,
            initial_states=states,
            specs=specs,
            task_description=task_description,
            config=EvaluationConfig(
                max_steps=settings.rollout.max_steps,
                action_chunk=settings.rollout.action_chunk,
                num_steps_wait=settings.rollout.num_steps_wait,
                binary_gripper=settings.rollout.binary_gripper,
                gripper_hysteresis_threshold=(
                    settings.rollout.gripper_hysteresis_threshold
                ),
            ),
            on_episode_complete=persist_episode,
        )
    finally:
        env.close()
    return summaries


def run_rollout(settings: ExperimentSettings, output_dir: str | Path) -> dict[str, Any]:
    """Run all initial states and write the workflow JSON/video rollout bank."""

    settings.validate()
    backend_manifest = openpi_runtime_preflight(settings)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    states, provenance = generate_initial_states(settings)
    manifest = {
        "schema_version": 2,
        "settings_fingerprint": settings.fingerprint(),
        "trajectory_protocol": protocol_metadata(),
        "backend": backend_manifest,
        "suite": settings.task.suite,
        "task_id": settings.task.task_id,
        "count": len(states),
        "state_shape": list(states.shape),
        "state_sha256": array_hash(states),
        "entries": provenance,
    }
    np.save(output / "initial_states.npy", states)
    (output / "initial_state_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    scene_seeds = {
        index: int(
            provenance[index].get("scene_model_seed", settings.rollout.policy_seed)
        )
        for index in range(len(states))
        if index < len(provenance)
    }
    specs = build_episode_specs(
        range(len(states)),
        trials_per_initial_state=1,
        state_count=len(states),
        seed=settings.rollout.policy_seed,
        scene_seeds=scene_seeds,
    )
    worker_gpus = [
        gpu
        for gpu in settings.rollout.gpus
        for _ in range(settings.rollout.workers_per_gpu)
    ]
    assignments = [specs[slot :: len(worker_gpus)] for slot in range(len(worker_gpus))]
    worker_items = [
        [(spec, states[spec.initial_state_index]) for spec in assignment]
        for assignment in assignments
    ]
    results: list[dict[str, Any]] = []
    if len(worker_gpus) == 1:
        results = _run_worker(settings, output, worker_gpus[0], worker_items[0])
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(worker_gpus), mp_context=context
        ) as pool:
            futures = [
                pool.submit(_run_worker, settings, output, gpu, items)
                for gpu, items in zip(worker_gpus, worker_items, strict=True)
                if items
            ]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())

    results.sort(key=lambda item: int(item["episode_index"]))
    metrics = aggregate_episode_results(results)
    task_description = (
        results[0]["task"]["description"] if results else settings.task.task_description
    )
    summary = {
        "schema_version": 2,
        "settings_fingerprint": settings.fingerprint(),
        "trajectory_protocol": protocol_metadata(),
        "backend": backend_manifest,
        "task": {
            "suite": settings.task.suite,
            "task_id": settings.task.task_id,
            "description": task_description,
        },
        "overall": metrics,
        "episodes": results,
        "initial_state_manifest": str((output / "initial_state_manifest.json").name),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    successful = [item for item in results if bool(item.get("success"))]
    failed = [item for item in results if not bool(item.get("success"))]
    LiberoRuntime.write_json(
        output / "successful_episodes.json",
        {"schema_version": 1, "count": len(successful), "episodes": successful},
    )
    LiberoRuntime.write_json(
        output / "failed_episodes.json",
        {"schema_version": 1, "count": len(failed), "episodes": failed},
    )
    return summary
