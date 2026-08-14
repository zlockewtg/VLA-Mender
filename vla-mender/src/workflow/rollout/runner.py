"""Task-neutral batch execution built on the shared episode evaluator."""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

import numpy as np

from .evaluator import (
    EpisodeEvaluation,
    EvaluationConfig,
    FrameCallback,
    evaluate_episode,
)


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    """Stable identity and seeds for one state/trial evaluation."""

    episode_index: int
    trial_index: int
    initial_state_index: int
    scene_seed: int
    policy_seed: int


@dataclasses.dataclass(frozen=True)
class EpisodeRun:
    """One completed batch item and its native evaluator outcome."""

    spec: EpisodeSpec
    outcome: EpisodeEvaluation

    def as_record(self, *, include_trajectory: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "episode_index": self.spec.episode_index,
            "trial_index": self.spec.trial_index,
            "initial_state_index": self.spec.initial_state_index,
            "scene_model_seed": self.spec.scene_seed,
            "policy_seed": self.outcome.policy_seed,
            "success": self.outcome.success,
            "truncated": self.outcome.truncated,
            "num_steps": self.outcome.num_steps,
            "episode_return": self.outcome.episode_return,
            "mean_inference_ms": self.outcome.mean_inference_ms,
            "first_success_step": self.outcome.first_success_step,
        }
        if include_trajectory:
            record.update(
                {
                    "states": self.outcome.states,
                    "actions": self.outcome.actions,
                    "successes": self.outcome.successes,
                    "rewards": self.outcome.rewards,
                }
            )
        return record


def policy_seed(
    *,
    seed: int,
    offset: int,
    trial_index: int,
    state_count: int,
    initial_state_index: int,
) -> int:
    """Return a seed independent of worker count and shard boundaries."""

    if state_count <= 0:
        raise ValueError("state_count must be positive")
    if trial_index < 0 or not 0 <= initial_state_index < state_count:
        raise ValueError("trial/state indices are outside the evaluation contract")
    return seed + offset + trial_index * state_count + initial_state_index


def build_episode_specs(
    initial_state_indices: Sequence[int],
    *,
    trials_per_initial_state: int,
    state_count: int,
    seed: int,
    policy_seed_offset: int = 0,
    scene_seeds: Mapping[int, int] | None = None,
) -> list[EpisodeSpec]:
    """Build deterministic trial-major specs for a task or worker shard."""

    indices = [int(index) for index in initial_state_indices]
    if trials_per_initial_state <= 0:
        raise ValueError("trials_per_initial_state must be positive")
    if len(indices) != len(set(indices)):
        raise ValueError("initial_state_indices must be unique")
    invalid = [index for index in indices if not 0 <= index < state_count]
    if invalid:
        raise ValueError(
            f"initial-state indices {invalid} are outside state_count={state_count}"
        )
    resolved_scene_seeds = scene_seeds or {}
    specs: list[EpisodeSpec] = []
    for trial_index in range(trials_per_initial_state):
        for initial_state_index in indices:
            specs.append(
                EpisodeSpec(
                    episode_index=len(specs),
                    trial_index=trial_index,
                    initial_state_index=initial_state_index,
                    scene_seed=int(resolved_scene_seeds.get(initial_state_index, seed)),
                    policy_seed=policy_seed(
                        seed=seed,
                        offset=policy_seed_offset,
                        trial_index=trial_index,
                        state_count=state_count,
                        initial_state_index=initial_state_index,
                    ),
                )
            )
    return specs


FrameCallbackContextFactory = Callable[
    [EpisodeSpec], AbstractContextManager[FrameCallback | None]
]
EpisodeCompleteCallback = Callable[[EpisodeRun], None]


def run_evaluation_batch(
    *,
    runtime: Any,
    env: Any,
    policy: Any,
    initial_states: np.ndarray | Mapping[int, np.ndarray],
    specs: Sequence[EpisodeSpec],
    task_description: str,
    config: EvaluationConfig,
    frame_callback_context: FrameCallbackContextFactory | None = None,
    on_episode_complete: EpisodeCompleteCallback | None = None,
) -> list[EpisodeRun]:
    """Evaluate a deterministic sequence while callers own process and artifacts.

    The caller owns ``env`` so a CLI worker and an in-process workflow can choose
    their lifecycle without duplicating episode semantics. Artifact adapters use
    the context factory to safely open and close per-episode writers.
    """

    if isinstance(initial_states, Mapping):
        states_by_index = {
            int(index): np.asarray(state, dtype=np.float64)
            for index, state in initial_states.items()
        }
        if any(
            state.ndim != 1 or not np.isfinite(state).all()
            for state in states_by_index.values()
        ):
            raise ValueError("mapped initial states must be finite vectors")
    else:
        states = np.asarray(initial_states, dtype=np.float64)
        if states.ndim != 2 or not np.isfinite(states).all():
            raise ValueError("initial_states must be a finite rank-2 array")
        states_by_index = {index: state for index, state in enumerate(states)}
    runs: list[EpisodeRun] = []
    for spec in specs:
        if spec.initial_state_index not in states_by_index:
            raise ValueError(
                f"initial-state index {spec.initial_state_index} is unavailable"
            )
        callback_manager = (
            frame_callback_context(spec)
            if frame_callback_context is not None
            else contextlib.nullcontext(None)
        )
        with callback_manager as frame_callback:
            outcome = evaluate_episode(
                runtime=runtime,
                env=env,
                policy=policy,
                initial_state=states_by_index[spec.initial_state_index],
                scene_seed=spec.scene_seed,
                policy_seed=spec.policy_seed,
                task_description=task_description,
                config=config,
                frame_callback=frame_callback,
            )
        run = EpisodeRun(spec=spec, outcome=outcome)
        runs.append(run)
        if on_episode_complete is not None:
            on_episode_complete(run)
    return runs
