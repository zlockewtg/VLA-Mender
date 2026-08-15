"""Shared OpenPI/LIBERO episode evaluator.

This module contains the outcome-changing loop used by both the standalone
LeRobot evaluation scripts and the pre-repair rollout workflow.  Persistence
is deliberately left to callers: the standalone evaluator writes LeRobot
episodes, while the workflow writes its public JSON/video trajectory contract.
"""

from __future__ import annotations

import collections
import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    """Parameters that change one native-done LIBERO evaluation episode."""

    max_steps: int
    action_chunk: int = 5
    num_steps_wait: int = 10
    binary_gripper: bool = False
    gripper_hysteresis_threshold: float = 0.2

    def validate(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.action_chunk <= 0:
            raise ValueError("action_chunk must be positive")
        if self.num_steps_wait < 0:
            raise ValueError("num_steps_wait must be non-negative")
        if not 0.0 <= self.gripper_hysteresis_threshold <= 1.0:
            raise ValueError("gripper_hysteresis_threshold must be in [0, 1]")


@dataclasses.dataclass(frozen=True)
class EvaluationFrame:
    """One pre-action observation and the transition produced by its action."""

    image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    action: np.ndarray
    reward: float
    success: bool
    truncated: bool


@dataclasses.dataclass(frozen=True)
class EpisodeEvaluation:
    """Native LIBERO episode outcome plus the public trajectory."""

    success: bool
    truncated: bool
    num_steps: int
    episode_return: float
    mean_inference_ms: float
    first_success_step: int | None
    policy_seed: int
    frames: tuple[EvaluationFrame, ...]

    @property
    def states(self) -> list[list[float]]:
        return [frame.state.astype(float).tolist() for frame in self.frames]

    @property
    def actions(self) -> list[list[float]]:
        return [frame.action.astype(float).tolist() for frame in self.frames]

    @property
    def rewards(self) -> list[float]:
        return [float(frame.reward) for frame in self.frames]

    @property
    def successes(self) -> list[bool]:
        return [bool(frame.success) for frame in self.frames]


FrameCallback = Callable[[EvaluationFrame], None]


def _seed_policy(seed: int) -> None:
    """Seed torch lazily so importing workflow.rollout stays lightweight."""

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dispatch_action(
    action: np.ndarray,
    *,
    env: Any,
    config: EvaluationConfig,
    last_gripper_action: float,
) -> np.ndarray:
    low, high = env.env.action_spec
    low_array = np.asarray(low, dtype=np.float32)
    high_array = np.asarray(high, dtype=np.float32)
    value = np.asarray(action, dtype=np.float32)
    if (
        value.ndim != 1
        or value.shape != low_array.shape
        or not np.isfinite(value).all()
    ):
        raise ValueError(
            f"policy action must be finite with shape {low_array.shape}, got {value.shape}"
        )
    value = np.clip(value, low_array, high_array).astype(np.float32)
    if config.binary_gripper:
        raw_gripper = float(value[-1])
        if raw_gripper >= config.gripper_hysteresis_threshold:
            value[-1] = 1.0
        elif raw_gripper <= -config.gripper_hysteresis_threshold:
            value[-1] = -1.0
        else:
            value[-1] = last_gripper_action
    return value


def evaluate_episode(
    *,
    runtime: Any,
    env: Any,
    policy: Any,
    initial_state: np.ndarray,
    scene_seed: int,
    policy_seed: int,
    task_description: str,
    config: EvaluationConfig,
    frame_callback: FrameCallback | None = None,
) -> EpisodeEvaluation:
    """Evaluate one initial state using the official native-done protocol.

    The environment is reset and seeded here so every caller gets identical
    semantics.  A frame callback may persist each transition, but the executed
    public trajectory is always returned for workflow diagnosis and replay.
    """

    config.validate()
    _seed_policy(policy_seed)
    env.seed(int(scene_seed))
    env.reset()
    obs = env.set_init_state(np.asarray(initial_state, dtype=np.float64))
    neutral_action = np.asarray(runtime.neutral_action(env), dtype=np.float32)
    for _ in range(config.num_steps_wait):
        obs, _, done, _ = env.step(neutral_action.tolist())
        if done:
            raise RuntimeError("LIBERO episode terminated during stabilization")

    action_plan: collections.deque[np.ndarray] = collections.deque()
    inference_times: list[float] = []
    frames: list[EvaluationFrame] = []
    first_success_step: int | None = None
    last_gripper_action = float(neutral_action[-1])

    for step_index in range(config.max_steps):
        image, wrist_image = runtime.observation_images(obs)
        state = np.asarray(runtime.public_state(obs), dtype=np.float32)
        if not action_plan:
            output = policy.infer(
                {
                    "observation/image": image,
                    "observation/wrist_image": wrist_image,
                    "observation/state": state,
                    "prompt": task_description,
                }
            )
            action_chunk = np.asarray(output["actions"], dtype=np.float32)
            if action_chunk.ndim != 2 or len(action_chunk) < config.action_chunk:
                raise ValueError(
                    f"policy returned {len(action_chunk)} actions, fewer than "
                    f"action_chunk={config.action_chunk}"
                )
            action_plan.extend(action_chunk[: config.action_chunk])
            inference_times.append(
                float(output.get("policy_timing", {}).get("infer_ms", 0.0))
            )

        action = _dispatch_action(
            action_plan.popleft(),
            env=env,
            config=config,
            last_gripper_action=last_gripper_action,
        )
        last_gripper_action = float(action[-1])
        next_obs, reward, done, _ = env.step(action.tolist())
        success = bool(done)
        if success and first_success_step is None:
            first_success_step = step_index + 1
        truncated = not success and step_index == config.max_steps - 1
        frame = EvaluationFrame(
            image=np.asarray(image, dtype=np.uint8),
            wrist_image=np.asarray(wrist_image, dtype=np.uint8),
            state=state,
            action=action,
            reward=float(reward),
            success=success,
            truncated=truncated,
        )
        frames.append(frame)
        if frame_callback is not None:
            frame_callback(frame)
        obs = next_obs
        if success or truncated:
            break

    return EpisodeEvaluation(
        success=first_success_step is not None,
        truncated=first_success_step is None,
        num_steps=len(frames),
        episode_return=float(sum(frame.reward for frame in frames)),
        mean_inference_ms=(float(np.mean(inference_times)) if inference_times else 0.0),
        first_success_step=first_success_step,
        policy_seed=int(policy_seed),
        frames=tuple(frames),
    )


def aggregate_episode_results(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Build deterministic success metrics shared by shard and workflow outputs."""

    def identity(item: dict[str, Any]) -> tuple[str, int, int, int]:
        task = item.get("task")
        task_mapping = task if isinstance(task, dict) else {}
        return (
            str(item.get("suite", task_mapping.get("suite", ""))),
            int(item.get("task_id", task_mapping.get("task_id", 0))),
            int(item["initial_state_index"]),
            int(item.get("trial_index", 0)),
        )

    ordered = sorted(
        episodes,
        key=identity,
    )
    successes = [item for item in ordered if bool(item.get("success"))]
    failed = [item for item in ordered if not bool(item.get("success"))]
    return {
        "episodes": len(ordered),
        "successes": len(successes),
        "failures": len(failed),
        "success_rate": len(successes) / len(ordered) if ordered else 0.0,
        "successful_initial_state_indices": [
            int(item["initial_state_index"]) for item in successes
        ],
        "failed_initial_state_indices": [
            int(item["initial_state_index"]) for item in failed
        ],
    }
