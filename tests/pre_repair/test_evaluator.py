from __future__ import annotations

import contextlib

import numpy as np

from workflow.rollout.evaluator import (
    EvaluationConfig,
    aggregate_episode_results,
    evaluate_episode,
)
from workflow.rollout.runner import build_episode_specs, run_evaluation_batch


class FakeEnv:
    def __init__(self, *, done_on_call: int | None):
        self.env = self
        self.action_spec = (np.full(7, -1.0), np.full(7, 1.0))
        self.done_on_call = done_on_call
        self.calls = 0
        self.actions: list[np.ndarray] = []
        self.seeds: list[int] = []

    def seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def reset(self):
        self.calls = 0
        self.actions = []
        return {"call": self.calls}

    def set_init_state(self, _state):
        return {"call": self.calls}

    def step(self, action):
        self.calls += 1
        self.actions.append(np.asarray(action, dtype=np.float32))
        done = self.done_on_call is not None and self.calls == self.done_on_call
        return {"call": self.calls}, float(self.calls), done, {}


class FakeRuntime:
    @staticmethod
    def neutral_action(_env):
        return np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)

    @staticmethod
    def observation_images(obs):
        value = np.full((2, 2, 3), obs["call"], dtype=np.uint8)
        return value, value.copy()

    @staticmethod
    def public_state(obs):
        return np.full(8, obs["call"], dtype=np.float32)


class FakePolicy:
    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []

    def infer(self, observation):
        self.prompts.append(observation["prompt"])
        chunks = (
            np.asarray(
                [
                    [2.0, 0, 0, 0, 0, 0, 0.5],
                    [0.0, 0, 0, 0, 0, 0, 0.1],
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [
                    [0.0, 0, 0, 0, 0, 0, -0.1],
                    [0.0, 0, 0, 0, 0, 0, -0.5],
                ],
                dtype=np.float32,
            ),
        )
        value = chunks[min(self.calls, len(chunks) - 1)]
        self.calls += 1
        return {"actions": value, "policy_timing": {"infer_ms": 4.0}}


def test_native_done_uses_shared_clip_chunk_and_binary_hysteresis(monkeypatch):
    monkeypatch.setattr("workflow.rollout.evaluator._seed_policy", lambda _seed: None)
    env = FakeEnv(done_on_call=5)  # two stabilization steps, then three policy steps
    policy = FakePolicy()

    outcome = evaluate_episode(
        runtime=FakeRuntime(),
        env=env,
        policy=policy,
        initial_state=np.zeros(3),
        scene_seed=7,
        policy_seed=11,
        task_description="open the middle drawer",
        config=EvaluationConfig(
            max_steps=10,
            action_chunk=2,
            num_steps_wait=2,
            binary_gripper=True,
            gripper_hysteresis_threshold=0.2,
        ),
    )

    assert outcome.success is True
    assert outcome.truncated is False
    assert outcome.first_success_step == 3
    assert outcome.num_steps == 3
    assert outcome.mean_inference_ms == 4.0
    assert env.seeds[-1] == 7
    assert policy.calls == 2
    assert policy.prompts == ["open the middle drawer"] * 2
    np.testing.assert_allclose(outcome.frames[0].action[0], 1.0)  # controller clip
    assert [float(frame.action[-1]) for frame in outcome.frames] == [1.0, 1.0, 1.0]
    assert outcome.successes == [False, False, True]


def test_timeout_is_truncated_on_exact_policy_budget(monkeypatch):
    monkeypatch.setattr("workflow.rollout.evaluator._seed_policy", lambda _seed: None)
    outcome = evaluate_episode(
        runtime=FakeRuntime(),
        env=FakeEnv(done_on_call=None),
        policy=FakePolicy(),
        initial_state=np.zeros(3),
        scene_seed=3,
        policy_seed=5,
        task_description="task",
        config=EvaluationConfig(max_steps=3, action_chunk=2, num_steps_wait=1),
    )

    assert outcome.success is False
    assert outcome.truncated is True
    assert outcome.num_steps == 3
    assert outcome.frames[-1].truncated is True
    assert outcome.first_success_step is None


def test_aggregate_accepts_standalone_task_text_and_workflow_task_mapping():
    metrics = aggregate_episode_results(
        [
            {
                "suite": "libero_goal",
                "task_id": 0,
                "task": "open the middle drawer",
                "initial_state_index": 1,
                "success": False,
            },
            {
                "task": {"suite": "libero_goal", "task_id": 0},
                "initial_state_index": 0,
                "success": True,
            },
        ]
    )

    assert metrics == {
        "episodes": 2,
        "successes": 1,
        "success_rate": 0.5,
        "successful_initial_state_indices": [0],
        "failed_initial_state_indices": [1],
    }


def test_batch_specs_and_execution_are_worker_partition_independent(monkeypatch):
    monkeypatch.setattr("workflow.rollout.evaluator._seed_policy", lambda _seed: None)
    specs = build_episode_specs(
        [1, 3],
        trials_per_initial_state=2,
        state_count=5,
        seed=7,
        policy_seed_offset=100,
        scene_seeds={1: 201, 3: 203},
    )
    assert [
        (
            spec.episode_index,
            spec.trial_index,
            spec.initial_state_index,
            spec.scene_seed,
            spec.policy_seed,
        )
        for spec in specs
    ] == [
        (0, 0, 1, 201, 108),
        (1, 0, 3, 203, 110),
        (2, 1, 1, 201, 113),
        (3, 1, 3, 203, 115),
    ]

    opened: list[int] = []
    persisted_frames: list[tuple[int, bool]] = []
    completed: list[int] = []

    @contextlib.contextmanager
    def frame_context(spec):
        opened.append(spec.episode_index)

        def persist(frame):
            persisted_frames.append((spec.episode_index, frame.success))

        yield persist

    runs = run_evaluation_batch(
        runtime=FakeRuntime(),
        env=FakeEnv(done_on_call=1),
        policy=FakePolicy(),
        initial_states=np.zeros((5, 3)),
        specs=specs,
        task_description="task",
        config=EvaluationConfig(max_steps=2, action_chunk=2, num_steps_wait=0),
        frame_callback_context=frame_context,
        on_episode_complete=lambda run: completed.append(run.spec.episode_index),
    )

    assert opened == [0, 1, 2, 3]
    assert completed == [0, 1, 2, 3]
    assert persisted_frames == [(0, True), (1, True), (2, True), (3, True)]
    assert [run.as_record()["policy_seed"] for run in runs] == [108, 110, 113, 115]
