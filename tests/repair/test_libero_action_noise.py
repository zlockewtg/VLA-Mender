from __future__ import annotations

import numpy as np

from workflow.research.libero_backend import NativeLiberoPolicyEnv
from workflow.rollout.action_noise import OscActionNoise


NOISE_CONFIG = {
    "type": "ornstein_uhlenbeck",
    "seed": 42000,
    "rho": 0.85,
    "standard_deviation": [0.006, 0.006, 0.006, 0.003, 0.003, 0.003],
    "maximum_absolute": [0.018, 0.018, 0.018, 0.009, 0.009, 0.009],
}


def test_osc_noise_is_deterministic_six_dimensional_and_never_changes_gripper() -> None:
    first = OscActionNoise(NOISE_CONFIG)
    second = OscActionNoise(NOISE_CONFIG)

    first_samples = np.asarray([first.sample(7) for _ in range(32)])
    second_samples = np.asarray([second.sample(7) for _ in range(32)])

    assert np.array_equal(first_samples, second_samples)
    assert np.all(first_samples[:, :6] != 0.0)
    assert np.array_equal(first_samples[:, 6], np.zeros(32))
    assert np.all(np.abs(first_samples[:, :3]) <= 0.018)
    assert np.all(np.abs(first_samples[:, 3:6]) <= 0.009)


class _FakeInnerEnv:
    action_spec = (-np.ones(7), np.ones(7))


class _FakeEnv:
    def __init__(self) -> None:
        self.env = _FakeInnerEnv()
        self.commands: list[np.ndarray] = []

    def check_success(self) -> bool:
        return False

    def step(self, command: list[float]):
        self.commands.append(np.asarray(command, dtype=np.float64))
        observation = {"robot0_gripper_qpos": np.array([0.04])}
        return observation, 0.0, False, {}


class _FakeRuntime:
    control_space = "osc"

    @staticmethod
    def observation_images(observation):
        del observation
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        return image, image

    @staticmethod
    def public_state(observation):
        del observation
        return np.zeros(8, dtype=np.float32)


def test_dispatch_records_the_exact_noisy_command_sent_to_env() -> None:
    observation = {"robot0_gripper_qpos": np.array([0.04])}
    env = _FakeEnv()
    policy_env = NativeLiberoPolicyEnv(
        env,
        _FakeRuntime(),
        observation,
        10,
        action_noise=NOISE_CONFIG,
    )
    nominal = np.array([0.2, -0.3, 0.4, 0.01, -0.02, 0.03, 1.0])

    policy_env.step_osc_pose(nominal)

    recorded = np.asarray(policy_env.actions[0])
    sampled = np.asarray(policy_env.sampled_action_noises[0])
    stored_nominal = np.asarray(policy_env.nominal_actions[0])
    expected = np.clip(stored_nominal + sampled, -1.0, 1.0)
    assert np.allclose(recorded, expected)
    assert np.allclose(env.commands[0], expected)
    assert np.all(sampled[:6] != 0.0)
    assert sampled[6] == 0.0
    assert recorded[6] == nominal[6]
