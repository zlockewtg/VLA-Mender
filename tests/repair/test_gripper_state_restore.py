from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from workflow.failure_diagnosis.failure_diagnosis import (
    _restore_gripper as restore_handoff_gripper,
)
from workflow.research.libero_backend import (
    _restore_gripper as restore_repair_gripper,
)


def _fake_env(initial_state: np.ndarray) -> SimpleNamespace:
    gripper = SimpleNamespace(current_action=initial_state.copy())
    robot = SimpleNamespace(gripper=gripper)
    return SimpleNamespace(env=SimpleNamespace(robots=[robot]))


@pytest.mark.parametrize(
    "restore",
    [restore_handoff_gripper, restore_repair_gripper],
)
@pytest.mark.parametrize(
    "saved_state",
    [np.array([1.0, -1.0]), np.array([-1.0, 1.0])],
)
def test_restore_gripper_preserves_full_panda_actuator_state(
    restore, saved_state: np.ndarray
) -> None:
    env = _fake_env(np.zeros(1))

    restore(env, saved_state)

    restored = env.env.robots[0].gripper.current_action
    assert restored.shape == (2,)
    assert np.array_equal(restored, saved_state)
    assert not np.shares_memory(restored, saved_state)


@pytest.mark.parametrize(
    "restore",
    [restore_handoff_gripper, restore_repair_gripper],
)
def test_restore_gripper_keeps_legacy_scalar_state_shape(restore) -> None:
    env = _fake_env(np.zeros(1))

    restore(env, np.array([-1.0]))

    restored = env.env.robots[0].gripper.current_action
    assert restored.shape == (1,)
    assert np.array_equal(restored, np.array([-1.0]))
