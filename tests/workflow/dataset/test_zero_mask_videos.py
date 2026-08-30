from __future__ import annotations

import numpy as np

from workflow.dataset.zero_mask_videos import ZeroArmMaskSettings, compute_base_zero_arm_mask


def test_base_zero_arm_mask_preserves_arm_and_gripper_motion() -> None:
    actions = np.zeros((5, 7), dtype=np.float32)
    states = np.zeros((5, 8), dtype=np.float32)
    actions[1, 0] = 0.041  # 2.05 mm after the 0.05 m action scale.
    actions[2, 6] = 1.0  # Command transition at frame 2.
    actions[3, 6] = 1.0
    states[3, 6] = 1.0e-4  # Physical change preserves frame 2.

    timeline = compute_base_zero_arm_mask(actions, states, ZeroArmMaskSettings())

    assert timeline.arm_motion.tolist() == [False, True, False, False, False]
    assert timeline.gripper_command_change.tolist() == [False, False, True, False, True]
    assert timeline.gripper_state_change.tolist() == [False, False, True, True, False]
    assert timeline.zero_masked.tolist() == [True, False, False, False, False]


def test_chunk_boundary_exceptions_are_not_applied_to_stable_timeline() -> None:
    actions = np.zeros((3, 7), dtype=np.float32)
    states = np.zeros((3, 8), dtype=np.float32)

    timeline = compute_base_zero_arm_mask(actions, states, ZeroArmMaskSettings())

    assert timeline.zero_masked.tolist() == [True, True, True]


def test_state_delta_mask_uses_next_state_motion_not_command() -> None:
    actions = np.ones((4, 7), dtype=np.float32)
    actions[:, 6] = -1.0
    states = np.zeros((4, 8), dtype=np.float32)
    states[2:, 0] = 0.003

    timeline = compute_base_zero_arm_mask(
        actions,
        states,
        ZeroArmMaskSettings(mode="state_delta"),
    )

    assert timeline.arm_motion.tolist() == [False, True, False, False]
    assert timeline.zero_masked.tolist() == [True, False, True, True]
