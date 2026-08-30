from __future__ import annotations

import numpy as np

from knowledge.api.franka.libero_osc_reduced_skill_library import (
    _gripper_action_from_observed_width,
    _interpolate_pose_minimum_jerk,
    _minimum_jerk,
    _slew_limit_arm_command,
)
from knowledge.api.franka.libero_reduced_skill_library import (
    _classify_grasp_resume_phase,
)


def test_minimum_jerk_has_smooth_zero_velocity_endpoints() -> None:
    samples = np.asarray([_minimum_jerk(value) for value in np.linspace(0.0, 1.0, 9)])

    assert samples[0] == 0.0
    assert samples[-1] == 1.0
    assert np.all(np.diff(samples) > 0.0)
    assert np.diff(samples)[0] < np.diff(samples)[1]
    assert np.diff(samples)[-1] < np.diff(samples)[-2]
    assert np.allclose(np.diff(samples), np.diff(samples)[::-1])


def test_minimum_jerk_pose_interpolation_reaches_both_endpoints() -> None:
    start_position = np.array([0.1, -0.2, 0.3])
    target_position = np.array([0.5, 0.2, 0.7])
    start_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    target_quaternion = np.array([0.0, 0.0, 0.0, 1.0])

    first_position, first_quaternion = _interpolate_pose_minimum_jerk(
        start_position,
        start_quaternion,
        target_position,
        target_quaternion,
        0.0,
    )
    final_position, final_quaternion = _interpolate_pose_minimum_jerk(
        start_position,
        start_quaternion,
        target_position,
        target_quaternion,
        1.0,
    )

    assert np.allclose(first_position, start_position)
    assert np.allclose(first_quaternion, start_quaternion)
    assert np.allclose(final_position, target_position)
    assert np.isclose(abs(np.dot(final_quaternion, target_quaternion)), 1.0)


def test_arm_command_slew_limit_prevents_residual_to_unit_jump() -> None:
    previous = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    command = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    limited = _slew_limit_arm_command(command, previous, max_delta_norm=0.2)

    assert np.linalg.norm(limited[:6] - previous) <= 0.2 + 1.0e-12
    assert limited[-1] == command[-1]


def test_grasp_resume_at_contact_never_returns_to_pregrasp() -> None:
    audit = _classify_grasp_resume_phase(
        np.array([0.20, -0.10, 0.064]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.20, -0.10, 0.150]),
        np.array([0.20, -0.10, 0.067]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.15,
        contact_ready_tolerance_m=0.006,
    )

    assert audit["resume_phase"] == "contact_ready"
    assert audit["phase_regression_avoided"] is True


def test_grasp_resume_inside_descent_corridor_continues_forward() -> None:
    audit = _classify_grasp_resume_phase(
        np.array([0.202, -0.098, 0.105]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.20, -0.10, 0.150]),
        np.array([0.20, -0.10, 0.067]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.15,
        contact_ready_tolerance_m=0.006,
    )

    assert audit["resume_phase"] == "descent_corridor"
    assert 0.0 < audit["descent_segment_progress"] < 1.0


def test_grasp_resume_off_corridor_requires_pregrasp() -> None:
    audit = _classify_grasp_resume_phase(
        np.array([0.29, -0.10, 0.105]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.20, -0.10, 0.150]),
        np.array([0.20, -0.10, 0.067]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.15,
        contact_ready_tolerance_m=0.006,
    )

    assert audit["resume_phase"] == "pregrasp_required"
    assert audit["phase_regression_avoided"] is False


def test_observed_intermediate_gripper_restores_close_hold_after_reset() -> None:
    assert _gripper_action_from_observed_width(0.51) == 1.0
    assert _gripper_action_from_observed_width(0.76) == 1.0
    assert _gripper_action_from_observed_width(0.10) == 1.0
    assert _gripper_action_from_observed_width(0.90) == -1.0
