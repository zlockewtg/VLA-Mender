from pathlib import Path

import pytest

from workflow.parameters import ControllerSettings, ExperimentSettings, ResetSettings, TaskSettings


def settings(**kwargs):
    return ExperimentSettings(task=TaskSettings("libero_goal", 0, Path("checkpoint")), **kwargs)


def test_reset_dynamics_requires_osc_target():
    value = settings(reset=ResetSettings(dynamics="quiescent_osc"),
                     controller=ControllerSettings("osc", "joint"))
    with pytest.raises(ValueError, match="target_control_space=osc"):
        value.validate()


def test_fingerprint_is_stable_and_changes_with_stride():
    first = settings().fingerprint()
    second = settings().fingerprint()
    changed = settings(reset=ResetSettings(frame_stride=7)).fingerprint()
    assert first == second
    assert first != changed
