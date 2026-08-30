from pathlib import Path

import pytest

from workflow.parameters import (
    ControllerSettings,
    ExperimentSettings,
    ResetSettings,
    TaskSettings,
    load_experiment_plan,
    load_settings,
)


def settings(**kwargs):
    return ExperimentSettings(task=TaskSettings("libero_goal", 0, Path("checkpoint")), **kwargs)


def test_reset_dynamics_requires_osc_target():
    value = settings(reset=ResetSettings(dynamics="quiescent_osc"),
                     controller=ControllerSettings("osc", "joint"))
    with pytest.raises(ValueError, match="target_control_space=osc"):
        value.validate()


def test_fingerprint_is_stable_and_changes_with_reset_dynamics():
    first = settings().fingerprint()
    second = settings().fingerprint()
    changed = settings(reset=ResetSettings(dynamics="quiescent_osc")).fingerprint()
    assert first == second
    assert first != changed


def test_v5_prevention_steps_are_part_of_experiment_identity():
    first = settings(
        reset=ResetSettings(
            candidate_selection="pre_window_and_endpoints", prevention_steps=10
        )
    ).fingerprint()
    changed = settings(
        reset=ResetSettings(
            candidate_selection="pre_window_and_endpoints", prevention_steps=12
        )
    ).fingerprint()
    assert first != changed


def test_v4_window_endpoint_fingerprint_remains_backward_compatible():
    value = settings(reset=ResetSettings(candidate_selection="window_endpoints"))
    assert "prevention_steps" not in value.as_dict()["reset"]


def test_window_start_only_omits_irrelevant_prevention_steps_from_identity():
    first = settings(
        reset=ResetSettings(
            candidate_selection="window_start_only", prevention_steps=10
        )
    )
    second = settings(
        reset=ResetSettings(
            candidate_selection="window_start_only", prevention_steps=20
        )
    )
    assert first.fingerprint() == second.fingerprint()
    assert "prevention_steps" not in first.as_dict()["reset"]


def test_pre_causal_only_omits_irrelevant_prevention_steps_from_identity():
    first = settings(
        reset=ResetSettings(candidate_selection="pre_causal_only", prevention_steps=10)
    )
    second = settings(
        reset=ResetSettings(candidate_selection="pre_causal_only", prevention_steps=20)
    )
    assert first.fingerprint() == second.fingerprint()
    assert "prevention_steps" not in first.as_dict()["reset"]


def test_failed_stage_entry_only_omits_irrelevant_prevention_steps():
    value = settings(
        reset=ResetSettings(candidate_selection="failed_stage_entry_only")
    )
    assert "prevention_steps" not in value.as_dict()["reset"]


def test_per_episode_stage_entry_only_omits_irrelevant_prevention_steps():
    value = settings(
        reset=ResetSettings(candidate_selection="per_episode_stage_entry_only")
    )
    assert "prevention_steps" not in value.as_dict()["reset"]


def test_legacy_stride_reset_fields_are_rejected(tmp_path):
    source = tmp_path / "experiment.yaml"
    source.write_text(
        """\
task: {checkpoint: ./checkpoint}
reset: {frames_per_failure: 3, frame_stride: 5}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no longer supported"):
        load_settings(source)


def test_campaign_loads_ordered_tasks_with_shared_defaults_and_overrides(tmp_path):
    source = tmp_path / "campaign.yaml"
    source.write_text(
        """\
task:
  checkpoint: ./shared-checkpoint
  policy_config: pi0_libero
tasks:
  - suite: libero_object
    task_id: 0
  - key: spatial-last
    suite: libero_spatial
    task_id: 9
    checkpoint: ./other-checkpoint
    initial_states: {count: 3}
    rollout: {max_steps: 280}
initial_states: {provider: official, count: 5}
rollout: {gpus: [0, 1], max_steps: 300}
""",
        encoding="utf-8",
    )

    plan = load_experiment_plan(source)

    assert plan.is_campaign is True
    assert [task.key for task in plan.tasks] == [
        "libero_object-task000",
        "spatial-last",
    ]
    assert [(task.settings.task.suite, task.settings.task.task_id) for task in plan.tasks] == [
        ("libero_object", 0),
        ("libero_spatial", 9),
    ]
    assert plan.tasks[0].settings.task.checkpoint == (tmp_path / "shared-checkpoint").resolve()
    assert plan.tasks[1].settings.task.checkpoint == (tmp_path / "other-checkpoint").resolve()
    assert [task.settings.initial_states.count for task in plan.tasks] == [5, 3]
    assert [task.settings.rollout.max_steps for task in plan.tasks] == [300, 280]
    assert plan.tasks[0].settings.fingerprint() != plan.tasks[1].settings.fingerprint()
    with pytest.raises(ValueError, match="campaign settings contain a tasks list"):
        load_settings(source)


def test_campaign_rejects_duplicate_or_unsafe_task_keys(tmp_path):
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        """\
task: {checkpoint: ./checkpoint}
tasks:
  - {key: same, suite: libero_goal, task_id: 0}
  - {key: same, suite: libero_spatial, task_id: 1}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="task keys must be unique"):
        load_experiment_plan(duplicate)

    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text(
        """\
task: {checkpoint: ./checkpoint}
tasks:
  - {key: ../escape, suite: libero_goal, task_id: 0}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="task key must start"):
        load_experiment_plan(unsafe)
