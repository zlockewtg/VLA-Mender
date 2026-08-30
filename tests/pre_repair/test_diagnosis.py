import json

import pytest

from workflow.failure_diagnosis import select_reset_candidates, validate_diagnosis
from workflow.failure_diagnosis.failure_diagnosis import (
    _validate_materialization_rollout_contract,
)
from workflow.parameters import ExperimentSettings, ResetSettings, TaskSettings


def test_candidates_are_exact_failure_window_endpoints(tmp_path):
    settings = ExperimentSettings(
        task=TaskSettings("libero_goal", 0, tmp_path / "ckpt"),
        reset=ResetSettings(candidate_selection="window_endpoints"),
    )
    diagnosis = {"episodes": [{"episode_index": 3, "failure_phase": "transport",
                                "recoverable_window_start_frame_index": 5,
                                "recoverable_window_stop_frame_index": 18}]}
    value = select_reset_candidates(settings, diagnosis)
    assert [item["requested_frame_index"] for item in value["candidates"]] == [5, 18]
    assert value["selection"] == "failure window endpoints"
    assert value["intervention_points"] == ["window_start", "window_stop"]
    assert value["frames_per_failure"] == 2


def test_candidates_can_select_only_failure_window_start(tmp_path):
    settings = ExperimentSettings(
        task=TaskSettings("libero_goal", 0, tmp_path / "ckpt"),
        reset=ResetSettings(candidate_selection="window_start_only"),
    )
    diagnosis = {
        "episodes": [
            {
                "episode_index": 3,
                "failure_phase": "transport",
                "recoverable_window_start_frame_index": 15,
                "recoverable_window_stop_frame_index": 28,
            }
        ]
    }

    value = select_reset_candidates(settings, diagnosis)

    assert [item["requested_frame_index"] for item in value["candidates"]] == [15]
    assert [item["intervention_point"] for item in value["candidates"]] == [
        "window_start"
    ]
    assert value["selection"] == "failure window start only"
    assert value["intervention_points"] == ["window_start"]
    assert value["frames_per_failure"] == 1


def test_candidates_can_select_success_aligned_pre_causal_state(tmp_path):
    settings = ExperimentSettings(
        task=TaskSettings("libero_goal", 0, tmp_path / "ckpt"),
        reset=ResetSettings(candidate_selection="pre_causal_only"),
    )
    diagnosis = {
        "successful_reference_episodes": [
            {"episode_index": 1, "reason": "successful transport reference"}
        ],
        "episodes": [
            {
                "episode_index": 3,
                "failure_phase": "transport",
                "first_causal_frame_index": 28,
                "recoverable_window_start_frame_index": 15,
                "recoverable_window_stop_frame_index": 28,
                "successful_reference_episode_indices": [1],
                "successful_reference_comparison": (
                    "frame 27 matches retained transport in episode 1; "
                    "frame 28 is the first visible separation"
                ),
            }
        ],
    }

    value = select_reset_candidates(settings, diagnosis)

    assert [item["requested_frame_index"] for item in value["candidates"]] == [27]
    assert [item["intervention_point"] for item in value["candidates"]] == [
        "pre_causal"
    ]
    assert value["selection"] == (
        "successful-reference-aligned pre-causal state only"
    )
    assert value["intervention_points"] == ["pre_causal"]
    assert value["frames_per_failure"] == 1


def test_candidates_can_select_failed_behavior_stage_entry(tmp_path):
    settings = ExperimentSettings(
        task=TaskSettings("libero_goal", 0, tmp_path / "ckpt"),
        reset=ResetSettings(candidate_selection="failed_stage_entry_only"),
    )
    diagnosis = {
        "episodes": [
            {
                "episode_index": 3,
                "failure_phase": "transport",
                "first_causal_frame_index": 28,
                "recoverable_window_start_frame_index": 15,
                "recoverable_window_stop_frame_index": 28,
                "intervention_stage": "transport",
                "intervention_stage_start_frame_index": 12,
            }
        ]
    }

    value = select_reset_candidates(settings, diagnosis)

    assert [item["requested_frame_index"] for item in value["candidates"]] == [12]
    assert [item["intervention_point"] for item in value["candidates"]] == [
        "stage_entry"
    ]
    assert value["candidates"][0]["intervention_stage"] == "transport"
    assert value["selection"] == "failed behavior stage entry only"
    assert value["intervention_points"] == ["stage_entry"]
    assert value["frames_per_failure"] == 1


def test_candidates_require_per_episode_stage_boundary_evidence(tmp_path):
    settings = ExperimentSettings(
        task=TaskSettings("libero_goal", 0, tmp_path / "ckpt"),
        reset=ResetSettings(candidate_selection="per_episode_stage_entry_only"),
    )
    diagnosis = {
        "observable_stage_graph": [
            {
                "stage_name": "engagement",
                "observable_entry_condition": "sustained task-directed descent",
                "observable_exit_condition": "object is retained",
                "required_prerequisites": ["alignment"],
                "relevant_entities": ["target object", "gripper"],
                "supporting_successful_episode_indices": [1],
            }
        ],
        "successful_reference_episodes": [
            {"episode_index": 1, "reason": "successful engagement reference"}
        ],
        "episodes": [
            {
                "episode_index": 3,
                "failure_phase": "pick",
                "first_causal_frame_index": 54,
                "recoverable_window_start_frame_index": 34,
                "recoverable_window_stop_frame_index": 54,
                "intervention_stage": "pick",
                "intervention_stage_start_frame_index": 34,
                "stage_entry_evidence": {
                    "preceding_stage": "alignment",
                    "inspected_frame_indices": list(range(31, 38)),
                    "camera_evidence": {
                        "wide": "stage_boundary_evidence/episode_000003_wide.png",
                        "wrist": "stage_boundary_evidence/episode_000003_wrist.png",
                    },
                    "contact_sheet": "stage_boundary_evidence/episode_000003.png",
                    "observable_transition": (
                        "horizontal approach gives way to sustained vertical descent"
                    ),
                    "state_action_transition": "z motion becomes dominant at frame 34",
                    "persistence_evidence": "descent persists through frames 35-37",
                    "successful_reference_episode_indices": [1],
                    "why_previous_frame_is_too_early": "frame 33 is still alignment",
                    "why_later_frame_is_too_late": "descent is already active",
                },
            }
        ]
    }

    value = select_reset_candidates(settings, diagnosis)

    assert value["selection"] == (
        "per-episode observed behavior stage entry only"
    )
    assert value["candidates"][0]["requested_frame_index"] == 34
    assert value["candidates"][0]["stage_entry_evidence"] == (
        diagnosis["episodes"][0]["stage_entry_evidence"]
    )

    del diagnosis["episodes"][0]["stage_entry_evidence"]
    with pytest.raises(ValueError, match="stage_entry_evidence"):
        select_reset_candidates(settings, diagnosis)


def test_pre_causal_selection_requires_concrete_success_comparison(tmp_path):
    settings = ExperimentSettings(
        task=TaskSettings("libero_goal", 0, tmp_path / "ckpt"),
        reset=ResetSettings(candidate_selection="pre_causal_only"),
    )
    diagnosis = {
        "successful_reference_episodes": [
            {"episode_index": 1, "reason": "successful reference"}
        ],
        "episodes": [
            {
                "episode_index": 3,
                "failure_phase": "transport",
                "first_causal_frame_index": 28,
                "recoverable_window_start_frame_index": 15,
                "recoverable_window_stop_frame_index": 28,
                "successful_reference_episode_indices": [1],
            }
        ],
    }

    with pytest.raises(ValueError, match="successful_reference_comparison"):
        select_reset_candidates(settings, diagnosis)


def test_v5_candidates_add_exact_pre_window_prevention_point(tmp_path):
    settings = ExperimentSettings(
        task=TaskSettings("libero_goal", 0, tmp_path / "ckpt"),
        reset=ResetSettings(
            candidate_selection="pre_window_and_endpoints",
            prevention_steps=10,
        ),
    )
    diagnosis = {
        "episodes": [
            {
                "episode_index": 3,
                "failure_phase": "transport",
                "recoverable_window_start_frame_index": 15,
                "recoverable_window_stop_frame_index": 28,
            }
        ]
    }

    value = select_reset_candidates(settings, diagnosis)

    assert [item["requested_frame_index"] for item in value["candidates"]] == [5, 15, 28]
    assert [item["intervention_point"] for item in value["candidates"]] == [
        "pre_window",
        "window_start",
        "window_stop",
    ]
    assert value["selection"] == "pre-window prevention plus failure window endpoints"
    assert value["prevention_steps"] == 10
    assert value["frames_per_failure"] == 3


def test_v5_prevention_point_must_be_exact_and_outside_window(tmp_path):
    settings = ExperimentSettings(
        task=TaskSettings("libero_goal", 0, tmp_path / "ckpt"),
        reset=ResetSettings(
            candidate_selection="pre_window_and_endpoints",
            prevention_steps=10,
        ),
    )
    diagnosis = {
        "episodes": [
            {
                "episode_index": 3,
                "failure_phase": "grasp",
                "recoverable_window_start_frame_index": 9,
                "recoverable_window_stop_frame_index": 18,
            }
        ]
    }

    with pytest.raises(ValueError, match="exact 10-step pre-window"):
        select_reset_candidates(settings, diagnosis)


def test_materialization_can_reuse_frozen_v4_rollout_for_v5_reset_only_change(tmp_path):
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    source = ExperimentSettings(
        task=TaskSettings("libero_goal", 0, tmp_path / "ckpt"),
        reset=ResetSettings(candidate_selection="window_endpoints"),
    )
    target = ExperimentSettings(
        task=source.task,
        reset=ResetSettings(
            candidate_selection="pre_window_and_endpoints", prevention_steps=10
        ),
    )
    summary = {
        "schema_version": 2,
        "settings_fingerprint": source.fingerprint(),
        "trajectory_protocol": {
            "name": "vla-mender.libero.openpi",
            "version": 2,
            "state_timing": "pre_action",
            "transition": "state_t_and_action_t_lead_to_state_t_plus_1",
            "visibility": "public_observation_only",
        },
        "episodes": [],
        "overall": {"episodes": 0, "successes": 0, "failures": 0},
    }
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "settings_fingerprint": source.fingerprint(),
                "requested_settings": source.as_dict(),
            }
        )
    )

    fingerprint, reused = _validate_materialization_rollout_contract(
        target, rollout, summary
    )

    assert fingerprint == source.fingerprint()
    assert reused is True


def test_materialization_rejects_non_reset_drift_in_frozen_rollout(tmp_path):
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    source = ExperimentSettings(
        task=TaskSettings("libero_goal", 0, tmp_path / "ckpt"),
        reset=ResetSettings(candidate_selection="window_endpoints"),
    )
    target = ExperimentSettings(
        task=TaskSettings("libero_goal", 1, tmp_path / "ckpt"),
        reset=ResetSettings(candidate_selection="pre_window_and_endpoints"),
    )
    summary = {
        "schema_version": 2,
        "settings_fingerprint": source.fingerprint(),
        "trajectory_protocol": {
            "name": "vla-mender.libero.openpi",
            "version": 2,
            "state_timing": "pre_action",
            "transition": "state_t_and_action_t_lead_to_state_t_plus_1",
            "visibility": "public_observation_only",
        },
        "episodes": [],
        "overall": {"episodes": 0, "successes": 0, "failures": 0},
    }
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "settings_fingerprint": source.fingerprint(),
                "requested_settings": source.as_dict(),
            }
        )
    )

    with pytest.raises(ValueError, match="rollout-affecting contract drift"):
        _validate_materialization_rollout_contract(target, rollout, summary)


def test_diagnosis_requires_all_failed_episodes(tmp_path):
    rollout = tmp_path / "rollout"
    (rollout / "episodes").mkdir(parents=True)
    (rollout / "summary.json").write_text(json.dumps({"episodes": [
        {"episode_index": 0, "num_steps": 10, "success": False},
        {"episode_index": 1, "num_steps": 10, "success": True},
    ]}))
    diagnosis = tmp_path / "diagnosis.json"
    diagnosis.write_text(json.dumps({"schema_version": 1, "episodes": []}))
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_diagnosis(rollout, diagnosis)


def test_diagnosis_allows_empty_modes_when_rollout_has_no_failures(tmp_path):
    rollout = tmp_path / "rollout"
    rollout.mkdir(parents=True)
    (rollout / "summary.json").write_text(json.dumps({"episodes": [
        {"episode_index": 0, "num_steps": 10, "success": True},
    ]}))
    diagnosis = tmp_path / "diagnosis.json"
    diagnosis.write_text(json.dumps({
        "schema_version": 1,
        "successful_reference_episodes": [{"episode_index": 0, "reason": "reference"}],
        "failure_modes": [],
        "episodes": [],
    }))

    value = validate_diagnosis(rollout, diagnosis)

    assert value["failure_modes"] == []
    assert select_reset_candidates(
        ExperimentSettings(task=TaskSettings("libero_goal", 0, tmp_path / "ckpt")),
        value,
    )["candidates"] == []


def test_diagnosis_requires_causal_at_window_stop_without_fixed_minimum_span(tmp_path):
    rollout = tmp_path / "rollout"
    rollout.mkdir(parents=True)
    (rollout / "summary.json").write_text(json.dumps({"episodes": [
        {"episode_index": 0, "num_steps": 40, "success": True},
        {"episode_index": 3, "num_steps": 100, "success": False},
    ]}))
    diagnosis = tmp_path / "diagnosis.json"
    record = {
        "episode_index": 3,
        "failure_phase": "placement",
        "failure_mode_id": "FM-01",
        "failure_category": "placement",
        "failure_mode": "rim contact",
        "first_causal_frame_index": 30,
        "recoverable_window_start_frame_index": 10,
        "recoverable_window_stop_frame_index": 35,
        "evidence": ["public evidence"],
        "confidence": 0.9,
    }
    value = {
        "schema_version": 1,
        "successful_reference_episodes": [{
            "episode_index": 0,
            "reason": "phase-aligned successful reference",
        }],
        "failure_modes": [{
            "failure_mode_id": "FM-01",
            "label": "rim contact",
            "category": "placement",
            "episode_indices": [3],
        }],
        "episodes": [record],
    }
    diagnosis.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="invalid causal/window"):
        validate_diagnosis(rollout, diagnosis)

    record["recoverable_window_stop_frame_index"] = 30
    record["recoverable_window_start_frame_index"] = 29
    diagnosis.write_text(json.dumps(value))
    assert validate_diagnosis(rollout, diagnosis)["episodes"] == [record]
