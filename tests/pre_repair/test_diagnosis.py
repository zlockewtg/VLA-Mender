import json

import pytest

from workflow.failure_diagnosis import select_reset_candidates, validate_diagnosis
from workflow.parameters import ExperimentSettings, ResetSettings, TaskSettings


def test_stride_candidates_are_inside_window(tmp_path):
    settings = ExperimentSettings(task=TaskSettings("libero_goal", 0, tmp_path / "ckpt"),
                                  reset=ResetSettings(frames_per_failure=3, frame_stride=4))
    diagnosis = {"episodes": [{"episode_index": 3, "failure_phase": "transport",
                                "recoverable_window_start_frame_index": 5,
                                "recoverable_window_stop_frame_index": 18}]}
    value = select_reset_candidates(settings, diagnosis)
    assert [item["requested_frame_index"] for item in value["candidates"]] == [5, 9, 13]


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
