from __future__ import annotations

import json
from pathlib import Path

import pytest

from run.pre_repair.generate_prompt import main as generate_prompt
from workflow.pipeline import prepare_prompt


def _settings(tmp_path: Path) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """\
task:
  suite: libero_goal
  task_id: 0
  checkpoint: ./checkpoint
  policy_config: pi0_libero
  task_description: open the middle drawer
initial_states:
  provider: official
  count: 2
  seed_start: 100000
  state_manifest: null
rollout:
  control_frequency_hz: 20
  max_steps: 300
  policy_seed: 7
  gpus: [0]
  workers_per_gpu: 1
  action_chunk: 5
  inference_steps: 5
  num_steps_wait: 10
  binary_gripper: false
  gripper_hysteresis_threshold: 0.2
controller:
  source_control_space: osc
  target_control_space: osc
reset:
  candidate_selection: pre_window_and_endpoints
  prevention_steps: 10
  dynamics: preserve_full_state
backend:
  name: openpi
  openpi_environment: /opt/venv/openpi
  openpi_source: ./openpi
""",
        encoding="utf-8",
    )
    return path


def _campaign_settings(tmp_path: Path) -> Path:
    path = tmp_path / "campaign.yaml"
    path.write_text(
        """\
task:
  checkpoint: ./checkpoint
  policy_config: pi0_libero
  task_description: null
tasks:
  - {key: object-0, suite: libero_object, task_id: 0}
  - {key: goal-0, suite: libero_goal, task_id: 0}
initial_states: {provider: official, count: 2}
rollout: {gpus: [0], max_steps: 300}
controller: {source_control_space: osc, target_control_space: osc}
reset: {candidate_selection: pre_window_and_endpoints, prevention_steps: 10, dynamics: preserve_full_state}
backend:
  name: openpi
  openpi_environment: /opt/venv/openpi
  openpi_source: ./openpi
""",
        encoding="utf-8",
    )
    return path


def test_settings_only_generation_uses_yaml_directory_as_run_root(tmp_path):
    settings = _settings(tmp_path)

    generate_prompt(["--settings", str(settings)])

    prompt_path = tmp_path / "prompt.generated.md"
    prompt = prompt_path.read_text(encoding="utf-8")
    assert f"--settings {settings.resolve()}" in prompt
    assert f"--output {tmp_path.resolve()}" in prompt
    assert "{{" not in prompt
    assert "<resolved-settings.yaml>" not in prompt
    assert "`diagnosis.json` alone is\nnot completion" in prompt
    assert "Stage 6 — required reset-bank materialization" in prompt
    assert f"{tmp_path.resolve()}/repair_handoff/" in prompt
    assert '"artifact_type": "vla_mender.repair_handoff"' in prompt
    assert '"method": "pre-window prevention plus failure window endpoints"' in prompt
    assert '"pre_window"' in prompt
    assert '"window_start"' in prompt
    assert '"window_stop"' in prompt
    assert '"prevention_steps": 10' in prompt
    assert '"frames_per_failure": 3' in prompt
    assert "pre_window = recoverable_window_start_frame_index - 10" in prompt
    assert "No interior stride sampling" in prompt
    assert "same semantic subtask phase that fails" in prompt
    assert "`0 <= start < stop == causal < num_frames`" in prompt
    assert not (tmp_path / "prompt.generated.manifest.json").exists()


def test_window_start_only_prompt_materializes_one_stage_start_reset(tmp_path):
    settings = _settings(tmp_path)
    settings.write_text(
        settings.read_text(encoding="utf-8").replace(
            "candidate_selection: pre_window_and_endpoints",
            "candidate_selection: window_start_only",
        ),
        encoding="utf-8",
    )

    generate_prompt(["--settings", str(settings)])

    prompt = (tmp_path / "prompt.generated.md").read_text(encoding="utf-8")
    assert "Reset candidates per failure: exactly 1 (`window_start`)" in prompt
    assert "Every failed trajectory contributes exactly one intervention point" in prompt
    assert "Do not materialize `pre_window`,\n`window_stop`, or any interior frame" in prompt
    assert '"method": "failure window start only"' in prompt
    assert '"intervention_points": [' in prompt
    assert '"window_start"' in prompt
    assert '"frames_per_failure": 1' in prompt
    assert "Preventive lead" not in prompt
    assert "{{" not in prompt


def test_pre_causal_prompt_requires_success_aligned_last_normal_state(tmp_path):
    settings = _settings(tmp_path)
    settings.write_text(
        settings.read_text(encoding="utf-8").replace(
            "candidate_selection: pre_window_and_endpoints",
            "candidate_selection: pre_causal_only",
        ),
        encoding="utf-8",
    )

    generate_prompt(["--settings", str(settings)])

    prompt = (tmp_path / "prompt.generated.md").read_text(encoding="utf-8")
    assert "Reset candidates per failure: exactly 1 (`pre_causal`)" in prompt
    assert "pre_causal = first_causal_frame_index - 1" in prompt
    assert "phase-aligning both\ncamera views" in prompt
    assert '"method": "successful-reference-aligned pre-causal state only"' in prompt
    assert '"pre_causal"' in prompt
    assert '"frames_per_failure": 1' in prompt
    assert "successful_reference_episode_indices" in prompt
    assert "successful_reference_comparison" in prompt
    assert "{{" not in prompt


def test_failed_stage_entry_prompt_materializes_one_stage_entry(tmp_path):
    settings = _settings(tmp_path)
    settings.write_text(
        settings.read_text(encoding="utf-8").replace(
            "candidate_selection: pre_window_and_endpoints",
            "candidate_selection: failed_stage_entry_only",
        ),
        encoding="utf-8",
    )

    generate_prompt(["--settings", str(settings)])

    prompt = (tmp_path / "prompt.generated.md").read_text(encoding="utf-8")
    assert "Reset candidates per failure: exactly 1 (`stage_entry`)" in prompt
    assert "stage_entry = intervention_stage_start_frame_index" in prompt
    assert "`pick` includes target approach, alignment, and grasp acquisition" in prompt
    assert "off-center basket approach or rim contact is a\ntransport-stage failure" in prompt
    assert '"method": "failed behavior stage entry only"' in prompt
    assert '"stage_entry"' in prompt
    assert '"frames_per_failure": 1' in prompt
    assert "{{" not in prompt


def test_per_episode_stage_entry_prompt_forbids_fixed_frames(tmp_path):
    settings = _settings(tmp_path)
    settings.write_text(
        settings.read_text(encoding="utf-8").replace(
            "candidate_selection: pre_window_and_endpoints",
            "candidate_selection: per_episode_stage_entry_only",
        ),
        encoding="utf-8",
    )

    generate_prompt(["--settings", str(settings)])

    prompt = (tmp_path / "prompt.generated.md").read_text(encoding="utf-8")
    assert "infer a task-specific observable behavior-stage graph" in prompt
    assert "Do not assume a fixed stage vocabulary" in prompt
    assert "Never reuse a frame number or offset" in prompt
    assert "why_previous_frame_is_too_early" in prompt
    assert "why_later_frame_is_too_late" in prompt
    assert "camera_evidence" in prompt
    assert "alphabet" not in prompt.lower()
    assert "stage_entry_evidence" in prompt
    assert '"method": "per-episode observed behavior stage entry only"' in prompt
    assert "{{" not in prompt


def test_pipeline_prompt_renders_run_root_not_diagnosis_directory(tmp_path):
    settings = _settings(tmp_path)
    run_root = tmp_path / "run"

    manifest = prepare_prompt(settings, run_root)

    resolved = run_root / "experiment.resolved.yaml"
    prompt_path = run_root / "failure_diagnosis" / "prompt.md"
    prompt = prompt_path.read_text(encoding="utf-8")
    assert f"--settings {resolved.resolve()}" in prompt
    assert f"--output {run_root.resolve()}" in prompt
    assert f"--output {(run_root / 'failure_diagnosis').resolve()}" not in prompt
    assert manifest["run_root"] == str(run_root.resolve())
    assert manifest["resolved_settings"] == str(resolved.resolve())
    assert (run_root / "prompt_manifest.json").is_file()


def test_campaign_prompt_materializes_ordered_isolated_task_contracts(tmp_path):
    settings = _campaign_settings(tmp_path)

    generate_prompt(["--settings", str(settings)])

    prompt = (tmp_path / "prompt.generated.md").read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "campaign_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_type"] == "vla_mender.pre_repair_campaign"
    assert manifest["execution"] == "sequential_fail_fast"
    assert [task["key"] for task in manifest["tasks"]] == ["object-0", "goal-0"]
    assert "complete the full task-level prompt for one task before starting the\nnext task" in prompt
    assert "Start the next listed task only after" in prompt
    assert prompt.index("`object-0`") < prompt.index("`goal-0`")
    for index, task in enumerate(manifest["tasks"]):
        task_root = tmp_path / "tasks" / f"{index:03d}_{task['key']}"
        resolved = task_root / "experiment.resolved.yaml"
        task_prompt = task_root / "failure_diagnosis" / "prompt.md"
        assert resolved.is_file()
        assert task_prompt.is_file()
        rendered = task_prompt.read_text(encoding="utf-8")
        assert f"--settings {resolved.resolve()}" in rendered
        assert f"--output {task_root.resolve()}" in rendered
        assert "{{" not in rendered


def test_pipeline_prompt_accepts_campaign_and_writes_campaign_prompt(tmp_path):
    settings = _campaign_settings(tmp_path)
    run_root = tmp_path / "campaign-run"

    manifest = prepare_prompt(settings, run_root)

    assert manifest["task_count"] == 2
    assert (run_root / "campaign_prompt.md").is_file()
    assert (run_root / "tasks" / "000_object-0" / "experiment.resolved.yaml").is_file()
    assert (run_root / "tasks" / "001_goal-0" / "experiment.resolved.yaml").is_file()


def test_campaign_regeneration_reuses_matching_contract_and_rejects_drift(tmp_path):
    settings = _campaign_settings(tmp_path)
    generate_prompt(["--settings", str(settings)])
    generate_prompt(["--settings", str(settings)])

    changed = settings.read_text(encoding="utf-8").replace(
        "task_id: 0}", "task_id: 1}", 1
    )
    settings.write_text(changed, encoding="utf-8")

    with pytest.raises(FileExistsError, match="different campaign contract"):
        generate_prompt(["--settings", str(settings)])
