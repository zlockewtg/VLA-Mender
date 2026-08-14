from __future__ import annotations

import json
from pathlib import Path

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
  frames_per_failure: 3
  frame_stride: 5
  dynamics: preserve_full_state
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
    manifest_path = tmp_path / "prompt.generated.manifest.json"
    prompt = prompt_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert f"--settings {settings.resolve()}" in prompt
    assert f"--output {tmp_path.resolve()}" in prompt
    assert "{{" not in prompt
    assert "<resolved-settings.yaml>" not in prompt
    assert manifest["settings"] == str(settings.resolve())
    assert manifest["run_root"] == str(tmp_path.resolve())
    assert manifest["prompt"] == str(prompt_path.resolve())
    assert manifest["prompt_sha256"]


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
