from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from run.repair.generate_prompt import generate
from workflow.research import RepairCampaign
from workflow.research.config import RepairConfigError, load_repair_config, resolve_repair_inputs


def _publish_unified_handoff(settings_path: Path) -> Path:
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    run_root = Path(settings["tasks"][0]["run_root"])
    failure_root = run_root / "failure_diagnosis"
    handoff_root = run_root / "repair_handoff"
    shutil.copytree(failure_root / "private_reset_states", handoff_root / "private_reset_states")
    shutil.copytree(failure_root / "agent_views", handoff_root / "agent_views")
    jobs = json.loads((failure_root / "repair_jobs.json").read_text(encoding="utf-8"))["jobs"]
    public = json.loads(
        (failure_root / "public_reset_bank.json").read_text(encoding="utf-8")
    )["resets"]
    resets = []
    for job, report in zip(jobs, public, strict=True):
        resets.append(
            {
                **report,
                **job,
                "requested_frame_index": job["reset_frame_index"],
                "reset_state": f"private_reset_states/{job['reset_state']}",
                "reset_state_file_sha256": report["private_state_file_sha256"],
                "agent_view": f"agent_views/{job['agent_view']}",
                "agent_view_file_sha256": report["agent_view_sha256"],
                "verified": True,
            }
        )
    diagnosis = {
        "schema_version": 1,
        "successful_reference_episodes": [],
        "failure_modes": [{"failure_mode_id": "FM-01"}],
        "episodes": [
            {"episode_index": item["episode_index"], "failure_mode_id": "FM-01"}
            for item in resets
        ],
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "vla_mender.repair_handoff",
        "complete": True,
        "settings_fingerprint": "test-fingerprint",
        "source": {"run_root": str(run_root)},
        "diagnosis": diagnosis,
        "selection": {"frames_per_failure": 1, "frame_stride": 5},
        "summary": {
            "failure_episode_count": len(resets),
            "failure_mode_count": 1,
            "reset_count": len(resets),
            "replay_verified_count": len(resets),
            "all_replays_verified": True,
        },
        "resets": resets,
    }
    handoff_path = handoff_root / "manifest.json"
    handoff_path.write_text(json.dumps(manifest), encoding="utf-8")
    return handoff_path


def test_generator_consumes_only_prepared_run_root(repair_settings: Path) -> None:
    manifest = generate(repair_settings)
    output = Path(manifest["prompt"]).parent
    expected = {
        "repair_resolved.yaml",
        "prompt_generated.md",
        "prompt_manifest.json",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    assert not (output / "repair_jobs_resolved.json").exists()
    assert all(path.name.count(".") == 1 for path in output.iterdir() if path.is_file())

    resolved = yaml.safe_load((output / "repair_resolved.yaml").read_text(encoding="utf-8"))
    assert "settings_fingerprint" not in manifest
    assert "settings_fingerprint" not in resolved
    assert set(resolved["tasks"][0]) == {"run_root"}
    assert resolved["environment"]["python"] == sys.executable
    assert resolved["artifacts"]["evidence_dedupe"] == "auto"
    partitions = [job["initial_partition"] for job in resolved["jobs"]]
    assert partitions.count("debug") == 2
    assert partitions.count("validation") == 3
    prompt = (output / "prompt_generated.md").read_text(encoding="utf-8")
    assert "RepairCampaign" in prompt
    assert "no final" in prompt
    assert "prepared_task_libero_goal_task3" in prompt


def test_generator_prefers_single_unified_handoff(repair_settings: Path) -> None:
    handoff_path = _publish_unified_handoff(repair_settings)
    manifest = generate(repair_settings)
    resolved = yaml.safe_load(Path(manifest["resolved_settings"]).read_text(encoding="utf-8"))
    task = resolved["resolved_tasks"][0]
    assert task["repair_handoff"] == str(handoff_path.parent)
    assert task["handoff_manifest"] == str(handoff_path)
    assert "diagnosis" not in task
    assert all("repair_handoff" in job["reset_state"] for job in resolved["jobs"])


def test_generator_refreshes_changed_settings_in_same_output(repair_settings: Path) -> None:
    first = generate(repair_settings)
    assert "settings_fingerprint" not in first
    settings = yaml.safe_load(repair_settings.read_text(encoding="utf-8"))
    settings["runtime"]["max_steps"] = 37
    repair_settings.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")

    second = generate(repair_settings)
    resolved = yaml.safe_load(Path(second["resolved_settings"]).read_text(encoding="utf-8"))
    assert "settings_fingerprint" not in second
    assert "settings_fingerprint" not in resolved
    assert resolved["runtime"]["max_steps"] == 37


def test_v2_prompt_requires_one_task_per_agent_without_coordinator(
    repair_settings_v2: Path,
) -> None:
    manifest = generate(repair_settings_v2)
    prompt = Path(manifest["prompt"]).read_text(encoding="utf-8")

    assert "## Required task-agent assignment" in prompt
    assert "You are one of the IDE task agents" in prompt
    assert "no coordination-only agent" in prompt
    assert "including the initial agent receiving this prompt" in prompt
    assert "initial agent owns exactly one prepared task" in prompt
    assert "every peer task subagent owns exactly one other prepared task" in prompt
    assert "one prepared task has at most one live task agent" in prompt
    assert "no agent is coordination-only or combines multiple prepared tasks" in prompt
    assert "immediately launches one\npeer task subagent for each remaining task" in prompt
    assert "Run up to `1` task agents concurrently" in prompt
    assert "undispatched tasks queued for a new dedicated agent" in prompt


def test_campaign_reads_legacy_split_job_inventory(repair_settings: Path) -> None:
    manifest = generate(repair_settings)
    resolved_path = Path(manifest["resolved_settings"])
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    jobs = resolved.pop("jobs")
    inventory_path = resolved_path.parent / "repair_jobs_resolved.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": resolved["schema_version"],
                "tasks": resolved["resolved_tasks"],
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )
    resolved["resolved_jobs"] = str(inventory_path)
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    campaign = RepairCampaign.open(resolved_path)
    assert len(campaign.jobs) == len(jobs)


def test_unified_handoff_ignores_global_fingerprint_mismatch(
    repair_settings: Path,
) -> None:
    _publish_unified_handoff(repair_settings)
    settings = yaml.safe_load(repair_settings.read_text(encoding="utf-8"))
    summary_path = Path(settings["tasks"][0]["run_root"]) / "rollout" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["settings_fingerprint"] = "different-rollout-revision"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    resolved = resolve_repair_inputs(load_repair_config(repair_settings))
    assert resolved["jobs"]


def test_unified_handoff_rejects_attachment_hash_mismatch(repair_settings: Path) -> None:
    handoff_path = _publish_unified_handoff(repair_settings)
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff["resets"][0]["reset_state_file_sha256"] = "0" * 64
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    with pytest.raises(RepairConfigError, match="reset state file hash mismatch"):
        resolve_repair_inputs(load_repair_config(repair_settings))
