from __future__ import annotations

from pathlib import Path

import pytest

from run.repair.generate_prompt import generate
from workflow.research import RepairCampaign
from workflow.research.runtime import _preflight_contract


def _campaign(repair_settings: Path) -> RepairCampaign:
    manifest = generate(repair_settings)
    return RepairCampaign.open(manifest["resolved_settings"])


def test_preflight_cache_uses_explicit_environment_contract() -> None:
    project = {
        "source_root": "/repo/src",
        "knowledge_root": "/repo/knowledge",
    }
    environment = {
        "python": "/venv/bin/python",
        "working_directory": "/repo",
        "libero_root": "/data/libero",
        "env": {"SAM3_URL": "http://127.0.0.1:14014"},
    }
    contract = _preflight_contract(project, environment)
    assert "settings_fingerprint" not in contract
    assert contract["python"] == "/venv/bin/python"
    assert contract["libero_root"] == "/data/libero"

    changed = {**environment, "python": "/other/bin/python"}
    assert _preflight_contract(project, changed) != contract


def test_fake_workers_resume_and_publish_experience(repair_settings: Path) -> None:
    campaign = _campaign(repair_settings)
    task_key = next(iter(campaign.tasks))
    task = campaign.open_task(task_key, gpu_id=0)
    program = task.programs.save("RESULT = True", mode_ids=["FM-01"])
    results = task.evaluate(program.id, mode_ids=["FM-01"])
    assert len(results) == 5
    assert all(result["success"] for result in results)
    assert not any(result["cached"] for result in results)

    one = task.evaluate(program.id, reset_ids=[task.jobs()[0]["source_job_id"]])
    assert one[0]["cached"] is True
    repeated = task.evaluate(
        program.id,
        reset_ids=[task.jobs()[0]["source_job_id"]],
        force=True,
    )
    assert repeated[0]["cached"] is False
    assert "reruns" in Path(repeated[0]["attempt_path"]).parts

    # Two overlapping mode experiments that select the same new reset/program
    # execute once; the waiter resumes the committed result.
    fork = task.programs.save("RESULT = 'fork'", mode_ids=["FM-01"])
    reset_id = task.jobs()[0]["job_id"]
    first_handle = task.evaluate_async(fork.id, reset_ids=[reset_id])
    second_handle = task.evaluate_async(fork.id, reset_ids=[reset_id])
    overlap = [first_handle.results()[0], second_handle.results()[0]]
    assert sorted(result["cached"] for result in overlap) == [False, True]
    published = campaign.experience.publish_program(
        task_key, "FM-01", program.id, results, description="verified fake repair"
    )
    skill = campaign.experience.publish_skill(
        "approach_cup",
        "def approach_cup():\n    return 'approach'\n",
        derived_from=program.id,
        task_key=task_key,
        mode_ids=["FM-01"],
    )
    assert Path(published["path"]).is_file()
    assert campaign.experience.source(skill["id"]).startswith("def approach_cup")
    task.close(status="completed")


def test_v2_fully_cached_results_do_not_start_runtime(
    repair_settings_v2: Path, monkeypatch
) -> None:
    campaign = _campaign(repair_settings_v2)
    task_key = next(iter(campaign.tasks))
    task = campaign.open_task(task_key, gpu_id=0)
    program = campaign.program_store.save(
        "RESULT = True",
        task_key=task_key,
        mode_ids=["FM-01"],
    )
    job = task.jobs()[0]
    pending = campaign.attempt_store.begin(program, job)
    result = {
        "schema_version": 1,
        "outcome": "success",
        "success": True,
        "job_id": job["job_id"],
        "source_job_id": job["source_job_id"],
        "program_sha256": program.sha256,
    }
    campaign.attempt_store.commit(
        pending,
        program=program,
        job=job,
        result=result,
    )

    def reject_runtime_start() -> None:
        raise AssertionError("a fully cached evaluation must not start runtime services")

    monkeypatch.setattr(task, "ensure_runtime", reject_runtime_start)
    recovered = task._evaluate_program(program, [job])

    assert recovered[0]["cached"] is True
    assert recovered[0]["canonical"] is True
    assert recovered[0]["source_job_id"] == job["source_job_id"]
    task.close(status="in_progress")


def test_task_and_gpu_lease_prevent_duplicate_agent(repair_settings: Path) -> None:
    first = _campaign(repair_settings)
    second = RepairCampaign.open(first.settings_path)
    task_key = next(iter(first.tasks))
    first_task = first.open_task(task_key, gpu_id=0)
    first_task.ensure_runtime()
    second_task = second.open_task(task_key, gpu_id=0)
    with pytest.raises(RuntimeError, match="already leased"):
        second_task.ensure_runtime()
    first_task.close(status="completed")
    second_task.close(status="not_started")
