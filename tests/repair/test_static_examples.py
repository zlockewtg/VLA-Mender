from __future__ import annotations

import json
from pathlib import Path

import yaml

from run.repair.generate_prompt import generate
from workflow.research import RepairCampaign


REMOVED_FIELDS = {
    "borrowed_from",
    "consulted_knowledge_refs",
    "knowledge_intake_ref",
    "knowledge_intake_sha256",
}


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def test_example_and_skill_catalogs_remain_directly_readable() -> None:
    knowledge = _repo() / "vla-mender" / "knowledge"
    root = json.loads((knowledge / "examples" / "manifest.json").read_text())
    assert [item["id"] for item in root["catalogs"]] == ["aspire", "repair"]

    aspire = json.loads((knowledge / "examples" / "aspire" / "manifest.json").read_text())
    repair = json.loads((knowledge / "examples" / "repair" / "manifest.json").read_text())
    skills = json.loads((knowledge / "skills" / "manifest.json").read_text())
    assert aspire["strategy_count"] == len(aspire["strategies"]) == 84
    assert repair["strategy_count"] == len(repair["strategies"]) == 2
    assert skills["skill_count"] == len(skills["skills"]) == 6
    assert (knowledge / "examples" / "aspire" / aspire["strategies"][0]["relative_path"]).is_file()
    assert (knowledge / "examples" / "repair" / repair["strategies"][0]["relative_path"]).is_file()
    assert (knowledge / "skills" / skills["skills"][0]["relative_path"]).is_file()
    assert (knowledge / "api" / "README.md").is_file()


def test_generated_prompt_uses_only_prompt_guided_filesystem_review(
    repair_settings_v2: Path,
) -> None:
    manifest = generate(repair_settings_v2)
    assert "knowledge_indexes" not in manifest
    assert "selected_static_examples" not in manifest

    prompt = Path(manifest["prompt"]).read_text(encoding="utf-8")
    for required in (
        "Required task-start knowledge review",
        "task_start_knowledge_review.json",
        "Do not print the review",
        "action_analysis",
        "reusable_code_checklist",
        "searched_scopes",
        "/examples/aspire",
        "/examples/repair",
        "/skills/WORKFLOW.md",
        "/api/README.md",
        "libero_backend.py",
        "_api_for",
        "functions()",
        "no fixed top-k",
        "none found",
        "smallest clear, robust code change",
        "Functional completion and trajectory continuity",
        "stop-start motion",
        "Only then promote the final solution",
    ):
        assert required in prompt
    for removed in (
        "campaign.knowledge",
        "record_knowledge_intake",
        "knowledge_intake",
        "borrowed_from",
        "consulted_knowledge_refs",
        "knowledge_indexes",
        "Settings fingerprint",
    ):
        assert removed not in prompt


def test_candidate_runtime_and_promotion_need_no_knowledge_artifact(
    repair_settings_v2: Path,
) -> None:
    manifest = generate(repair_settings_v2)
    campaign = RepairCampaign.open(manifest["resolved_settings"])
    task_key = next(iter(campaign.tasks))
    task = campaign.open_task(task_key, gpu_id=0)
    fm = task.open_failure_mode("FM-01")
    seeds = [job["source_job_id"] for job in fm.jobs]

    candidate = fm.create_initial_candidate(
        source="FAKE_SUCCESS_SEEDS = " + repr(seeds),
        representative_seed_ids=seeds[:3],
        strategy_summary="prompt-guided strategy with no persisted knowledge inventory",
    )
    candidate.evaluate_smoke()
    candidate.decide("expand", rationale="all representative seeds succeeded")
    candidate.evaluate_remaining_fm_seeds()
    promotion = fm.promote_best(candidate.id)
    assert promotion.promoted is True

    task_state = json.loads(task.task_state_path.read_text(encoding="utf-8"))
    candidate_manifest = candidate.manifest()
    best_manifest = json.loads((fm.root / "current_best" / "manifest.json").read_text())
    events = [json.loads(line) for line in fm.progress_path.read_text().splitlines()]
    created = next(item for item in events if item["event"] == "candidate_created")
    assert "knowledge_intake" not in task_state
    assert not (task.task_root / "knowledge_intake.json").exists()
    assert REMOVED_FIELDS.isdisjoint(candidate_manifest)
    assert REMOVED_FIELDS.isdisjoint(best_manifest)
    assert REMOVED_FIELDS.isdisjoint(created)
    task.close(status="completed")


def test_repair_campaign_artifacts_have_no_settings_fingerprint(
    repair_settings_v2: Path,
) -> None:
    manifest = generate(repair_settings_v2)
    resolved_path = Path(manifest["resolved_settings"])
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert "settings_fingerprint" not in manifest
    assert "settings_fingerprint" not in resolved

    campaign = RepairCampaign.open(resolved_path)
    assert campaign.jobs
