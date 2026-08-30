from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from run.repair.generate_prompt import generate
from workflow.research import (
    ExplorationReviewRequired,
    RepairCampaign,
    SoftBudgetReviewRequired,
)


def _campaign(settings: Path) -> RepairCampaign:
    manifest = generate(settings)
    return RepairCampaign.open(manifest["resolved_settings"])


def _open_task(campaign: RepairCampaign, task_key: str):
    return campaign.open_task(task_key, gpu_id=0)


def _program(*successes: str) -> str:
    return "FAKE_SUCCESS_SEEDS = " + repr(list(successes))


def _record_failure_videos(candidate, seed_ids: list[str] | None = None) -> None:
    manifest = candidate.manifest()
    selected = seed_ids or [
        seed
        for seed, result in manifest.get("results", {}).items()
        if result.get("outcome") == "policy_failure"
    ]
    analyses = {
        seed: {
            "wide_view": f"wide view shows the scene-level failure for {seed}",
            "wrist_view": f"wrist view shows the contact failure for {seed}",
            "failure_moment": "the visible deviation starts immediately before contact",
            "mechanism_evidence": "both camera views support the shared policy mechanism",
        }
        for seed in selected
        if manifest.get("results", {}).get(seed, {}).get("outcome") == "policy_failure"
    }
    candidate.record_failure_video_analysis(analyses)


def test_v2_config_has_one_open_seed_pool_and_v2_prompt(repair_settings_v2: Path) -> None:
    manifest = generate(repair_settings_v2)
    resolved = yaml.safe_load(Path(manifest["resolved_settings"]).read_text(encoding="utf-8"))
    assert resolved["schema_version"] == 2
    assert all("initial_partition" not in job for job in resolved["jobs"])
    assert resolved["repair"] == {
        "budget": {"soft_task_hours": 4.0},
        "smoke": {"min_seeds": 3, "max_seeds": 8},
        "exploration_review": {
            "consecutive_no_gain_candidates": 3,
            "per_seed_policy_attempts": 8,
        },
        "allow_abandon": True,
    }
    prompt = Path(manifest["prompt"]).read_text(encoding="utf-8").lower()
    for removed_term in ("debug", "validation", "final split", "initial_partition"):
        assert removed_term not in prompt
    assert "candidate smoke gate: `3` to `8` seeds" in prompt
    assert "promote_best" in prompt
    assert "trajectory trace alone is never sufficient" in prompt
    assert "record_failure_video_analysis" in prompt
    assert "successful-video vs failed-video contact-path comparison" in prompt
    assert "substantially finer frame interval" in prompt
    assert "stop blind" in prompt and "policy iteration" in prompt
    assert "another candidate before completing" in prompt
    assert "task_start_knowledge_review.json" in prompt
    assert "smallest clear, robust code change" in prompt
    assert "functional completion and trajectory continuity" in prompt
    assert "stop-start motion" in prompt
    assert "only then promote the final solution" in prompt


def test_v2_net_coverage_single_experience_and_partial_finish(
    repair_settings_v2: Path,
) -> None:
    campaign = _campaign(repair_settings_v2)
    task_key = next(iter(campaign.tasks))
    task = _open_task(campaign, task_key)
    with pytest.raises(ValueError, match="one open seed pool"):
        task.jobs(partition="debug")
    fm = task.open_failure_mode("FM-01")
    seeds = [job["source_job_id"] for job in fm.jobs]

    initial = fm.create_initial_candidate(
        source=_program(seeds[0], seeds[1]),
        representative_seed_ids=seeds[:3],
        strategy_summary="initial grasp strategy",
    )
    initial.evaluate_smoke()
    _record_failure_videos(initial)
    initial.decide("expand", rationale="two representative evaluator successes")
    initial.evaluate_remaining_fm_seeds()
    _record_failure_videos(initial)
    first = fm.promote_best(initial.id)
    assert first.promoted is True
    assert first.candidate_successes == 2

    round_one = fm.start_round(
        mechanism="shared contact miss",
        seed_ids=seeds[2:],
        evidence_summary="all three videos show the same missed contact",
    )
    challenger = round_one.create_candidate(
        source=_program(seeds[1], seeds[2], seeds[3]),
        parent_ref=fm.current_best_ref() or "",
        change_summary="adjust one shared contact mechanism",
    )
    challenger.evaluate_smoke(seed_ids=seeds[2:])
    _record_failure_videos(challenger)
    challenger.decide("expand", rationale="two target seeds now succeed")
    challenger.evaluate_remaining_cluster_seeds()
    second = fm.promote_best(challenger.id, skills={"contact": "def contact():\n    return 1"})
    _record_failure_videos(challenger)
    assert second.promoted is True
    assert second.candidate_successes == 3
    assert second.added_seed_ids == tuple(sorted([seeds[2], seeds[3]]))
    assert second.regressed_seed_ids == (seeds[0],)

    state = fm.state()
    assert state["current_success_seed_ids"] == sorted([seeds[1], seeds[2], seeds[3]])
    assert set(state["active_failed_seed_ids"]) == {seeds[0], seeds[4]}
    items = campaign.experience.search(kinds=["fm_experience"], task_key=task_key)
    assert len(items) == 1
    assert items[0]["candidate_id"] == challenger.id
    candidate_program = Path(challenger.manifest()["path"]) / "program.py"
    current_best_program = fm.root / "current_best" / "program.py"
    experience_program = Path(items[0]["path"])
    assert {
        candidate_program.stat().st_ino,
        current_best_program.stat().st_ino,
        experience_program.stat().st_ino,
    } == {candidate_program.stat().st_ino}

    # Superseded successes are compact; current-best successes and policy failures remain full.
    initial_state = state["candidates"][initial.id]
    old_success_path = Path(initial_state["results"][seeds[0]]["attempt_path"])
    assert not (old_success_path / "wide.mp4").exists()
    current_success_path = Path(
        state["candidates"][challenger.id]["results"][seeds[1]]["attempt_path"]
    )
    current_failure_path = Path(
        state["candidates"][challenger.id]["results"][seeds[0]]["attempt_path"]
    )
    assert (current_success_path / "wide.mp4").is_file()
    assert (current_failure_path / "trajectory.json").is_file()
    canonical_attempt = fm.state()["candidates"][challenger.id]["results"][seeds[1]][
        "attempt_path"
    ]
    rerun = challenger.evaluate_stability([seeds[1]])[0]
    assert "rerun_0001" in Path(rerun["attempt_path"]).parts
    assert (
        fm.state()["candidates"][challenger.id]["results"][seeds[1]]["attempt_path"]
        == canonical_attempt
    )

    round_two = fm.start_round(
        mechanism="long-tail edge contact",
        seed_ids=[seeds[0], seeds[4]],
        evidence_summary="both active seeds retain edge-contact policy failures",
    )
    rejected = round_two.create_candidate(
        source=_program(seeds[0]),
        parent_ref=fm.current_best_ref() or "",
        change_summary="one edge-contact adjustment",
    )
    rejected.evaluate_smoke(seed_ids=[seeds[0], seeds[4]])
    _record_failure_videos(rejected)
    rejected.decide("expand", rationale="one target seed improved")
    rejected.evaluate_remaining_cluster_seeds()
    third = fm.promote_best(rejected.id)
    assert third.promoted is False
    assert fm.current_best_ref() == challenger.id
    rejected_success = Path(
        fm.state()["candidates"][rejected.id]["results"][seeds[0]]["attempt_path"]
    )
    assert not (rejected_success / "wide.mp4").exists()
    assert len(campaign.experience.search(kinds=["fm_experience"], task_key=task_key)) == 1

    with pytest.raises(RuntimeError, match="coverage requires status in_progress"):
        task.close(status="completed")

    fm.mark_abandoned(
        [seeds[0], seeds[4]],
        reason="soft-budget tradeoff: two long-tail contact failures",
    )
    experience = campaign.experience.search(
        kinds=["fm_experience"], task_key=task_key
    )[0]
    assert experience["abandoned_seed_ids"] == sorted([seeds[0], seeds[4]])
    summary = task.finish()
    assert summary == {
        "status": "completed_partial",
        "total": 5,
        "solved": 3,
        "abandoned": 2,
        "active_failed": 0,
        "finished_at": summary["finished_at"],
    }

    task_root = campaign.root / "tasks" / campaign.tasks[task_key]["task_slug"]
    directory_names = [path.name for path in task_root.rglob("*") if path.is_dir()]
    assert not any(re.fullmatch(r"[0-9a-f]{32,}", name) for name in directory_names)
    assert not any(re.fullmatch(r"[0-9a-f]{8}-[0-9a-f-]{27,}", name) for name in directory_names)


def test_v2_seed_specific_portfolio_reopens_and_adds_targeted_win(
    repair_settings_v2: Path,
) -> None:
    campaign = _campaign(repair_settings_v2)
    task_key = next(iter(campaign.tasks))
    task = _open_task(campaign, task_key)
    fm = task.open_failure_mode("FM-01")
    seeds = [job["source_job_id"] for job in fm.jobs]

    initial = fm.create_initial_candidate(
        source=_program(seeds[0], seeds[1]),
        representative_seed_ids=seeds[:3],
        strategy_summary="global baseline",
    )
    initial.evaluate_smoke()
    _record_failure_videos(initial)
    initial.decide("expand", rationale="baseline has two evaluator successes")
    initial.evaluate_remaining_fm_seeds()
    _record_failure_videos(initial)
    fm.promote_best(initial.id)
    fm.mark_abandoned(
        seeds[2:],
        reason="old assumption required one program to retain global coverage",
    )

    fm.reopen_abandoned(
        [seeds[2]],
        reason="targeted per-seed programs are now allowed",
    )
    targeted_round = fm.start_round(
        mechanism="seed_specific_contact_path",
        seed_ids=[seeds[2]],
        evidence_summary="latest wide and wrist videos show a seed-specific contact miss",
    )
    targeted = targeted_round.create_candidate(
        source=_program(seeds[2]),
        parent_ref=fm.current_best_ref() or "",
        change_summary="target only this seed's observed contact path",
    )
    targeted.evaluate_smoke(seed_ids=[seeds[2]])
    targeted.decide("expand", rationale="the target seed now succeeds")
    targeted.evaluate_remaining_cluster_seeds()
    promotion = fm.promote_seed_solutions(targeted.id, seed_ids=[seeds[2]])

    assert promotion.promoted is True
    assert promotion.added_seed_ids == (seeds[2],)
    assert promotion.regressed_seed_ids == ()
    state = fm.state()
    assert state["current_success_seed_ids"] == sorted(seeds[:3])
    assert state["abandoned_seed_ids"] == sorted(seeds[3:])
    assert state["seed_solutions"][seeds[0]]["candidate_id"] == initial.id
    assert state["seed_solutions"][seeds[2]]["candidate_id"] == targeted.id
    assert fm.current_best_ref() == initial.id
    assert fm.current_best_ref(seeds[2]) == targeted.id
    coverage = json.loads((fm.root / "current_best" / "coverage.json").read_text())
    assert coverage["solved"] == 3
    assert coverage["seed_solution_map"][seeds[2]]["candidate_id"] == targeted.id
    assert Path(
        coverage["seed_solution_map"][seeds[2]]["program_path"]
    ).is_file()

    fm.reopen_abandoned(
        [seeds[3]],
        reason="reuse the portfolio program on another active seed",
    )
    reused = targeted.evaluate_targeted_seeds([seeds[3]])
    assert reused[0]["outcome"] == "policy_failure"
    assert seeds[3] in targeted.manifest()["target_seed_ids"]
    _record_failure_videos(targeted, [seeds[3]])
    fm.mark_abandoned(
        [seeds[3]],
        reason="targeted reuse did not solve this independently verified seed",
    )

    task.finish()


def test_v2_soft_budget_requires_agent_decision(repair_settings_v2: Path) -> None:
    campaign = _campaign(repair_settings_v2)
    task_key = next(iter(campaign.tasks))
    task = _open_task(campaign, task_key)
    task.ensure_runtime()
    state = json.loads(task.task_state_path.read_text(encoding="utf-8"))
    state["budget"]["started_at"] = (
        datetime.now(UTC) - timedelta(hours=5)
    ).isoformat()
    task.task_state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(SoftBudgetReviewRequired):
        task.require_budget_decision()
    review_state = json.loads(task.task_state_path.read_text(encoding="utf-8"))
    assert review_state["status"] == "budget_review_required"
    assert review_state["budget"]["review_required"] is True
    snapshot = task.extend_budget(2.0, reason="expected positive marginal coverage")
    assert snapshot["over_budget"] is False
    task.close(status="in_progress")


def test_v2_video_analysis_is_a_hard_gate_and_round_records_each_idea(
    repair_settings_v2: Path,
) -> None:
    campaign = _campaign(repair_settings_v2)
    task_key = next(iter(campaign.tasks))
    task = _open_task(campaign, task_key)
    fm = task.open_failure_mode("FM-01")
    seeds = [job["source_job_id"] for job in fm.jobs]

    initial = fm.create_initial_candidate(
        source=_program(seeds[0]),
        representative_seed_ids=seeds[:3],
        strategy_summary="video gate probe",
    )
    initial.evaluate_smoke()
    with pytest.raises(RuntimeError, match="wide/wrist video analysis"):
        initial.decide("stop", rationale="trace-only diagnosis must be rejected")

    _record_failure_videos(initial, [seeds[1]])
    with pytest.raises(RuntimeError, match="wide/wrist video analysis"):
        initial.decide("stop", rationale="one failed seed remains unanalyzed")
    _record_failure_videos(initial, [seeds[2]])
    initial.decide("stop", rationale="both failed seed videos show the same contact miss")

    analysis_path = Path(initial.manifest()["path"]) / "failure_video_analysis.json"
    recorded = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert set(recorded["analyses"]) == {seeds[1], seeds[2]}
    assert recorded["analyses"][seeds[1]]["video_files"]["wide"]["size_bytes"] > 0
    assert recorded["analyses"][seeds[1]]["video_files"]["wrist"]["size_bytes"] > 0

    round_one = fm.start_round(
        mechanism="visible shared contact miss",
        seed_ids=[seeds[1], seeds[2]],
        evidence_summary="wide and wrist videos show the same pre-contact offset",
    )
    first_idea = round_one.create_candidate(
        source=_program(seeds[0], seeds[1]),
        parent_ref=initial.id,
        change_summary="first contact-offset idea",
    )
    second_idea = round_one.create_candidate(
        source=_program(seeds[0], seeds[2]),
        parent_ref=initial.id,
        change_summary="second contact-offset idea",
    )
    assert first_idea.manifest()["idea_index"] == 1
    assert second_idea.manifest()["idea_index"] == 2
    failure_analysis = json.loads(
        (Path(round_one._record()["path"]) / "failure_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure_analysis["candidate_ids"] == [first_idea.id, second_idea.id]
    assert all(
        failure_analysis["evidence_refs"][seed]["video_analysis"]
        for seed in [seeds[1], seeds[2]]
    )
    task.close(status="in_progress")


def test_v2_reset_failure_is_excluded_from_mechanism_clusters(
    repair_settings_v2: Path,
) -> None:
    campaign = _campaign(repair_settings_v2)
    task_key = next(iter(campaign.tasks))
    task = _open_task(campaign, task_key)
    fm = task.open_failure_mode("FM-01")
    seeds = [job["source_job_id"] for job in fm.jobs]
    with pytest.raises(ValueError, match="exactly 3"):
        fm.create_initial_candidate(
            source="RESULT = True",
            representative_seed_ids=seeds[:2],
            strategy_summary="invalid initial smoke set",
        )
    source = repr(
        {
            seeds[0]: "success",
            seeds[1]: "policy_failure",
            seeds[2]: "reset_failure",
        }
    )
    initial = fm.create_initial_candidate(
        source=f"FAKE_OUTCOMES = {source}",
        representative_seed_ids=seeds[:3],
        strategy_summary="classification probe",
    )
    initial.evaluate_smoke()
    reset_result = initial.manifest()["results"][seeds[2]]
    reset_path = Path(reset_result["attempt_path"])
    assert "diagnostics" in reset_path.parts
    assert not (reset_path.parents[1] / "result.json").exists()
    with pytest.raises(ValueError, match="policy-failure evidence"):
        fm.start_round(
            mechanism="not a policy mechanism",
            seed_ids=[seeds[2]],
            evidence_summary="reset verification failed",
        )
    task.close(status="in_progress")


def test_v2_has_one_canonical_result_and_reference_only_indexes(
    repair_settings_v2: Path,
) -> None:
    campaign = _campaign(repair_settings_v2)
    task_key = next(iter(campaign.tasks))
    task = _open_task(campaign, task_key)
    fm = task.open_failure_mode("FM-01")
    seeds = [job["source_job_id"] for job in fm.jobs]
    candidate = fm.create_initial_candidate(
        source=_program(seeds[0]),
        representative_seed_ids=seeds[:3],
        strategy_summary="canonical result probe",
    )

    first = candidate.evaluate_smoke()
    second = candidate.evaluate_smoke()
    assert all(result.get("reused_candidate_result") for result in second)
    candidate_root = Path(candidate.manifest()["path"])
    assert len(list((candidate_root / "evaluations").glob("eval_*"))) == 1
    assert not list(candidate_root.rglob("rerun_*"))

    raw_state = json.loads(fm.state_path.read_text(encoding="utf-8"))
    raw_entry = raw_state["candidates"][candidate.id]
    assert set(raw_entry) == {"manifest_ref"}
    raw_manifest = json.loads(Path(raw_entry["manifest_ref"]).read_text(encoding="utf-8"))
    assert all(set(reference) == {"result_ref"} for reference in raw_manifest["results"].values())
    summary = json.loads(
        next((candidate_root / "evaluations").glob("eval_*/summary.json")).read_text(
            encoding="utf-8"
        )
    )
    assert "results" not in summary
    assert set(summary["result_refs"]) == set(seeds[:3])

    for result in first:
        attempt = Path(result["attempt_path"])
        assert (attempt / "result.json").is_file()
        assert not (attempt / "worker_request.json").exists()
        assert not (attempt / "worker_result.json").exists()
        assert not (attempt / "fake_backend.json").exists()

    _record_failure_videos(candidate)
    candidate.decide("stop", rationale="prepare a distinct-code deduplication probe")
    round_one = fm.start_round(
        mechanism="same observed behavior",
        seed_ids=seeds[1:3],
        evidence_summary="probe identical evidence emitted by distinct code",
    )
    with pytest.raises(ValueError, match="identical program already exists"):
        round_one.create_candidate(
            source=_program(seeds[0]),
            parent_ref=candidate.id,
            change_summary="must not duplicate the same candidate program",
        )
    equivalent = round_one.create_candidate(
        source=_program(seeds[0]) + "\nDUMMY_CHANGE = 1",
        parent_ref=candidate.id,
        change_summary="syntactically distinct but behaviorally equivalent probe",
    )
    equivalent_results = equivalent.evaluate_smoke(seed_ids=seeds[1:3])
    first_trajectory = Path(first[1]["attempt_path"]) / "trajectory.json"
    second_trajectory = Path(equivalent_results[0]["attempt_path"]) / "trajectory.json"
    assert first_trajectory.stat().st_ino == second_trajectory.stat().st_ino
    assert first_trajectory.stat().st_nlink >= 3  # blob anchor plus two readable paths
    assert fm.state()["exploration_review"]["required"] is True
    assert any(
        reason["kind"] == "smoke_behavior_matches_parent"
        for reason in fm.state()["exploration_review"]["reasons"]
    )
    task.close(status="in_progress")


def test_v2_low_value_review_is_soft_and_reasoned(
    repair_settings_v2: Path,
) -> None:
    value = yaml.safe_load(repair_settings_v2.read_text(encoding="utf-8"))
    value["campaign"]["output_dir"] = str(
        repair_settings_v2.parent / "repair_campaign_v2_soft_review"
    )
    value["repair"]["exploration_review"] = {
        "consecutive_no_gain_candidates": 1,
        "per_seed_policy_attempts": 2,
    }
    settings = repair_settings_v2.parent / "repair_v2_soft_review.yaml"
    settings.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    campaign = _campaign(settings)
    task_key = next(iter(campaign.tasks))
    task = _open_task(campaign, task_key)
    fm = task.open_failure_mode("FM-01")
    seeds = [job["source_job_id"] for job in fm.jobs]
    initial = fm.create_initial_candidate(
        source=_program(seeds[0]),
        representative_seed_ids=seeds[:3],
        strategy_summary="soft review probe",
    )
    initial.evaluate_smoke()
    _record_failure_videos(initial)
    initial.decide("stop", rationale="no useful smoke gain")

    with pytest.raises(ExplorationReviewRequired):
        fm.start_round(
            mechanism="new visible mechanism",
            seed_ids=seeds[1:3],
            evidence_summary="new video-grounded hypothesis",
        )
    review = fm.state()["exploration_review"]
    assert review["required"] is True
    assert review["reasons"][0]["kind"] == "consecutive_candidates_without_coverage_gain"

    fm.continue_exploration(reason="wrist video identifies a different contact offset")
    round_one = fm.start_round(
        mechanism="new visible mechanism",
        seed_ids=seeds[1:3],
        evidence_summary="new video-grounded hypothesis",
    )
    assert round_one.id.startswith("round_0002_")
    assert fm.state()["exploration_review"]["required"] is False
    followup = round_one.create_candidate(
        source=_program(seeds[0], seeds[1]) + "\nSECOND_HYPOTHESIS = 1",
        parent_ref=initial.id,
        change_summary="second policy attempt on the active seeds",
    )
    followup.evaluate_smoke(seed_ids=seeds[1:3])
    third_attempt = round_one.create_candidate(
        source=_program(seeds[0], seeds[1]) + "\nTHIRD_HYPOTHESIS = 1",
        parent_ref=initial.id,
        change_summary="third policy attempt on the active seeds",
    )
    third_attempt.evaluate_smoke(seed_ids=seeds[1:3])
    assert any(
        reason["kind"] == "repeated_policy_attempts_on_seed"
        for reason in fm.state()["exploration_review"]["reasons"]
    )
    task.close(status="in_progress")


def test_v2_one_task_uses_four_gpu_local_runtimes(
    repair_settings_v2: Path,
) -> None:
    value = yaml.safe_load(repair_settings_v2.read_text(encoding="utf-8"))
    value["campaign"]["output_dir"] = str(
        repair_settings_v2.parent / "repair_campaign_v2_four_gpu"
    )
    value["resources"].update(
        {
            "gpus": [0, 1, 2, 3],
            "gpus_per_task": 4,
            "workers_per_gpu": 1,
        }
    )
    value["resources"]["services"] = {"profile": "default", "manage": False}
    value["repair"]["smoke"] = {"min_seeds": 5, "max_seeds": 5}
    settings = repair_settings_v2.parent / "repair_v2_four_gpu.yaml"
    settings.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    campaign = _campaign(settings)
    task_key = next(iter(campaign.tasks))
    with pytest.raises(ValueError, match="requires 4 GPUs"):
        campaign.open_task(task_key, gpu_id=0)
    task = campaign.open_task(task_key, gpu_ids=[0, 1, 2, 3])
    fm = task.open_failure_mode("FM-01")
    seeds = [job["source_job_id"] for job in fm.jobs]
    initial = fm.create_initial_candidate(
        source=_program(*seeds),
        representative_seed_ids=seeds,
        strategy_summary="four GPU dispatch probe",
    )
    results = initial.evaluate_smoke()

    assert len(task._runtimes) == 4
    assert {runtime.gpu_id for runtime in task._runtimes} == {0, 1, 2, 3}
    assert {int(result["gpu_id"]) for result in results} == {0, 1, 2, 3}
    assert len(results) == 5  # one freed GPU-local worker pulled the fifth seed
    assert [
        runtime.service_manager.endpoints()["sam3"] for runtime in task._runtimes
    ] == [
        "http://127.0.0.1:14014",
        "http://127.0.0.1:14114",
        "http://127.0.0.1:14214",
        "http://127.0.0.1:14314",
    ]
    status = json.loads(task.status_path.read_text(encoding="utf-8"))
    assert status["gpu_ids"] == [0, 1, 2, 3]
    assert status["workers_per_gpu"] == 1
    task.close(status="in_progress")
