"""Persistent v2 failure-mode repair state machine."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from .artifacts import ProgramRecord, ReadableAttemptStore
from .util import (
    append_jsonl,
    atomic_hardlink,
    atomic_write_json,
    atomic_write_text,
    locked_file,
    read_json,
    readable_slug,
    safe_component,
    sha256_text,
    utc_now,
)

if TYPE_CHECKING:
    from .campaign import TaskSession


POLICY_OUTCOMES = {"success", "policy_failure"}
EXCLUDED_OUTCOMES = {"policy_invalid", "reset_failure", "infrastructure_failure"}
VIDEO_ANALYSIS_FIELDS = (
    "wide_view",
    "wrist_view",
    "failure_moment",
    "mechanism_evidence",
)


class SoftBudgetReviewRequired(RuntimeError):
    """A task crossed its soft wall-clock budget and needs an agent decision."""


class ExplorationReviewRequired(RuntimeError):
    """Recent FM exploration has low marginal value and needs an agent decision."""


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    candidate_id: str
    previous_successes: int
    candidate_successes: int
    added_seed_ids: tuple[str, ...]
    regressed_seed_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "candidate_id": self.candidate_id,
            "previous_successes": self.previous_successes,
            "candidate_successes": self.candidate_successes,
            "net_gain": self.candidate_successes - self.previous_successes,
            "added_seed_ids": list(self.added_seed_ids),
            "regressed_seed_ids": list(self.regressed_seed_ids),
        }


class FailureModeSession:
    """One task/FM repair state, including rounds, candidates, and coverage."""

    def __init__(self, task: TaskSession, mode_id: str):
        if int(task.campaign.settings["schema_version"]) < 2:
            raise RuntimeError("failure-mode state machine requires a v2 repair campaign")
        self.task = task
        self.mode_id = str(mode_id)
        self.jobs = [
            dict(job) for job in task.jobs(mode_ids=[self.mode_id])
        ]
        if not self.jobs:
            raise KeyError(f"unknown failure mode for task {task.task_key}: {self.mode_id}")
        self.root = task.task_root / "failure_modes" / safe_component(self.mode_id)
        self.state_path = self.root / "state.json"
        self.lock_path = self.root / "state.lock"
        self.progress_path = task.task_root / "progress.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def _smoke_min(self) -> int:
        return int(self.task.campaign.settings["repair"]["smoke"]["min_seeds"])

    @property
    def _smoke_max(self) -> int:
        return int(self.task.campaign.settings["repair"]["smoke"]["max_seeds"])

    @property
    def _no_gain_review_threshold(self) -> int:
        return int(
            self.task.campaign.settings["repair"]["exploration_review"][
                "consecutive_no_gain_candidates"
            ]
        )

    @property
    def _seed_attempt_review_threshold(self) -> int:
        return int(
            self.task.campaign.settings["repair"]["exploration_review"][
                "per_seed_policy_attempts"
            ]
        )

    @staticmethod
    def _review_state(state: dict[str, Any]) -> dict[str, Any]:
        seeds = [str(seed) for seed in state.get("seed_ids", [])]
        review = state.setdefault(
            "exploration_review",
            {
                "required": False,
                "reasons": [],
                "consecutive_no_gain_candidates": 0,
                "policy_attempt_counts": {},
                "acknowledged_attempt_counts": {},
                "decisions": [],
            },
        )
        for key in ("policy_attempt_counts", "acknowledged_attempt_counts"):
            counts = review.setdefault(key, {})
            for seed in seeds:
                counts.setdefault(seed, 0)
        return review

    @staticmethod
    def _add_review_reason(review: dict[str, Any], reason: dict[str, Any]) -> None:
        signature = (reason.get("kind"), tuple(reason.get("seed_ids", [])))
        existing = {
            (item.get("kind"), tuple(item.get("seed_ids", [])))
            for item in review.get("reasons", [])
        }
        if signature not in existing:
            review.setdefault("reasons", []).append(reason)
        review["required"] = True

    def require_exploration_decision(self) -> None:
        with locked_file(self.lock_path):
            state = read_json(self.state_path)
            review = self._review_state(state)
            required = bool(review.get("required"))
            reasons = copy.deepcopy(review.get("reasons", []))
        if required:
            raise ExplorationReviewRequired(
                "FM exploration needs a marginal-value review before more rollout work; "
                f"reasons={reasons}. Call continue_exploration(reason=...) or abandon "
                "evidence-backed long-tail seeds."
            )

    def continue_exploration(self, *, reason: str) -> dict[str, Any]:
        """Acknowledge a soft low-value review and authorize further exploration."""

        if not reason.strip():
            raise ValueError("exploration continuation requires a non-empty reason")
        with locked_file(self.lock_path):
            state = read_json(self.state_path)
            review = self._review_state(state)
            if not review.get("required"):
                raise RuntimeError("no exploration review is currently required")
            review.setdefault("decisions", []).append(
                {
                    "decision": "continue",
                    "reason": reason,
                    "reviewed_reasons": copy.deepcopy(review.get("reasons", [])),
                    "decided_at": utc_now(),
                }
            )
            review["acknowledged_attempt_counts"] = copy.deepcopy(
                review["policy_attempt_counts"]
            )
            review["consecutive_no_gain_candidates"] = 0
            review["required"] = False
            review["reasons"] = []
            state["updated_at"] = utc_now()
            atomic_write_json(self.state_path, state)
            snapshot = copy.deepcopy(review)
        append_jsonl(
            self.progress_path,
            {
                "event": "exploration_review_decision",
                "failure_mode_id": self.mode_id,
                "decision": "continue",
                "reason": reason,
                "recorded_at": utc_now(),
            },
        )
        return snapshot

    def _record_no_gain_locked(
        self,
        state: dict[str, Any],
        *,
        candidate_id: str,
        disposition: str,
    ) -> None:
        review = self._review_state(state)
        review["consecutive_no_gain_candidates"] = (
            int(review.get("consecutive_no_gain_candidates", 0)) + 1
        )
        if review["consecutive_no_gain_candidates"] >= self._no_gain_review_threshold:
            self._add_review_reason(
                review,
                {
                    "kind": "consecutive_candidates_without_coverage_gain",
                    "candidate_id": candidate_id,
                    "disposition": disposition,
                    "count": review["consecutive_no_gain_candidates"],
                    "threshold": self._no_gain_review_threshold,
                },
            )

    def _record_policy_attempts_locked(
        self,
        state: dict[str, Any],
        results: list[dict[str, Any]],
        *,
        stage: str,
    ) -> None:
        if stage == "stability":
            return
        review = self._review_state(state)
        counts = review["policy_attempt_counts"]
        acknowledged = review["acknowledged_attempt_counts"]
        for result in results:
            if bool(result.get("cached")) or result.get("outcome") not in POLICY_OUTCOMES:
                continue
            seed = str(result["source_job_id"])
            counts[seed] = int(counts.get(seed, 0)) + 1
            delta = counts[seed] - int(acknowledged.get(seed, 0))
            if delta >= self._seed_attempt_review_threshold:
                self._add_review_reason(
                    review,
                    {
                        "kind": "repeated_policy_attempts_on_seed",
                        "seed_ids": [seed],
                        "attempts_since_review": delta,
                        "total_attempts": counts[seed],
                        "threshold": self._seed_attempt_review_threshold,
                    },
                )

    def _initialize(self) -> None:
        with locked_file(self.lock_path):
            if self.state_path.exists():
                return
            seeds = [str(job["source_job_id"]) for job in self.jobs]
            atomic_write_json(
                self.state_path,
                {
                    "schema_version": 2,
                    "task_key": self.task.task_key,
                    "task_slug": self.task.task_slug,
                    "failure_mode_id": self.mode_id,
                    "seed_ids": seeds,
                    "current_best": None,
                    "seed_solutions": {},
                    "current_success_seed_ids": [],
                    "active_failed_seed_ids": seeds,
                    "abandoned_seed_ids": [],
                    "historical_successes": {},
                    "latest_outcomes": {},
                    "rounds": [],
                    "candidates": {},
                    "next_round": 1,
                    "next_candidate": 1,
                    "next_evaluation": 1,
                    "exploration_review": {
                        "required": False,
                        "reasons": [],
                        "consecutive_no_gain_candidates": 0,
                        "policy_attempt_counts": {seed: 0 for seed in seeds},
                        "acknowledged_attempt_counts": {seed: 0 for seed in seeds},
                        "decisions": [],
                    },
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                },
            )

    def state(self) -> dict[str, Any]:
        with locked_file(self.lock_path):
            state = read_json(self.state_path)
        state["candidates"] = {
            candidate_id: self._hydrate_candidate_entry(entry)
            for candidate_id, entry in state.get("candidates", {}).items()
        }
        return state

    @staticmethod
    def _result_reference(result: dict[str, Any]) -> dict[str, str]:
        if "result_ref" in result:
            return {"result_ref": str(result["result_ref"])}
        attempt = Path(str(result.get("attempt_path", "")))
        result_path = attempt / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"authoritative result is missing: {result_path}")
        return {"result_ref": str(result_path.resolve())}

    @staticmethod
    def _hydrate_result(reference: dict[str, Any]) -> dict[str, Any]:
        if "result_ref" not in reference:
            return dict(reference)
        result_path = Path(str(reference["result_ref"]))
        result = read_json(result_path)
        result["attempt_path"] = str(result_path.parent)
        result["result_ref"] = str(result_path)
        result["canonical"] = (
            "reruns" not in result_path.parts and "diagnostics" not in result_path.parts
        )
        return result

    @classmethod
    def _dehydrate_candidate(cls, candidate: dict[str, Any]) -> dict[str, Any]:
        record = copy.deepcopy(candidate)
        record["results"] = {
            seed: cls._result_reference(result)
            for seed, result in record.get("results", {}).items()
        }
        record["rerun_results"] = [
            cls._result_reference(result) for result in record.get("rerun_results", [])
        ]
        record.pop("failure_video_analyses", None)
        return record

    @classmethod
    def _hydrate_candidate_entry(cls, entry: dict[str, Any]) -> dict[str, Any]:
        if "manifest_ref" in entry:
            candidate = read_json(Path(str(entry["manifest_ref"])))
        else:
            candidate = copy.deepcopy(entry)
        candidate["results"] = {
            seed: cls._hydrate_result(result)
            for seed, result in candidate.get("results", {}).items()
        }
        candidate["rerun_results"] = [
            cls._hydrate_result(result)
            for result in candidate.get("rerun_results", [])
        ]
        analysis_ref = candidate.get("failure_video_analysis_ref")
        if analysis_ref and Path(str(analysis_ref)).is_file():
            candidate["failure_video_analyses"] = read_json(
                Path(str(analysis_ref))
            ).get("analyses", {})
        return candidate

    @classmethod
    def _candidate_from_raw_state(
        cls, state: dict[str, Any], candidate_id: str
    ) -> dict[str, Any]:
        try:
            return cls._hydrate_candidate_entry(state["candidates"][candidate_id])
        except KeyError as exc:
            raise KeyError(f"unknown candidate: {candidate_id}") from exc

    def _normalize_seed_ids(self, values: Iterable[str]) -> list[str]:
        aliases: dict[str, str] = {}
        for job in self.jobs:
            source = str(job["source_job_id"])
            aliases[source] = source
            aliases[str(job["job_id"])] = source
            aliases[str(job.get("seed_slug", source))] = source
        normalized: list[str] = []
        missing: list[str] = []
        for value in values:
            key = str(value)
            if key not in aliases:
                missing.append(key)
            elif aliases[key] not in normalized:
                normalized.append(aliases[key])
        if missing:
            raise KeyError(f"seed IDs are not part of {self.mode_id}: {sorted(missing)}")
        return normalized

    def _job(self, seed_id: str) -> dict[str, Any]:
        for job in self.jobs:
            if str(job["source_job_id"]) == seed_id:
                return dict(job)
        raise KeyError(seed_id)

    def _seed_history_path(self, seed_id: str) -> Path:
        job = self._job(seed_id)
        slug = str(job.get("seed_slug") or seed_id)
        return self.root / "seeds" / safe_component(slug) / "history.jsonl"

    def _candidate(self, candidate_id: str) -> dict[str, Any]:
        with locked_file(self.lock_path):
            return self._candidate_from_raw_state(read_json(self.state_path), candidate_id)

    @staticmethod
    def _write_candidate_manifest(candidate: dict[str, Any]) -> None:
        atomic_write_json(
            Path(str(candidate["path"])) / "manifest.json",
            FailureModeSession._dehydrate_candidate(candidate),
        )

    def _candidate_program(self, candidate: dict[str, Any]) -> ProgramRecord:
        path = Path(str(candidate["path"])) / "program.py"
        if sha256_text(path.read_text(encoding="utf-8")) != candidate["program_sha256"]:
            raise RuntimeError(f"candidate program changed after creation: {path}")
        return ProgramRecord(
            id=str(candidate["id"]),
            sha256=str(candidate["program_sha256"]),
            path=path,
            task_key=self.task.task_key,
            mode_ids=(self.mode_id,),
            derived_from=(str(candidate["parent_ref"]),)
            if candidate.get("parent_ref")
            else (),
            created_at=str(candidate["created_at"]),
        )

    def _validate_parent_reference(self, reference: str) -> None:
        state = self.state()
        if reference in state.get("candidates", {}):
            candidate = state["candidates"][reference]
            if not any(
                bool(result.get("success"))
                for result in candidate.get("results", {}).values()
            ):
                analyzed_failures = {
                    seed
                    for seed, analysis in candidate.get(
                        "failure_video_analyses", {}
                    ).items()
                    if analysis.get("result_ref")
                    == candidate.get("results", {}).get(seed, {}).get("result_ref")
                    and candidate.get("results", {}).get(seed, {}).get("outcome")
                    == "policy_failure"
                }
                if not analyzed_failures:
                    raise ValueError(
                        "candidate parent has neither verified success nor a current "
                        f"video-analyzed policy result: {reference}"
                    )
            return
        try:
            self.task.campaign.experience.get(reference)
            return
        except KeyError:
            pass
        raise KeyError(reference)

    def _create_round(
        self,
        *,
        mechanism: str,
        seed_ids: Iterable[str],
        evidence_summary: str,
        initial: bool,
    ) -> RepairRound:
        self.require_exploration_decision()
        seeds = self._normalize_seed_ids(seed_ids)
        if not seeds:
            raise ValueError("a repair round must target at least one seed")
        mechanism_slug = readable_slug(mechanism, fallback="mechanism")
        with locked_file(self.lock_path):
            state = read_json(self.state_path)
            if not initial:
                active = set(state["active_failed_seed_ids"])
                non_active = sorted(set(seeds) - active)
                if non_active:
                    raise ValueError(f"mechanism cluster contains non-active seeds: {non_active}")
                invalid: list[str] = []
                missing_video_analysis: list[str] = []
                latest_evidence: dict[
                    str, tuple[dict[str, Any], dict[str, Any] | None]
                ] = {}
                for seed in seeds:
                    try:
                        latest_evidence[seed] = (
                            self._latest_policy_failure_with_video_analysis(seed)
                        )
                    except ValueError:
                        invalid.append(seed)
                        continue
                    if latest_evidence[seed][1] is None:
                        missing_video_analysis.append(seed)
                if invalid:
                    raise ValueError(
                        "mechanism clusters require policy-failure evidence: "
                        + ", ".join(invalid)
                    )
                if missing_video_analysis:
                    raise ValueError(
                        "mechanism clusters require recorded wide/wrist video analysis for "
                        "the latest policy failure: "
                        + ", ".join(missing_video_analysis)
                    )
            index = int(state["next_round"])
            round_id = f"round_{index:04d}_{mechanism_slug}"
            path = self.root / "rounds" / round_id
            evidence_refs: dict[str, dict[str, Any]] = {}
            if not initial:
                for seed in seeds:
                    latest, video_analysis = latest_evidence[seed]
                    attempt = Path(str(latest.get("attempt_path", "")))
                    evidence_refs[seed] = {
                        "candidate_id": latest.get("candidate_id"),
                        "evaluation_id": latest.get("evaluation_id"),
                        "attempt_path": str(attempt),
                        "wide_video": str(attempt / "wide.mp4"),
                        "wrist_video": str(attempt / "wrist.mp4"),
                        "trajectory": str(attempt / "trajectory.json"),
                        "evaluator_result": str(attempt / "result.json"),
                        "video_analysis": video_analysis,
                    }
            record = {
                "id": round_id,
                "mechanism": mechanism,
                "mechanism_slug": mechanism_slug,
                "seed_ids": seeds,
                "evidence_summary": evidence_summary,
                "evidence_refs": evidence_refs,
                "initial": initial,
                "path": str(path),
                "candidate_ids": [],
                "created_at": utc_now(),
            }
            path.mkdir(parents=True, exist_ok=False)
            atomic_write_json(path / "failure_analysis.json", record)
            state["rounds"].append(record)
            state["next_round"] = index + 1
            state["updated_at"] = utc_now()
            atomic_write_json(self.state_path, state)
        append_jsonl(
            self.progress_path,
            {
                "event": "round_started",
                "failure_mode_id": self.mode_id,
                "round_id": round_id,
                "mechanism": mechanism,
                "seed_ids": seeds,
                "initial": initial,
                "recorded_at": utc_now(),
            },
        )
        return RepairRound(self, round_id)

    def _history_entries(self, seed_id: str) -> list[dict[str, Any]]:
        path = self._seed_history_path(seed_id)
        if not path.is_file():
            return []
        import json

        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def _latest_policy_failure_with_video_analysis(
        self, seed_id: str
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        history = self._history_entries(seed_id)
        failures: list[dict[str, Any]] = []
        for item in history:
            if item.get("event") != "evaluation_result" or not item.get("result_ref"):
                continue
            result = self._hydrate_result({"result_ref": item["result_ref"]})
            if result.get("outcome") == "policy_failure":
                failures.append({**item, **result})
        if not failures:
            raise ValueError(f"seed has no policy-failure evidence: {seed_id}")
        latest = failures[-1]
        analyses = [
            item
            for item in history
            if item.get("event") == "failure_video_analysis"
            and item.get("candidate_id") == latest.get("candidate_id")
            and item.get("result_ref") == latest.get("result_ref")
        ]
        if not analyses:
            return latest, None
        analysis_event = analyses[-1]
        analysis_document = read_json(Path(str(analysis_event["analysis_ref"])))
        analysis = analysis_document.get("analyses", {}).get(seed_id)
        return latest, dict(analysis) if analysis else None

    def create_initial_candidate(
        self,
        *,
        source: str,
        representative_seed_ids: Iterable[str],
        strategy_summary: str,
    ) -> RepairCandidate:
        state = self.state()
        if state.get("current_best"):
            raise RuntimeError(
                "an initial candidate cannot be created after an FM best has been promoted"
            )
        all_seeds = list(state["seed_ids"])
        representative = self._normalize_seed_ids(representative_seed_ids)
        required = min(self._smoke_min, len(all_seeds))
        if len(representative) != required:
            raise ValueError(f"initial candidate smoke test requires exactly {required} seeds")
        round_session = self._create_round(
            mechanism="initial_strategy",
            seed_ids=all_seeds,
            evidence_summary=strategy_summary,
            initial=True,
        )
        return round_session._create_candidate(
            source=source,
            parent_ref="",
            change_summary=strategy_summary,
            representative_seed_ids=representative,
        )

    def start_round(
        self,
        *,
        mechanism: str,
        seed_ids: Iterable[str],
        evidence_summary: str,
    ) -> RepairRound:
        return self._create_round(
            mechanism=mechanism,
            seed_ids=seed_ids,
            evidence_summary=evidence_summary,
            initial=False,
        )

    def current_best_ref(self, seed_id: str | None = None) -> str | None:
        state = self.state()
        if seed_id is not None:
            seed = self._normalize_seed_ids([seed_id])[0]
            solution = state.get("seed_solutions", {}).get(seed)
            if solution:
                return str(solution["candidate_id"])
        current = state.get("current_best")
        return str(current["candidate_id"]) if current else None

    def _seed_solutions_locked(self, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Return the seed portfolio, lazily upgrading pre-portfolio v2 state."""

        solutions = state.setdefault("seed_solutions", {})
        current = state.get("current_best")
        if not current:
            return solutions
        candidate_id = str(current["candidate_id"])
        candidate = self._candidate_from_raw_state(state, candidate_id)
        for seed in state.get("current_success_seed_ids", []):
            if seed in solutions:
                continue
            result = dict(candidate.get("results", {}).get(seed) or {})
            if not bool(result.get("success")):
                continue
            solutions[seed] = {
                "candidate_id": candidate_id,
                "program_sha256": str(candidate["program_sha256"]),
                "result_ref": self._result_reference(result)["result_ref"],
                "verified_at": str(result.get("finished_at", current.get("promoted_at", utc_now()))),
            }
        return solutions

    def candidate(self, candidate_id: str) -> RepairCandidate:
        self._candidate(candidate_id)
        return RepairCandidate(self, candidate_id)

    def _evaluate_candidate(
        self,
        candidate_id: str,
        *,
        stage: str,
        seed_ids: list[str],
        execution_mode: str = "canonical",
        update_canonical_results: bool = True,
    ) -> list[dict[str, Any]]:
        self.task.require_budget_decision()
        if execution_mode != "diagnostic_retry":
            self.require_exploration_decision()
        candidate = self._candidate(candidate_id)
        if execution_mode == "canonical" and update_canonical_results:
            existing = candidate.get("results", {})
            missing = [seed for seed in seed_ids if seed not in existing]
            if not missing:
                return [
                    {
                        **dict(existing[seed]),
                        "cached": True,
                        "reused_candidate_result": True,
                    }
                    for seed in seed_ids
                ]
            seed_ids = missing
        program = self._candidate_program(candidate)
        jobs = [self._job(seed) for seed in seed_ids]
        with locked_file(self.lock_path):
            state = read_json(self.state_path)
            evaluation_index = int(state["next_evaluation"])
            state["next_evaluation"] = evaluation_index + 1
            state["updated_at"] = utc_now()
            atomic_write_json(self.state_path, state)
        evaluation_id = f"eval_{evaluation_index:04d}_{readable_slug(stage)}"
        evaluation_root = Path(str(candidate["path"])) / "evaluations" / evaluation_id
        evaluation_root.mkdir(parents=True, exist_ok=False)
        results = self.task._evaluate_program(
            program, jobs, execution_mode=execution_mode
        )
        result_refs = {
            str(result["source_job_id"]): self._result_reference(result)
            for result in results
        }
        summary = {
            "schema_version": 2,
            "evaluation_id": evaluation_id,
            "candidate_id": candidate_id,
            "stage": stage,
            "execution_mode": execution_mode,
            "updates_canonical_results": update_canonical_results,
            "seed_ids": seed_ids,
            "success_seed_ids": [
                str(result["source_job_id"])
                for result in results
                if bool(result.get("success"))
            ],
            "result_refs": result_refs,
            "cache_hit_seed_ids": sorted(
                str(result["source_job_id"])
                for result in results
                if bool(result.get("cached"))
            ),
            "finished_at": utc_now(),
        }
        atomic_write_json(evaluation_root / "summary.json", summary)
        with locked_file(self.lock_path):
            state = read_json(self.state_path)
            record = self._candidate_from_raw_state(state, candidate_id)
            record.setdefault("evaluations", []).append(
                {
                    "id": evaluation_id,
                    "stage": stage,
                    "execution_mode": execution_mode,
                    "path": str(evaluation_root),
                    "seed_ids": seed_ids,
                }
            )
            result_map = record.setdefault("results", {})
            rerun_results = record.setdefault("rerun_results", [])
            for result in results:
                seed = str(result["source_job_id"])
                result_ref = self._result_reference(result)
                if update_canonical_results:
                    result_map[seed] = result
                    state["latest_outcomes"][seed] = str(result.get("outcome", ""))
                else:
                    rerun_results.append(result)
                if update_canonical_results and result.get("success"):
                    state["historical_successes"][seed] = {
                        "candidate_id": candidate_id,
                        "program_sha256": candidate["program_sha256"],
                        "result_ref": result_ref["result_ref"],
                        "verified_at": result.get("finished_at", utc_now()),
                    }
                append_jsonl(
                    self._seed_history_path(seed),
                    {
                        "event": "evaluation_result",
                        "candidate_id": candidate_id,
                        "program_sha256": candidate["program_sha256"],
                        "evaluation_id": evaluation_id,
                        "stage": stage,
                        "stability_rerun": not update_canonical_results,
                        "result_ref": result_ref["result_ref"],
                        "cache_hit": bool(result.get("cached")),
                        "recorded_at": utc_now(),
                    },
                )
            if stage == "smoke":
                parent_ref = str(record.get("parent_ref", ""))
                if parent_ref in state.get("candidates", {}):
                    parent = self._candidate_from_raw_state(state, parent_ref)
                    comparable: list[str] = []
                    equivalent: list[str] = []
                    for result in results:
                        seed = str(result["source_job_id"])
                        parent_result = parent.get("results", {}).get(seed)
                        if not parent_result:
                            continue
                        comparable.append(seed)
                        current_trace = result.get("evidence_sha256", {}).get(
                            "trajectory.json"
                        )
                        parent_trace = parent_result.get("evidence_sha256", {}).get(
                            "trajectory.json"
                        )
                        if (
                            current_trace
                            and current_trace == parent_trace
                            and result.get("outcome") == parent_result.get("outcome")
                        ):
                            equivalent.append(seed)
                    if comparable and set(equivalent) == set(comparable):
                        record["behaviorally_equivalent_to_parent_seed_ids"] = sorted(
                            equivalent
                        )
                        self._add_review_reason(
                            self._review_state(state),
                            {
                                "kind": "smoke_behavior_matches_parent",
                                "candidate_id": candidate_id,
                                "parent_ref": parent_ref,
                                "seed_ids": sorted(equivalent),
                            },
                        )
            self._record_policy_attempts_locked(state, results, stage=stage)
            record["updated_at"] = utc_now()
            self._write_candidate_manifest(record)
            state["updated_at"] = utc_now()
            atomic_write_json(self.state_path, state)
        append_jsonl(
            self.progress_path,
            {
                "event": "evaluation",
                "failure_mode_id": self.mode_id,
                "candidate_id": candidate_id,
                "stage": stage,
                "attempt_count": len(results),
                "success_count": sum(bool(result.get("success")) for result in results),
                "elapsed_hours": self.task.budget_snapshot()["elapsed_hours"],
                "recorded_at": utc_now(),
            },
        )
        return results

    def promote_best(
        self,
        candidate_id: str,
        *,
        skills: dict[str, str] | None = None,
        description: str = "",
    ) -> PromotionResult:
        snapshot = self.state()
        candidate_snapshot = dict(snapshot["candidates"].get(candidate_id) or {})
        if not candidate_snapshot:
            raise KeyError(candidate_id)
        old_success_snapshot = set(snapshot.get("current_success_seed_ids", []))
        missing_old_successes = sorted(
            old_success_snapshot - set(candidate_snapshot.get("results", {}))
        )
        if missing_old_successes:
            self._evaluate_candidate(
                candidate_id,
                stage="promotion",
                seed_ids=missing_old_successes,
            )
        with locked_file(self.lock_path):
            state = read_json(self.state_path)
            candidate = self._candidate_from_raw_state(state, candidate_id)
            if candidate.get("status") not in {"expanded", "promoted"}:
                raise RuntimeError("candidate must finish expanded evaluation before promotion")
            old_best = state.get("current_best")
            old_successes = set(state.get("current_success_seed_ids", []))
            comparison = old_successes | set(candidate["target_seed_ids"])
            results = dict(candidate.get("results", {}))
            missing = sorted(seed for seed in comparison if seed not in results)
            if missing:
                raise RuntimeError(f"candidate lacks promotion evidence for seeds: {missing}")
            excluded = sorted(
                seed
                for seed in comparison
                if str(results[seed].get("outcome")) in EXCLUDED_OUTCOMES
            )
            if excluded:
                raise RuntimeError(
                    "candidate has unresolved reset/infra/invalid outcomes: " + ", ".join(excluded)
                )
            success_set = {
                seed for seed, result in results.items() if bool(result.get("success"))
            }
            added = tuple(sorted(success_set - old_successes))
            regressed = tuple(sorted(old_successes - success_set))
            promotion = PromotionResult(
                promoted=len(success_set) > len(old_successes),
                candidate_id=candidate_id,
                previous_successes=len(old_successes),
                candidate_successes=len(success_set),
                added_seed_ids=added,
                regressed_seed_ids=regressed,
            )
            if not promotion.promoted:
                candidate["status"] = "not_promoted"
                candidate["promotion"] = promotion.to_dict()
                self._write_candidate_manifest(candidate)
                self._record_no_gain_locked(
                    state,
                    candidate_id=candidate_id,
                    disposition="not_promoted",
                )
                state["updated_at"] = utc_now()
                atomic_write_json(self.state_path, state)
                self._compact_successes(candidate)
                return promotion

            program = self._candidate_program(candidate)
            best_root = self.root / "current_best"
            best_root.mkdir(parents=True, exist_ok=True)
            atomic_hardlink(program.path, best_root / "program.py")
            all_seeds = set(state["seed_ids"])
            abandoned = set(state["abandoned_seed_ids"]) - success_set
            active = all_seeds - success_set - abandoned
            best_manifest = {
                "schema_version": 2,
                "candidate_id": candidate_id,
                "program_sha256": candidate["program_sha256"],
                "parent_ref": candidate.get("parent_ref", ""),
                "target_mechanism": candidate["target_mechanism"],
                "description": description or candidate.get("change_summary", ""),
                "success_seed_ids": sorted(success_set),
                "failed_seed_ids": sorted(active),
                "abandoned_seed_ids": sorted(abandoned),
                "promotion": promotion.to_dict(),
                "promoted_at": utc_now(),
            }
            atomic_write_json(best_root / "manifest.json", best_manifest)
            atomic_write_json(
                best_root / "coverage.json",
                {
                    "total": len(all_seeds),
                    "solved": len(success_set),
                    "active_failed": len(active),
                    "abandoned": len(abandoned),
                    "success_seed_ids": sorted(success_set),
                    "active_failed_seed_ids": sorted(active),
                    "abandoned_seed_ids": sorted(abandoned),
                },
            )
            self.task.campaign.experience.promote_fm_experience(
                task_key=self.task.task_key,
                task_slug=self.task.task_slug,
                mode_id=self.mode_id,
                program_path=program.path,
                manifest=best_manifest,
                skills=skills,
            )
            if old_best and old_best.get("candidate_id") != candidate_id:
                previous_id = str(old_best["candidate_id"])
                if previous_id in state.get("candidates", {}):
                    previous = self._candidate_from_raw_state(state, previous_id)
                    previous["status"] = "superseded"
                    retained_paths = {
                        str(result.get("attempt_path"))
                        for result in candidate.get("results", {}).values()
                        if result.get("success") and result.get("attempt_path")
                    }
                    self._compact_successes(previous, preserve_paths=retained_paths)
                    self._write_candidate_manifest(previous)
            candidate["status"] = "promoted"
            candidate["promotion"] = promotion.to_dict()
            self._write_candidate_manifest(candidate)
            state["current_best"] = {
                "candidate_id": candidate_id,
                "program_sha256": candidate["program_sha256"],
                "promoted_at": utc_now(),
            }
            state["current_success_seed_ids"] = sorted(success_set)
            state["active_failed_seed_ids"] = sorted(active)
            state["abandoned_seed_ids"] = sorted(abandoned)
            state["latest_outcomes"] = {
                seed: str(result.get("outcome", "")) for seed, result in results.items()
            }
            self._review_state(state)["consecutive_no_gain_candidates"] = 0
            state["updated_at"] = utc_now()
            atomic_write_json(self.state_path, state)
        append_jsonl(
            self.progress_path,
            {
                "event": "promotion",
                "failure_mode_id": self.mode_id,
                **promotion.to_dict(),
                "elapsed_hours": self.task.budget_snapshot()["elapsed_hours"],
                "recorded_at": utc_now(),
            },
        )
        return promotion

    def promote_seed_solutions(
        self,
        candidate_id: str,
        *,
        seed_ids: Iterable[str] | None = None,
        description: str = "",
    ) -> PromotionResult:
        """Add evaluator-confirmed seed wins to a portfolio without regression checks.

        This is intentionally separate from :meth:`promote_best`: a targeted
        candidate only has to solve the declared seed(s). Existing seed
        solutions keep their own previously verified programs.
        """

        requested: list[str]
        with locked_file(self.lock_path):
            state = read_json(self.state_path)
            candidate = self._candidate_from_raw_state(state, candidate_id)
            if candidate.get("status") not in {
                "expanded",
                "promoted",
                "promoted_seed_specific",
                "stopped",
                "not_promoted",
            }:
                raise RuntimeError(
                    "candidate must have a completed evaluation decision before "
                    "seed-specific promotion"
                )
            requested = self._normalize_seed_ids(
                seed_ids if seed_ids is not None else candidate["target_seed_ids"]
            )
            outside_target = sorted(set(requested) - set(candidate["target_seed_ids"]))
            if outside_target:
                raise ValueError(
                    "seed-specific promotion may only use candidate target seeds: "
                    + ", ".join(outside_target)
                )
            results = dict(candidate.get("results", {}))
            missing = sorted(seed for seed in requested if seed not in results)
            if missing:
                raise RuntimeError(
                    "candidate lacks seed-specific promotion evidence for: "
                    + ", ".join(missing)
                )
            excluded = sorted(
                seed
                for seed in requested
                if str(results[seed].get("outcome")) in EXCLUDED_OUTCOMES
            )
            if excluded:
                raise RuntimeError(
                    "candidate has unresolved reset/infra/invalid outcomes: "
                    + ", ".join(excluded)
                )
            unsuccessful = sorted(seed for seed in requested if not bool(results[seed].get("success")))
            if unsuccessful:
                raise ValueError(
                    "only evaluator-confirmed successes may be promoted for a seed: "
                    + ", ".join(unsuccessful)
                )

            solutions = self._seed_solutions_locked(state)
            old_successes = set(solutions)
            added = tuple(sorted(set(requested) - old_successes))
            promotion = PromotionResult(
                promoted=bool(added),
                candidate_id=candidate_id,
                previous_successes=len(old_successes),
                candidate_successes=len(old_successes | set(requested)),
                added_seed_ids=added,
                regressed_seed_ids=(),
            )
            if not added:
                return promotion

            program = self._candidate_program(candidate)
            best_root = self.root / "current_best"
            programs_root = best_root / "programs"
            programs_root.mkdir(parents=True, exist_ok=True)
            portfolio_program = programs_root / f"{candidate_id}.py"
            atomic_hardlink(program.path, portfolio_program)
            verified_at = utc_now()
            for seed in added:
                result = results[seed]
                solutions[seed] = {
                    "candidate_id": candidate_id,
                    "program_sha256": str(candidate["program_sha256"]),
                    "program_path": str(portfolio_program),
                    "result_ref": self._result_reference(result)["result_ref"],
                    "verified_at": str(result.get("finished_at", verified_at)),
                }

            all_seeds = set(state["seed_ids"])
            success_set = set(solutions)
            abandoned = set(state["abandoned_seed_ids"]) - success_set
            active = all_seeds - success_set - abandoned
            coverage = {
                "total": len(all_seeds),
                "solved": len(success_set),
                "active_failed": len(active),
                "abandoned": len(abandoned),
                "success_seed_ids": sorted(success_set),
                "active_failed_seed_ids": sorted(active),
                "abandoned_seed_ids": sorted(abandoned),
                "seed_solution_map": copy.deepcopy(solutions),
            }
            atomic_write_json(best_root / "coverage.json", coverage)
            manifest_path = best_root / "manifest.json"
            best_manifest = read_json(manifest_path) if manifest_path.is_file() else {}
            best_manifest.update(
                {
                    "schema_version": 2,
                    "portfolio": True,
                    "description": description or best_manifest.get("description", ""),
                    "success_seed_ids": sorted(success_set),
                    "failed_seed_ids": sorted(active),
                    "abandoned_seed_ids": sorted(abandoned),
                    "seed_solution_map": copy.deepcopy(solutions),
                    "updated_at": verified_at,
                }
            )
            atomic_write_json(manifest_path, best_manifest)
            candidate["status"] = "promoted_seed_specific"
            candidate.setdefault("seed_specific_promotions", []).append(
                {
                    "seed_ids": list(added),
                    "description": description,
                    "promoted_at": verified_at,
                }
            )
            self._write_candidate_manifest(candidate)
            state["current_success_seed_ids"] = sorted(success_set)
            state["active_failed_seed_ids"] = sorted(active)
            state["abandoned_seed_ids"] = sorted(abandoned)
            for seed in added:
                state["latest_outcomes"][seed] = str(results[seed].get("outcome", ""))
            self._review_state(state)["consecutive_no_gain_candidates"] = 0
            state["updated_at"] = verified_at
            atomic_write_json(self.state_path, state)
            self.task.campaign.experience.update_fm_metadata(
                task_key=self.task.task_key,
                task_slug=self.task.task_slug,
                mode_id=self.mode_id,
                updates={
                    "portfolio": True,
                    "success_seed_ids": sorted(success_set),
                    "failed_seed_ids": sorted(active),
                    "abandoned_seed_ids": sorted(abandoned),
                    "seed_solution_map": copy.deepcopy(solutions),
                },
            )

        for seed in added:
            append_jsonl(
                self._seed_history_path(seed),
                {
                    "event": "seed_solution_promoted",
                    "candidate_id": candidate_id,
                    "program_sha256": candidate["program_sha256"],
                    "result_ref": self._result_reference(results[seed])["result_ref"],
                    "recorded_at": utc_now(),
                },
            )
        append_jsonl(
            self.progress_path,
            {
                "event": "seed_solution_promotion",
                "failure_mode_id": self.mode_id,
                **promotion.to_dict(),
                "elapsed_hours": self.task.budget_snapshot()["elapsed_hours"],
                "recorded_at": utc_now(),
            },
        )
        return promotion

    def _compact_successes(
        self,
        candidate: dict[str, Any],
        *,
        preserve_paths: set[str] | None = None,
    ) -> None:
        store = self.task.campaign.attempt_store
        if not isinstance(store, ReadableAttemptStore):
            return
        candidate_root = Path(str(candidate["path"])).resolve()
        preserved = set(preserve_paths or ())
        result_values = [
            *candidate.get("results", {}).values(),
            *candidate.get("rerun_results", []),
        ]
        for result in result_values:
            attempt_path = str(result.get("attempt_path", ""))
            if not result.get("success") or not attempt_path or attempt_path in preserved:
                continue
            attempt = Path(attempt_path).resolve()
            if attempt.is_relative_to(candidate_root):
                store.compact_success(attempt)

    def mark_abandoned(self, seed_ids: Iterable[str], *, reason: str) -> dict[str, Any]:
        if not bool(self.task.campaign.settings["repair"].get("allow_abandon", False)):
            raise RuntimeError("this campaign does not allow abandoning repair seeds")
        if not reason.strip():
            raise ValueError("abandon reason cannot be empty")
        seeds = self._normalize_seed_ids(seed_ids)
        with locked_file(self.lock_path):
            state = read_json(self.state_path)
            active = set(state["active_failed_seed_ids"])
            invalid = sorted(set(seeds) - active)
            if invalid:
                raise ValueError(f"only active failed seeds may be abandoned: {invalid}")
            no_attempt = [
                seed
                for seed in seeds
                if not any(
                    entry.get("event") == "evaluation_result"
                    and entry.get("result_ref")
                    and self._hydrate_result(
                        {"result_ref": entry["result_ref"]}
                    ).get("outcome")
                    in POLICY_OUTCOMES
                    for entry in self._history_entries(seed)
                )
            ]
            if no_attempt:
                raise ValueError(f"abandoned seeds require repair evidence: {no_attempt}")
            missing_video_analysis: list[str] = []
            for seed in seeds:
                try:
                    _, analysis = self._latest_policy_failure_with_video_analysis(seed)
                except ValueError:
                    analysis = None
                if analysis is None:
                    missing_video_analysis.append(seed)
            if missing_video_analysis:
                raise ValueError(
                    "abandonment requires recorded wide/wrist video analysis for the latest "
                    "policy failure: "
                    + ", ".join(missing_video_analysis)
                )
            abandoned = set(state["abandoned_seed_ids"]) | set(seeds)
            active -= set(seeds)
            state["abandoned_seed_ids"] = sorted(abandoned)
            state["active_failed_seed_ids"] = sorted(active)
            state.setdefault("abandon_decisions", []).append(
                {"seed_ids": seeds, "reason": reason, "decided_at": utc_now()}
            )
            review = self._review_state(state)
            if review.get("required"):
                review.setdefault("decisions", []).append(
                    {
                        "decision": "abandon",
                        "seed_ids": seeds,
                        "reason": reason,
                        "reviewed_reasons": copy.deepcopy(review.get("reasons", [])),
                        "decided_at": utc_now(),
                    }
                )
                review["acknowledged_attempt_counts"] = copy.deepcopy(
                    review["policy_attempt_counts"]
                )
                review["consecutive_no_gain_candidates"] = 0
                review["required"] = False
                review["reasons"] = []
            state["updated_at"] = utc_now()
            atomic_write_json(self.state_path, state)
            current_best = state.get("current_best")
            success_ids = list(state["current_success_seed_ids"])
            active_ids = list(state["active_failed_seed_ids"])
            abandoned_ids = list(state["abandoned_seed_ids"])
        for seed in seeds:
            append_jsonl(
                self._seed_history_path(seed),
                {"event": "abandoned", "reason": reason, "recorded_at": utc_now()},
            )
        if current_best:
            coverage = {
                "total": len(state["seed_ids"]),
                "solved": len(success_ids),
                "active_failed": len(active_ids),
                "abandoned": len(abandoned_ids),
                "success_seed_ids": success_ids,
                "active_failed_seed_ids": active_ids,
                "abandoned_seed_ids": abandoned_ids,
            }
            best_root = self.root / "current_best"
            atomic_write_json(best_root / "coverage.json", coverage)
            best_manifest = read_json(best_root / "manifest.json")
            best_manifest.update(
                {
                    "failed_seed_ids": active_ids,
                    "abandoned_seed_ids": abandoned_ids,
                    "updated_at": utc_now(),
                }
            )
            atomic_write_json(best_root / "manifest.json", best_manifest)
            self.task.campaign.experience.update_fm_metadata(
                task_key=self.task.task_key,
                task_slug=self.task.task_slug,
                mode_id=self.mode_id,
                updates={
                    "failed_seed_ids": active_ids,
                    "abandoned_seed_ids": abandoned_ids,
                },
            )
        append_jsonl(
            self.progress_path,
            {
                "event": "abandon",
                "failure_mode_id": self.mode_id,
                "seed_ids": seeds,
                "reason": reason,
                "elapsed_hours": self.task.budget_snapshot()["elapsed_hours"],
                "recorded_at": utc_now(),
            },
        )
        return self.state()

    def reopen_abandoned(self, seed_ids: Iterable[str], *, reason: str) -> dict[str, Any]:
        """Return explicitly abandoned seeds to the active repair set."""

        if not reason.strip():
            raise ValueError("reopen reason cannot be empty")
        seeds = self._normalize_seed_ids(seed_ids)
        with locked_file(self.lock_path):
            state = read_json(self.state_path)
            abandoned = set(state["abandoned_seed_ids"])
            invalid = sorted(set(seeds) - abandoned)
            if invalid:
                raise ValueError(f"only abandoned seeds may be reopened: {invalid}")
            abandoned -= set(seeds)
            active = set(state["active_failed_seed_ids"]) | set(seeds)
            success = set(state["current_success_seed_ids"])
            active -= success
            state["abandoned_seed_ids"] = sorted(abandoned)
            state["active_failed_seed_ids"] = sorted(active)
            state.setdefault("reopen_decisions", []).append(
                {"seed_ids": seeds, "reason": reason, "decided_at": utc_now()}
            )
            state["updated_at"] = utc_now()
            atomic_write_json(self.state_path, state)
            success_ids = list(state["current_success_seed_ids"])
            active_ids = list(state["active_failed_seed_ids"])
            abandoned_ids = list(state["abandoned_seed_ids"])
            solutions = copy.deepcopy(self._seed_solutions_locked(state))
            atomic_write_json(self.state_path, state)
        for seed in seeds:
            append_jsonl(
                self._seed_history_path(seed),
                {"event": "reopened", "reason": reason, "recorded_at": utc_now()},
            )
        best_root = self.root / "current_best"
        if (best_root / "coverage.json").is_file():
            coverage = {
                "total": len(state["seed_ids"]),
                "solved": len(success_ids),
                "active_failed": len(active_ids),
                "abandoned": len(abandoned_ids),
                "success_seed_ids": success_ids,
                "active_failed_seed_ids": active_ids,
                "abandoned_seed_ids": abandoned_ids,
            }
            if solutions:
                coverage["seed_solution_map"] = solutions
            atomic_write_json(best_root / "coverage.json", coverage)
            best_manifest = read_json(best_root / "manifest.json")
            best_manifest.update(
                {
                    "failed_seed_ids": active_ids,
                    "abandoned_seed_ids": abandoned_ids,
                    "updated_at": utc_now(),
                }
            )
            atomic_write_json(best_root / "manifest.json", best_manifest)
            self.task.campaign.experience.update_fm_metadata(
                task_key=self.task.task_key,
                task_slug=self.task.task_slug,
                mode_id=self.mode_id,
                updates={
                    "failed_seed_ids": active_ids,
                    "abandoned_seed_ids": abandoned_ids,
                },
            )
        append_jsonl(
            self.progress_path,
            {
                "event": "reopen_abandoned",
                "failure_mode_id": self.mode_id,
                "seed_ids": seeds,
                "reason": reason,
                "recorded_at": utc_now(),
            },
        )
        self.task._write_status("running")
        return self.state()


class RepairRound:
    def __init__(self, failure_mode: FailureModeSession, round_id: str):
        self.failure_mode = failure_mode
        self.id = round_id

    def _record(self) -> dict[str, Any]:
        state = self.failure_mode.state()
        return next(dict(item) for item in state["rounds"] if item["id"] == self.id)

    def _create_candidate(
        self,
        *,
        source: str,
        parent_ref: str,
        change_summary: str,
        representative_seed_ids: list[str] | None = None,
    ) -> RepairCandidate:
        if not source.strip():
            raise ValueError("candidate source cannot be empty")
        canonical_source = source.rstrip() + "\n"
        ast.parse(canonical_source, filename="<repair_candidate>")
        digest = sha256_text(canonical_source)
        round_record = self._record()
        fm = self.failure_mode
        if not round_record["initial"]:
            fm.require_exploration_decision()
        if not round_record["initial"] and not parent_ref:
            raise ValueError("non-initial candidates require one primary parent_ref")
        if parent_ref:
            fm._validate_parent_reference(parent_ref)
        with locked_file(fm.lock_path):
            state = read_json(fm.state_path)
            duplicate_id = next(
                (
                    existing_id
                    for existing_id in state.get("candidates", {})
                    if fm._candidate_from_raw_state(state, existing_id).get("program_sha256")
                    == digest
                ),
                None,
            )
            if duplicate_id:
                raise ValueError(
                    f"identical program already exists as {duplicate_id}; reuse its canonical "
                    "results instead of creating a duplicate candidate"
                )
            index = int(state["next_candidate"])
            state_round = next(item for item in state["rounds"] if item["id"] == self.id)
            idea_index = len(state_round.get("candidate_ids", [])) + 1
            candidate_id = f"candidate_{index:04d}"
            folder = f"{candidate_id}_{round_record['mechanism_slug']}"
            path = Path(str(round_record["path"])) / folder
            path.mkdir(parents=True, exist_ok=False)
            # Candidate code is immutable; promoted/current-experience paths
            # hardlink these exact bytes instead of copying them.
            atomic_write_text(path / "program.py", canonical_source, mode=0o444)
            record = {
                "id": candidate_id,
                "idea_index": idea_index,
                "round_id": self.id,
                "target_mechanism": round_record["mechanism"],
                "target_seed_ids": list(round_record["seed_ids"]),
                "representative_seed_ids": representative_seed_ids or [],
                "parent_ref": parent_ref,
                "change_summary": change_summary,
                "program_sha256": digest,
                "path": str(path),
                "status": "created",
                "evaluations": [],
                "results": {},
                "created_at": utc_now(),
            }
            atomic_write_json(path / "manifest.json", record)
            state["candidates"][candidate_id] = {
                "manifest_ref": str(path / "manifest.json")
            }
            state_round["candidate_ids"].append(candidate_id)
            atomic_write_json(
                Path(str(state_round["path"])) / "failure_analysis.json",
                state_round,
            )
            state["next_candidate"] = index + 1
            state["updated_at"] = utc_now()
            atomic_write_json(fm.state_path, state)
        append_jsonl(
            fm.progress_path,
            {
                "event": "candidate_created",
                "failure_mode_id": fm.mode_id,
                "round_id": self.id,
                "candidate_id": candidate_id,
                "idea_index": idea_index,
                "mechanism": round_record["mechanism"],
                "parent_ref": parent_ref,
                "change_summary": change_summary,
                "program_sha256": digest,
                "recorded_at": utc_now(),
            },
        )
        return RepairCandidate(fm, candidate_id)

    def create_candidate(
        self,
        *,
        source: str,
        parent_ref: str,
        change_summary: str,
    ) -> RepairCandidate:
        return self._create_candidate(
            source=source,
            parent_ref=parent_ref,
            change_summary=change_summary,
        )


class RepairCandidate:
    def __init__(self, failure_mode: FailureModeSession, candidate_id: str):
        self.failure_mode = failure_mode
        self.id = candidate_id

    def manifest(self) -> dict[str, Any]:
        return self.failure_mode._candidate(self.id)

    def _validate_smoke(self, seeds: list[str], target: list[str], *, initial: bool) -> None:
        if not set(seeds).issubset(target):
            raise ValueError("smoke seeds must belong to the candidate target mechanism")
        if initial:
            required = min(self.failure_mode._smoke_min, len(target))
            if len(seeds) != required:
                raise ValueError(f"initial smoke requires exactly {required} seeds")
            return
        if len(target) < self.failure_mode._smoke_min:
            if set(seeds) != set(target):
                raise ValueError("small mechanism clusters must smoke-test every seed")
            return
        if not self.failure_mode._smoke_min <= len(seeds) <= self.failure_mode._smoke_max:
            raise ValueError(
                f"candidate smoke requires {self.failure_mode._smoke_min}-"
                f"{self.failure_mode._smoke_max} representative seeds"
            )

    def evaluate_smoke(self, seed_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        candidate = self.manifest()
        if candidate["status"] not in {"created", "smoke_tested"}:
            raise RuntimeError("smoke evaluation is closed after the candidate decision")
        state = self.failure_mode.state()
        round_record = next(
            item for item in state["rounds"] if item["id"] == candidate["round_id"]
        )
        requested = list(seed_ids or candidate.get("representative_seed_ids", []))
        seeds = self.failure_mode._normalize_seed_ids(requested)
        self._validate_smoke(
            seeds,
            list(candidate["target_seed_ids"]),
            initial=bool(round_record["initial"]),
        )
        results = self.failure_mode._evaluate_candidate(self.id, stage="smoke", seed_ids=seeds)
        with locked_file(self.failure_mode.lock_path):
            state = read_json(self.failure_mode.state_path)
            record = self.failure_mode._candidate_from_raw_state(state, self.id)
            record["representative_seed_ids"] = seeds
            record["status"] = "smoke_tested"
            self.failure_mode._write_candidate_manifest(record)
            state["updated_at"] = utc_now()
            atomic_write_json(self.failure_mode.state_path, state)
        return results

    def record_failure_video_analysis(
        self,
        analyses: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        """Record auditable wide/wrist observations for policy-failure attempts.

        The runtime can verify that both videos exist and that structured
        observations were recorded. It cannot verify the quality of an
        agent's visual reasoning, so the prompt remains responsible for asking
        for concrete, temporally grounded observations.
        """

        if not analyses:
            raise ValueError("failure video analysis cannot be empty")
        normalized: dict[str, dict[str, str]] = {}
        for provided_seed, raw in analyses.items():
            seed = self.failure_mode._normalize_seed_ids([provided_seed])[0]
            if not isinstance(raw, dict):
                raise TypeError(f"video analysis for {seed} must be a mapping")
            missing = [
                field
                for field in VIDEO_ANALYSIS_FIELDS
                if not str(raw.get(field, "")).strip()
            ]
            if missing:
                raise ValueError(
                    f"video analysis for {seed} is missing non-empty fields: {missing}"
                )
            normalized[seed] = {
                field: str(raw[field]).strip() for field in VIDEO_ANALYSIS_FIELDS
            }

        history_records: list[tuple[str, dict[str, Any]]] = []
        with locked_file(self.failure_mode.lock_path):
            state = read_json(self.failure_mode.state_path)
            candidate = self.failure_mode._candidate_from_raw_state(state, self.id)
            results = dict(candidate.get("results", {}))
            recorded = candidate.setdefault("failure_video_analyses", {})
            for seed, observations in normalized.items():
                result = dict(results.get(seed) or {})
                if result.get("outcome") != "policy_failure":
                    raise ValueError(
                        f"video analysis may only be recorded for a policy_failure result: {seed}"
                    )
                attempt = Path(str(result.get("attempt_path", "")))
                video_files: dict[str, dict[str, Any]] = {}
                for view, filename in (("wide", "wide.mp4"), ("wrist", "wrist.mp4")):
                    path = attempt / filename
                    if not path.is_file() or path.stat().st_size <= 0:
                        raise FileNotFoundError(
                            f"cannot analyze {seed}: required {view} video is missing or empty: {path}"
                        )
                    video_files[view] = {"path": str(path), "size_bytes": path.stat().st_size}
                evaluation_entries = [
                    item
                    for item in self.failure_mode._history_entries(seed)
                    if item.get("candidate_id") == self.id
                    and item.get("result_ref") == result.get("result_ref")
                ]
                if not evaluation_entries:
                    raise RuntimeError(f"policy-failure history is missing for {seed}")
                analysis = {
                    "candidate_id": self.id,
                    "evaluation_id": evaluation_entries[-1].get("evaluation_id"),
                    "result_ref": result.get("result_ref"),
                    "video_files": video_files,
                    "observations": observations,
                    "analyzed_at": utc_now(),
                }
                recorded[seed] = analysis
                history_records.append((seed, analysis))
            analysis_path = Path(str(candidate["path"])) / "failure_video_analysis.json"
            candidate["failure_video_analysis_ref"] = str(analysis_path)
            candidate["updated_at"] = utc_now()
            atomic_write_json(
                analysis_path,
                {
                    "schema_version": 2,
                    "candidate_id": self.id,
                    "analyses": recorded,
                    "updated_at": utc_now(),
                },
            )
            self.failure_mode._write_candidate_manifest(candidate)
            state["updated_at"] = utc_now()
            atomic_write_json(self.failure_mode.state_path, state)

        for seed, analysis in history_records:
            append_jsonl(
                self.failure_mode._seed_history_path(seed),
                {
                    "event": "failure_video_analysis",
                    "candidate_id": self.id,
                    "evaluation_id": analysis.get("evaluation_id"),
                    "result_ref": analysis.get("result_ref"),
                    "analysis_ref": str(analysis_path),
                    "recorded_at": utc_now(),
                },
            )
        append_jsonl(
            self.failure_mode.progress_path,
            {
                "event": "failure_video_analysis",
                "failure_mode_id": self.failure_mode.mode_id,
                "candidate_id": self.id,
                "seed_ids": sorted(normalized),
                "recorded_at": utc_now(),
            },
        )
        return self.manifest()

    def decide(self, decision: str, *, rationale: str) -> dict[str, Any]:
        if decision not in {"expand", "stop"}:
            raise ValueError("candidate decision must be expand or stop")
        if not rationale.strip():
            raise ValueError("candidate decision rationale cannot be empty")
        if decision == "expand":
            self.failure_mode.require_exploration_decision()
        with locked_file(self.failure_mode.lock_path):
            state = read_json(self.failure_mode.state_path)
            candidate = self.failure_mode._candidate_from_raw_state(state, self.id)
            if candidate["status"] != "smoke_tested":
                raise RuntimeError("candidate must finish smoke evaluation before a decision")
            smoke_policy_failures = sorted(
                seed
                for seed in candidate.get("representative_seed_ids", [])
                if str(candidate.get("results", {}).get(seed, {}).get("outcome"))
                == "policy_failure"
            )
            missing_video_analysis = sorted(
                seed
                for seed in smoke_policy_failures
                if (
                    seed not in candidate.get("failure_video_analyses", {})
                    or candidate["failure_video_analyses"][seed].get("result_ref")
                    != candidate.get("results", {}).get(seed, {}).get("result_ref")
                )
            )
            if missing_video_analysis:
                raise RuntimeError(
                    "smoke policy failures require recorded wide/wrist video analysis before "
                    "a candidate decision: "
                    + ", ".join(missing_video_analysis)
                )
            if decision == "expand":
                unresolved = sorted(
                    seed
                    for seed in candidate.get("representative_seed_ids", [])
                    if str(candidate.get("results", {}).get(seed, {}).get("outcome"))
                    not in POLICY_OUTCOMES
                )
                if unresolved:
                    raise RuntimeError(
                        "smoke reset/infra/invalid outcomes must be resolved before expansion: "
                        + ", ".join(unresolved)
                    )
            candidate["decision"] = {"value": decision, "rationale": rationale, "at": utc_now()}
            candidate["status"] = "expand_approved" if decision == "expand" else "stopped"
            self.failure_mode._write_candidate_manifest(candidate)
            if decision == "stop":
                self.failure_mode._record_no_gain_locked(
                    state,
                    candidate_id=self.id,
                    disposition="stopped_after_smoke",
                )
            state["updated_at"] = utc_now()
            atomic_write_json(self.failure_mode.state_path, state)
            if decision == "stop":
                self.failure_mode._compact_successes(candidate)
            decided = dict(candidate)
        append_jsonl(
            self.failure_mode.progress_path,
            {
                "event": "candidate_decision",
                "failure_mode_id": self.failure_mode.mode_id,
                "candidate_id": self.id,
                "decision": decision,
                "rationale": rationale,
                "recorded_at": utc_now(),
            },
        )
        return decided

    def _evaluate_remaining(self, *, initial_required: bool) -> list[dict[str, Any]]:
        candidate = self.manifest()
        state = self.failure_mode.state()
        round_record = next(
            item for item in state["rounds"] if item["id"] == candidate["round_id"]
        )
        if bool(round_record["initial"]) != initial_required:
            kind = "initial" if initial_required else "mechanism"
            raise RuntimeError(f"this operation requires a {kind} candidate")
        if candidate["status"] != "expand_approved":
            raise RuntimeError("candidate expansion must be approved after smoke evaluation")
        completed = set(candidate.get("results", {}))
        remaining = [seed for seed in candidate["target_seed_ids"] if seed not in completed]
        results = (
            self.failure_mode._evaluate_candidate(self.id, stage="expanded", seed_ids=remaining)
            if remaining
            else []
        )
        with locked_file(self.failure_mode.lock_path):
            state = read_json(self.failure_mode.state_path)
            record = self.failure_mode._candidate_from_raw_state(state, self.id)
            record["status"] = "expanded"
            self.failure_mode._write_candidate_manifest(record)
            state["updated_at"] = utc_now()
            atomic_write_json(self.failure_mode.state_path, state)
        return results

    def evaluate_remaining_fm_seeds(self) -> list[dict[str, Any]]:
        return self._evaluate_remaining(initial_required=True)

    def evaluate_remaining_cluster_seeds(self) -> list[dict[str, Any]]:
        return self._evaluate_remaining(initial_required=False)

    def evaluate_stability(self, seed_ids: Iterable[str]) -> list[dict[str, Any]]:
        """Explicitly rerun exact candidate/seeds without changing canonical coverage."""

        seeds = self.failure_mode._normalize_seed_ids(seed_ids)
        if not seeds:
            raise ValueError("stability evaluation requires at least one seed")
        return self.failure_mode._evaluate_candidate(
            self.id,
            stage="stability",
            seed_ids=seeds,
            execution_mode="stability",
            update_canonical_results=False,
        )

    def evaluate_targeted_seeds(self, seed_ids: Iterable[str]) -> list[dict[str, Any]]:
        """Evaluate an existing portfolio program on additional active seeds."""

        seeds = self.failure_mode._normalize_seed_ids(seed_ids)
        if not seeds:
            raise ValueError("targeted evaluation requires at least one seed")
        with locked_file(self.failure_mode.lock_path):
            state = read_json(self.failure_mode.state_path)
            candidate = self.failure_mode._candidate_from_raw_state(state, self.id)
            if candidate.get("status") not in {
                "expanded",
                "promoted",
                "promoted_seed_specific",
                "stopped",
                "not_promoted",
            }:
                raise RuntimeError(
                    "only an evaluated candidate may add targeted seeds"
                )
            active = set(state["active_failed_seed_ids"])
            invalid = sorted(set(seeds) - active)
            if invalid:
                raise ValueError(
                    "targeted evaluation may only add active failed seeds: "
                    + ", ".join(invalid)
                )
            targets = list(candidate.get("target_seed_ids", []))
            for seed in seeds:
                if seed not in targets:
                    targets.append(seed)
            candidate["target_seed_ids"] = targets
            candidate.setdefault("target_extensions", []).append(
                {"seed_ids": seeds, "extended_at": utc_now()}
            )
            self.failure_mode._write_candidate_manifest(candidate)
            state["updated_at"] = utc_now()
            atomic_write_json(self.failure_mode.state_path, state)
        return self.failure_mode._evaluate_candidate(
            self.id,
            stage="targeted",
            seed_ids=seeds,
        )

    def retry_reset_or_infrastructure(self, seed_ids: Iterable[str]) -> list[dict[str, Any]]:
        """Replace excluded diagnostic outcomes after their external cause is resolved."""

        candidate = self.manifest()
        seeds = self.failure_mode._normalize_seed_ids(seed_ids)
        invalid = sorted(
            seed
            for seed in seeds
            if str(candidate.get("results", {}).get(seed, {}).get("outcome"))
            not in {"reset_failure", "infrastructure_failure"}
        )
        if invalid:
            raise ValueError(
                "only reset or infrastructure outcomes may use diagnostic retry: "
                + ", ".join(invalid)
            )
        return self.failure_mode._evaluate_candidate(
            self.id,
            stage="diagnostic_retry",
            seed_ids=seeds,
            execution_mode="diagnostic_retry",
            update_canonical_results=True,
        )
