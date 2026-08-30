# Prompt-driven repair infrastructure

`workflow.research` provides the reusable runtime for prompt-launched repair
campaigns. Schema v2 uses one open seed pool per failure mode and a persistent
FM state machine; it does not create fixed dataset partitions or split an FM
into mechanism-specific modes.

## V2 workflow

Every task starts with prompt-guided analysis of its observable phases, required
actions, perception/control needs, and diagnosed failure mechanism. The agent
then searches the example catalogs, relevant `SKILL.md` files, API README,
runtime API selection, exposed `functions()` mappings, and implementations
directly from the filesystem. Before code or rollouts, it atomically saves the
action analysis, reusable-code checklist, and searched scopes as
`task_start_knowledge_review.json` under the readable task directory. The full
review is not printed in the conversation, creates no runtime gate, is not
hashed, and is not candidate provenance.

For each FM, the agent creates one initial full program, smoke-tests exactly
three representative seeds (or every seed if fewer exist), and expands an
effective version across the remaining FM seeds. Policy failures are then
grouped into temporary mechanism clusters. Each later candidate declares one
mechanism, one primary `parent_ref`, and a complete program. The parent is
normally a successful candidate or campaign experience. A targeted seed may
instead retain a failed candidate only when its current policy result has
recorded wide/wrist analysis that visibly proves the partial mechanism being
kept.

Candidate design optimizes for repair leverage: the smallest clear, robust
shared change should cover as many evidence-compatible clusters as possible.
Agents first look for a common phase error, API misuse, control invariant, or
reusable primitive, and add seed-specific branches only when video evidence
shows incompatible mechanisms or controls.

Candidates must pass a 3-8 seed smoke gate before expanding to the rest of the
cluster. `promote_best` automatically evaluates any current-best success seeds
needed for a comparable decision. A candidate replaces the current best only
when its verified success count is strictly larger. Regressed seeds return to
the active failure pool.

Every policy failure must have structured analysis of both `wide.mp4` and
`wrist.mp4`. The runtime verifies that both videos exist and refuses a smoke
decision or mechanism round until the matching failure attempt has recorded
wide-view, wrist-view, failure-moment, and mechanism observations. Trajectory
data may support that analysis but cannot replace video evidence.

```python
from workflow.research import RepairCampaign

campaign = RepairCampaign.open("/path/to/repair_resolved.yaml")
task = campaign.open_task("<task-key>", gpu_ids=[0, 1, 2, 3])
fm = task.open_failure_mode("FM-01")

initial = fm.create_initial_candidate(
    source=source,
    representative_seed_ids=["e000000-f000100", "e000001-f000101", "e000002-f000102"],
    strategy_summary="phase analysis and initial strategy",
)
initial.evaluate_smoke()
initial.record_failure_video_analysis({
    "e000002-f000102": {
        "wide_view": "the pan remains visibly offset from the burner",
        "wrist_view": "the gripper contacts the rim instead of closing around the handle",
        "failure_moment": "rim contact begins before closure and the pan then rotates away",
        "mechanism_evidence": "both views show a contact-placement error, not a reset error",
    }
})
initial.decide("expand", rationale="useful evaluator and wide/wrist video evidence")
initial.evaluate_remaining_fm_seeds()
initial.record_failure_video_analysis({
    "e000010-f000110": {
        "wide_view": "the pan approaches from an offset scene-level path",
        "wrist_view": "the fingers make shallow contact with the handle edge",
        "failure_moment": "the offset begins before grasp closure and persists through transport",
        "mechanism_evidence": "both videos support weak handle contact",
    },
    "e000011-f000111": {
        "wide_view": "the pan rotates away before reaching the burner center",
        "wrist_view": "the handle visibly slips during lateral motion",
        "failure_moment": "slip begins immediately after lifting",
        "mechanism_evidence": "the visible slip matches the same contact mechanism",
    },
})
fm.promote_best(initial.id)

round = fm.start_round(
    mechanism="weak_handle_contact",
    seed_ids=["e000010-f000110", "e000011-f000111"],
    evidence_summary="shared video and trace signature",
)
candidate = round.create_candidate(
    source=updated_source,
    parent_ref=fm.current_best_ref(),
    change_summary="one contact-specific change",
)
candidate.evaluate_smoke(seed_ids=["e000010-f000110", "e000011-f000111"])
candidate.record_failure_video_analysis({
    "e000011-f000111": {
        "wide_view": "the pan reaches the burner edge but remains off center",
        "wrist_view": "the handle slips after a shallow fingertip contact",
        "failure_moment": "slip starts during lateral transport and causes the final offset",
        "mechanism_evidence": "the visible slip is shared with the round's contact mechanism",
    }
})
candidate.decide("expand", rationale="target mechanism visibly improved")
candidate.evaluate_remaining_cluster_seeds()
fm.promote_best(candidate.id, skills={})

summary = task.finish()
```

## Outcomes and coverage

The runtime records `success`, `policy_failure`, `policy_invalid`,
`reset_failure`, and `infrastructure_failure`. Only evaluator-confirmed success
contributes to coverage. Only `policy_failure` may enter a mechanism cluster.
Reset and infrastructure problems remain diagnostic evidence and must be
resolved or excluded before promotion.

Every FM tracks current successes, active failures, abandoned seeds, historical
successes, a seed-to-candidate solution portfolio, rounds, candidates, and
per-seed JSONL history. Hard seeds may use targeted programs; coverage is the
union of evaluator-confirmed per-seed solutions, not necessarily the output of
one universal program. Completion is derived from that ledger:

- `completed`: every seed is currently solved;
- `completed_partial`: every seed is solved or explicitly abandoned;
- `in_progress`: at least one active failure remains;
- `problem_reported`: repair is blocked by an external problem.

## Experience and artifacts

Schema v2 stores readable task, FM, round, candidate, evaluation, rerun, and seed
names. SHA256 values live in manifests, not directory names. Each `(task, FM)`
has one current experience slot containing the global baseline plus a
seed-to-candidate portfolio, optional skills, provenance, and coverage.
`promote_best()` replaces the baseline; `promote_seed_solutions()` adds targeted
verified wins without requiring the candidate to preserve unrelated successes.
`evaluate_targeted_seeds()` can reuse the exact promoted program on additional
active seeds without creating a byte-identical candidate. A candidate stopped
for one cluster may also be reused this way: stopping is evidence about the
tested seeds, not a global ban on that program.
Older versions remain candidate history rather than separate searchable
experience entries.

Current-best successes and every policy failure retain full wide/wrist video and
trajectory evidence. Superseded successes keep code hash, evaluator result,
terminal observation, and logs while bulky video/trajectory files are compacted.
`runtime/cache_index.json` maps program SHA/reset identity to readable evidence;
exact results resume without hash-named directories. Forced stability checks use
sequential `rerun_0001` names and do not replace canonical coverage.

After full evaluator coverage is first reached but before final promotion and
`task.finish()`, the agent performs one bounded trajectory-continuity pass on
representative successes from the formerly different clusters. It checks the
videos and action trace for pauses, stop-start motion, oscillation, redundant
waiting/replanning, abrupt reversals, and phase-boundary discontinuities. Any
accepted simplification or smoothing change must be re-evaluated and preserve
task success; recorded trajectories are never post-processed to simulate an
improvement.

## Optional OSC rollout noise

LIBERO repair jobs may include an `action_noise` object to perturb the first six
normalized `OSC_POSE` command dimensions immediately before every simulator
step. The seventh gripper command is never perturbed. The implementation and
validation live in `workflow.rollout.action_noise`; `libero_backend.py` applies
the transform and records the exact command passed to `env.step`.

```python
job["action_noise"] = {
    "type": "ornstein_uhlenbeck",
    "seed": 7_000_000,
    "rho": 0.85,
    "standard_deviation": [0.03, 0.03, 0.03, 0.02, 0.02, 0.02],
    "maximum_absolute": [0.09, 0.09, 0.09, 0.06, 0.06, 0.06],
}
```

No noise is applied when the object is absent. With noise enabled,
`trajectory.json` contains `nominal_actions`, `sampled_action_noises`,
`applied_action_noises`, and `actions`; `actions` is the clipped command that
was actually executed. Cache identity includes the complete noise object, so
replaying the same reset with a new noise seed produces a real new rollout.
Dataset construction must consume `actions` and must not add or infer noise
after simulation.

## Budget and shared runtime

The v2 YAML supplies a per-task soft wall-clock budget. Crossing it pauses new
evaluation submission until the agent records a reasoned extension or abandons
expensive seeds. Abandonment requires an actual policy attempt and a concrete
reason backed by the latest failure's recorded wide/wrist analysis; no automatic
attempt threshold is imposed.

`campaign.py` owns task/GPU-group leases, dynamic seed dispatch, and the public
API. `runtime.py` owns each GPU-local fixed worker pool, subprocess isolation,
retries, and service group. `libero_backend.py`
restores prepared states and executes the policy. `state.py` owns the FM state
machine. `artifacts.py` and `experience.py` own readable evidence and the unique
FM experience slot.

Schema v1 campaigns retain their legacy API and content-addressed artifact
layout. They are not migrated automatically.

## Experiment ledger

A round represents one declared failure mechanism, not one fixed-size batch of
ideas. The agent may try multiple candidates in that round; each candidate is
one idea and receives a per-round `idea_index` plus an FM-wide readable
candidate ID. There is no hard candidate count. The agent stops based on video
evidence, marginal coverage, and budget.

`progress.jsonl` is the chronological task ledger for round creation, candidate
creation, evaluations, video analysis, expand/stop decisions, promotion, and
abandonment. `failure_analysis.json` lists the round's mechanism, seeds,
evidence, video observations, and candidate IDs. Each candidate retains its
program, manifest, SHA256, evaluation summaries, and per-seed evidence. Each
seed also has `history.jsonl`, so unsuccessful and superseded attempts remain
traceable even when bulky evidence for obsolete successes is compacted.
