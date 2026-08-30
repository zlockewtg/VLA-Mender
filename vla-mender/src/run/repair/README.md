# Standalone repair run

`run.repair` starts from a completed pre-repair `repair_handoff/`. Each task in
the YAML supplies only its pre-repair `run_root`; reset frames are read from the
verified handoff and are never copied into repair settings.

Generate the IDE coordinator prompt with:

```bash
cd /mnt/public/tgy/VLA-Mender
PYTHONPATH=vla-mender/src python -m run.repair.generate_prompt \
  --settings /path/to/repair_campaign/repair_example.yaml
```

Generation validates all task inputs and writes `repair_resolved.yaml`,
`prompt_generated.md`, and `prompt_manifest.json` under the new campaign output
directory. The resolved YAML contains the normalized settings, resolved tasks,
and complete seed-job inventory. It does not allocate a GPU or start a repair
attempt.

Schema v2 exposes every prepared FM seed from the beginning. Its `repair`
settings contain the soft task budget, smoke gate, low-value exploration review,
and abandonment policy:

```yaml
schema_version: 2
repair:
  budget:
    soft_task_hours: 4
  smoke:
    min_seeds: 3
    max_seeds: 8
  exploration_review:
    consecutive_no_gain_candidates: 3
    per_seed_policy_attempts: 8
  allow_abandon: true
```

GPU topology is explicit under `resources`. For example, one task using four
GPUs, one rollout worker and one service group on each GPU is:

```yaml
resources:
  gpus: [0, 1, 2, 3]
  gpus_per_task: 4
  workers_per_gpu: 1
```

The task opens with `gpu_ids=[0, 1, 2, 3]`. Seed jobs share one FM state machine
and are dynamically dispatched across the four GPU-local runtimes.

The exploration thresholds pause new rollout submission for an agent decision;
they do not impose a hard candidate or attempt cap. The agent can record a
reasoned continuation or abandon expensive, evidence-backed long-tail seeds.

One active task agent owns one declared GPU group and a fixed worker pool plus
service group on every GPU. The generated prompt requires the
FM smoke/analysis/cluster/parent/
candidate/promotion workflow and the shared `RepairCampaign` API. Prompt
generation also defines `campaign.parallel_tasks` as the maximum number of
dedicated task agents active at once, including the initial agent receiving the
prompt. There is no coordination-only agent: the initial agent repairs one task
and launches peer task subagents for the others. Every agent owns exactly one
prepared task and is forbidden from opening or repairing any other task. When
the agent platform has fewer slots, remaining tasks stay queued for new
dedicated agents instead of being combined under one task agent. The repair
runtime enforces task/GPU leases but does not itself create agents.
The generated prompt instructs the agent to analyze the task's phases, action needs, and
failure mechanism, then search `knowledge/examples`, `knowledge/skills`, and
the actual observation-only LIBERO API implementation directly. Before code or
rollouts, the agent prints a transient Markdown checklist of every plausibly
useful file or symbol. The checklist is not written into campaign artifacts,
hashed, frozen, or used as a runtime gate; the agent decides later which items
to adapt. Static knowledge does not replace the shared reset, service,
scheduler, artifact infrastructure, required candidate parent, or
evaluator-backed promotion requirements.

New v2 campaigns use readable artifact directories and one current experience
slot per task/FM. For each program/reset pair there is one canonical
`result.json`; candidate manifests, evaluation summaries, and seed histories
store references to it. Stability reruns and reset/infrastructure diagnostics
live in separately numbered readable directories and cannot replace canonical
coverage. Byte-identical evidence files are hardlinked behind their readable
attempt paths through a sequential runtime blob index; promoted program paths
are hardlinks to the immutable candidate program. Duplicate bytes are therefore
stored once without exposing hash directory names. Existing schema v1 campaigns
retain their old settings and layout and are not modified or migrated.
