# Sequential VLA pre-repair campaign

You are the coordinator for one ordered VLA/LIBERO pre-repair campaign. There
are {{TASK_COUNT}} independent task runs. Execute them strictly in the listed
order and complete the full task-level prompt for one task before starting the
next task.

## Campaign identity

- Campaign root: `{{CAMPAIGN_ROOT}}`
- Campaign manifest: `{{CAMPAIGN_MANIFEST}}`
- Campaign fingerprint: `{{CAMPAIGN_FINGERPRINT}}`
- Scheduling policy: sequential and fail-fast

Each task has an isolated resolved settings file, output root, fingerprint,
rollout, diagnosis, and repair handoff. Never write artifacts from one task
under another task's run root, and never reuse a rollout or diagnosis merely
because two tasks share a checkpoint or runtime parameters.

## Ordered tasks

{{TASK_LIST}}

## Coordinator contract

For each task in order:

1. Read its task prompt and resolved settings before executing anything.
2. Complete every stage in that task prompt, including rollout validation,
   public-evidence diagnosis, reset materialization, and read-back validation
   of `repair_handoff/manifest.json`.
3. Treat policy timeout or predicate failure as task evidence to diagnose.
   Treat a simulator, model, service, filesystem, or other infrastructure
   exception as infrastructure failure, not as a policy failure.
4. Start the next listed task only after the current task's handoff manifest
   exists, has `artifact_type=vla_mender.repair_handoff`, has `complete=true`,
   matches the current task fingerprint, and reports
   `summary.all_replays_verified=true`.
5. If the current task cannot reach that completion condition, stop the
   campaign at that task and report the exact blocker. Do not skip ahead or
   mark partial artifacts complete.

Shared GPUs and local services may be reused only after the previous task has
fully stopped using them. The campaign is sequential: do not run task-level
rollouts, diagnosis, or materialization concurrently.

The campaign is complete only when every listed task has a separately verified
`repair_handoff/manifest.json`. Do not continue into repair-policy execution or
model training.
