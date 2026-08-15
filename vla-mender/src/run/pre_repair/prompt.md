# VLA pre-repair task workflow

You are the coordinator for one VLA/LIBERO pre-repair run. Complete the full
pre-repair workflow in this prompt: execute or verify the VLA rollout, preserve
all successful and failed trajectories, prepare public evidence, diagnose each
failure, cluster failure modes, and define recoverable failure windows. Stop
when the validated diagnosis (and, if requested by the coordinator, the replay-
verified reset bank) is produced. Do not train a model, synthesize repair code,
or execute a repair policy.

## Resolved experiment contract

- Suite/task: `{{SUITE}}:{{TASK_ID}}`
- Task instruction: `{{TASK_DESCRIPTION}}`
- Checkpoint: `{{CHECKPOINT}}`
- Runtime backend: `{{RUNTIME_BACKEND}}`
- OpenPI commit: `{{OPENPI_COMMIT}}`
- OpenPI environment: `{{OPENPI_ENVIRONMENT}}`
- LIBERO resource root: `{{LIBERO_ROOT}}`
- Trajectory protocol: `{{TRAJECTORY_PROTOCOL}}`
- Initial-state provider: `{{STATE_PROVIDER}}`
- Initial-state count: `{{STATE_COUNT}}`
- State manifest: `{{STATE_MANIFEST}}`
- Control frequency: `{{CONTROL_FREQUENCY_HZ}}` Hz
- Maximum policy steps per episode: `{{MAX_STEPS}}`
- Policy seed: `{{POLICY_SEED}}`
- GPUs/workers: `{{GPUS}}` / {{WORKERS_PER_GPU}} worker per GPU
- Action chunk: `{{ACTION_CHUNK}}`
- Inference steps: `{{INFERENCE_STEPS}}`
- Reset stabilization steps: `{{NUM_STEPS_WAIT}}`
- Binary gripper: `{{BINARY_GRIPPER}}`
- Gripper hysteresis threshold: `{{GRIPPER_HYSTERESIS_THRESHOLD}}`
- Source control space: `{{SOURCE_CONTROL_SPACE}}`
- Target control space: `{{TARGET_CONTROL_SPACE}}`
- Reset dynamics: `{{RESET_DYNAMICS}}`
- Reset candidates per failure: `{{FRAMES_PER_FAILURE}}`
- Reset candidate stride: `{{FRAME_STRIDE}}` frames
- Run output: `{{OUTPUT_DIR}}`

These values are the experiment identity. Do not silently change them. If a
rollout must be rerun, use the same checkpoint, initial-state manifest, seed,
control space, frequency, horizon, action chunk, inference steps, GPU mapping,
and output contract.

Use `backend.libero_root` from the resolved experiment contract for LIBERO
resources. Do not create a run-local `libero_config/config.yaml`, and do not
require `LIBERO_CONFIG_PATH`; the workflow applies this resource root directly
in the simulator process.

## Stage 1 — execute and freeze the VLA rollout

If `{{OUTPUT_DIR}}/rollout/summary.json` is missing, run the generic rollout
entrypoint with the resolved settings. This entrypoint uses the shared
`workflow.rollout.state_provider`, `runner`, and `evaluator` core; do not replace
it with the standalone LeRobot campaign writer:

```bash
PYTHONPATH=vla-mender/src {{OPENPI_ENVIRONMENT}}/bin/python -m workflow.pipeline rollout \
  --settings {{SETTINGS_PATH}} \
  --output {{OUTPUT_DIR}}
```

If the rollout already exists, do not rerun it automatically. First validate
that it matches this contract. The pre-repair rollout command does not implement
resume or overwrite; reuse a complete matching rollout, otherwise stop and
select a new run output. Never write a new rollout into an existing run.

The rollout must preserve every episode, not only failures. For each episode,
save:

- `rollout/episodes/episode_<index>.json`, including the executed public state
  sequence, executed actions, rewards, per-step success flags, outcome, and
  initial-state provenance;
- `rollout/videos/episode_<index>_wide.mp4`;
- `rollout/videos/episode_<index>_wrist.mp4`;
- `rollout/summary.json` with the exact success/failure counts;
- `rollout/successful_episodes.json` and `rollout/failed_episodes.json` as
  episode manifests;
- `rollout/initial_state_manifest.json` and the coordinator-owned state array.

A success is an episode where LIBERO reports native `done`/success during the
configured policy budget; execution stops at that first success. A timeout or
predicate failure is a task failure. An infrastructure exception is not a task
failure: it aborts the rollout stage and leaves an incomplete run that must not
be diagnosed. Do not fabricate a trajectory to replace it.

Before diagnosis, verify: the requested episode count is complete; episode
indices are unique; every episode has finite state/action data; state/action
lengths agree; both videos exist and decode; success plus task-failure plus
failure counts account for every requested rollout; and the saved contract
matches this prompt. If an infrastructure error occurred, stop instead of
treating the partial output as complete.

## Stage 2 — prepare public evidence

Create the diagnosis handoff with the generic diagnosis entrypoint:

```bash
PYTHONPATH=vla-mender/src {{OPENPI_ENVIRONMENT}}/bin/python -m workflow.pipeline diagnose \
  --settings {{SETTINGS_PATH}} \
  --output {{OUTPUT_DIR}}
```

`failure_diagnosis/agent_input.json` must contain public evidence for every
failed episode and public reference evidence for successful episodes. A record
contains its episode index, sparse state/action timeline, outcome, and explicit
wide/wrist video paths. Do not expose private simulator state, BDDL/XML,
object coordinates, hidden predicates, rewards, `done`, reset payloads, or
seed-specific simulator internals to the diagnosis agent.

Use successful trajectories as behavioral references: compare stages,
observable motion, contact/engagement, and task completion. Do not copy a
successful action sequence as an open-loop repair policy and do not infer hidden
object state from unavailable fields.

## Stage 3 — diagnose each failed trajectory

For every failed episode, in episode-index order:

1. Inspect the sparse timeline and both camera views.
2. Identify the semantic `failure_phase` first, such as approach, alignment,
   grasp/contact acquisition, manipulation, transport, release, recovery, or
   timeout. Use the most specific phase supported by public evidence.
3. Identify `first_causal_frame_index`: the earliest frame after which the
   behavior leaves a recoverable path, not merely the frame where failure is
   most visible.
4. Define one inclusive recoverable window
   `[recoverable_window_start_frame_index,
   recoverable_window_stop_frame_index]`. It must contain the causal frame,
   remain within the trajectory, and stop before the state becomes
   irrecoverable. Do not default to the whole trajectory.
5. Require an evidence-derived failure window for each episode. Its index span
   must be strictly greater than 10: `window_stop - window_start > 10`
   (equivalently, at least 11 frame intervals). A window such as `[0, 10]`
   is therefore invalid. Do not reuse one fixed start/stop interval across all
   episodes; each episode must have its own window supported by its public
   video, state, and action evidence.
6. Record short evidence statements that are directly observable in the
   videos, state, or executed actions. Lower confidence when evidence is
   ambiguous.

Use zero-based frame indices and require
`0 <= start <= causal <= stop < num_frames`. A timeout is still diagnosed at
its earliest causal failure, not only at the final frame.

## Stage 4 — cluster failure modes

After every failed episode has a phase and window, group failures into a
small, task-local set of root-cause modes. A mode is a repeated causal
mechanism, not merely an outcome label or a frame range. Assign stable IDs
`FM-01`, `FM-02`, ... within this run. A phase may contain multiple modes, and
a mode may occur in more than one phase. Use only modes supported by the
recorded public evidence; create `FM-OTHER` for genuinely unclassifiable cases.

For each mode, report its label, broader category, episode membership,
frequency, one representative episode, and an evidence-based causal
mechanism. Do not include repair code, repair actions, or training advice.

## Stage 5 — required diagnosis output

Write `failure_diagnosis/diagnosis.json` with this schema:

```json
{
  "schema_version": 1,
  "successful_reference_episodes": [
    {"episode_index": 0, "reason": "observable successful phase reference"}
  ],
  "failure_modes": [
    {
      "failure_mode_id": "FM-01",
      "label": "short root-cause label",
      "category": "alignment",
      "episode_indices": [3, 7],
      "frequency": 2,
      "representative_episode_index": 3,
      "causal_mechanism": "short evidence-based mechanism"
    }
  ],
  "episodes": [
    {
      "episode_index": 3,
      "failure_phase": "approach/alignment",
      "failure_mode_id": "FM-01",
      "failure_category": "alignment",
      "failure_mode": "short root-cause label",
      "first_causal_frame_index": 86,
      "recoverable_window_start_frame_index": 78,
      "recoverable_window_stop_frame_index": 102,
      "evidence": ["short public evidence statement"],
      "confidence": 0.0
    }
  ]
}
```

The episode list must contain every task failure exactly once and no successful
or infrastructure-error episode. Every `failure_mode_id` must occur in
`failure_modes`, and every mode's `episode_indices` must match the episode
records. `evidence` must be non-empty and `confidence` must be in `[0, 1]`.
Do not add private reset data, hidden state, repaired actions, or unrelated
fields.

## Stage 6 — optional reset-bank materialization

Only after `diagnosis.json` is complete and schema-valid, the trusted
coordinator may run:

```bash
PYTHONPATH=vla-mender/src {{OPENPI_ENVIRONMENT}}/bin/python -m workflow.pipeline materialize \
  --settings {{SETTINGS_PATH}} \
  --output {{OUTPUT_DIR}} \
  --diagnosis {{OUTPUT_DIR}}/failure_diagnosis/diagnosis.json
```

This stage applies the configured stride/count, replays each selected action
prefix, verifies public-state agreement, applies `{{RESET_DYNAMICS}}`, and
writes reset candidates, replay verification, private reset states, and repair
jobs. The diagnosis agent does not access or select private reset payloads.

The pre-repair workflow ends with the rollout artifacts, public diagnosis,
clustered failure modes, validated windows, and optional replay-verified reset
bank. Do not continue into repair-policy execution or model training.
