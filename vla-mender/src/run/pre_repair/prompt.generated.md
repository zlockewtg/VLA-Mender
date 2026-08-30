# VLA pre-repair task workflow

You are the coordinator for one VLA/LIBERO pre-repair run. Complete the full
pre-repair workflow in this prompt: execute or verify the VLA rollout, preserve
all successful and failed trajectories, prepare public evidence, diagnose each
failure, cluster failure modes, and define recoverable failure windows. Stop
when the validated diagnosis (and, if requested by the coordinator, the replay-
verified reset bank) is produced. Do not train a model, synthesize repair code,
or execute a repair policy.

## Resolved experiment contract

- Suite/task: `libero_goal:0`
- Task instruction: `(use the task instruction from the rollout)`
- Checkpoint: `/absolute/path/to/openpi/checkpoint`
- Runtime backend: `openpi`
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- OpenPI environment: `/absolute/path/to/VLA-Mender/.venv-openpi`
- Trajectory protocol: `vla-mender.libero.openpi/v2`
- Initial-state provider: `randomized_bddl`
- Initial-state count: `50`
- State manifest: `(none)`
- Control frequency: `20` Hz
- Maximum policy steps per episode: `300`
- Policy seed: `7`
- GPUs/workers: `0, 1, 2, 3` / 1 worker per GPU
- Action chunk: `5`
- Inference steps: `5`
- Reset stabilization steps: `10`
- Binary gripper: `false`
- Gripper hysteresis threshold: `0.2`
- Source control space: `osc`
- Target control space: `osc`
- Reset dynamics: `preserve_full_state`
- Reset candidates per failure: `3`
- Reset candidate stride: `5` frames
- Run output: `/mnt/public/tgy/VLA-Mender/vla-mender/src/run/pre_repair`

These values are the experiment identity. Do not silently change them. If a
rollout must be rerun, use the same checkpoint, initial-state manifest, seed,
control space, frequency, horizon, action chunk, inference steps, GPU mapping,
and output contract.

## Stage 1 — execute and freeze the VLA rollout

If `/mnt/public/tgy/VLA-Mender/vla-mender/src/run/pre_repair/rollout/summary.json` is missing, run the generic rollout
entrypoint with the resolved settings. This entrypoint uses the shared
`workflow.rollout.state_provider`, `runner`, and `evaluator` core; do not replace
it with the standalone LeRobot campaign writer:

```bash
PYTHONPATH=vla-mender/src /absolute/path/to/VLA-Mender/.venv-openpi/bin/python -m workflow.pipeline rollout \
  --settings /mnt/public/tgy/VLA-Mender/vla-mender/src/run/pre_repair/experiment.example.yaml \
  --output /mnt/public/tgy/VLA-Mender/vla-mender/src/run/pre_repair
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
PYTHONPATH=vla-mender/src /absolute/path/to/VLA-Mender/.venv-openpi/bin/python -m workflow.pipeline diagnose \
  --settings /mnt/public/tgy/VLA-Mender/vla-mender/src/run/pre_repair/experiment.example.yaml \
  --output /mnt/public/tgy/VLA-Mender/vla-mender/src/run/pre_repair
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
3. Identify `first_causal_frame_index`: the frame where the causal failure
   result is most clearly observable from public evidence. This is the failure
   window's endpoint, not the first subtle deviation and not merely the timeout
   frame.
4. Identify `recoverable_window_start_frame_index`: the earliest frame where
   the trajectory begins a sustained, observable deviation from phase-aligned
   successful references. A small out-of-distribution deviation is expected,
   but intervention at this exact state must still be able to complete the task
   directly, without rolling back or replaying an earlier prefix. Compare both
   camera views, contact/engagement, task phase, and public state/action motion
   against successful trajectories. Reject a start frame that is already a
   gross outlier or has already lost the contact path needed for direct
   completion.
5. Define one inclusive failure window
   `[recoverable_window_start_frame_index,
   recoverable_window_stop_frame_index]`, and require
   `recoverable_window_stop_frame_index == first_causal_frame_index`. The
   window therefore runs from the earliest directly recoverable deviation to
   the clearest causal failure result. Do not default to the whole trajectory.
6. Require an evidence-derived failure window for each episode. Its index span
   must be strictly greater than 10: `window_stop - window_start > 10`
   (equivalently, at least 11 frame intervals). A window such as `[0, 10]`
   is therefore invalid. Do not reuse one fixed start/stop interval across all
   episodes; each episode must have its own window supported by its public
   video, state, and action evidence.
7. Record short evidence statements that are directly observable in the
   videos, state, or executed actions. Lower confidence when evidence is
   ambiguous.

Use zero-based frame indices and require
`0 <= start < stop == causal < num_frames`. A timeout alone is not a causal
frame when the failure mechanism becomes clearly observable earlier.

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
      "first_causal_frame_index": 102,
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
PYTHONPATH=vla-mender/src /absolute/path/to/VLA-Mender/.venv-openpi/bin/python -m workflow.pipeline materialize \
  --settings /mnt/public/tgy/VLA-Mender/vla-mender/src/run/pre_repair/experiment.example.yaml \
  --output /mnt/public/tgy/VLA-Mender/vla-mender/src/run/pre_repair \
  --diagnosis /mnt/public/tgy/VLA-Mender/vla-mender/src/run/pre_repair/failure_diagnosis/diagnosis.json
```

This stage applies the configured stride/count, replays each selected action
prefix, verifies public-state agreement, applies `preserve_full_state`, and
writes reset candidates, replay verification, private reset states, and repair
jobs. The diagnosis agent does not access or select private reset payloads.

Reset candidates are the earliest stride-aligned frames beginning at
`recoverable_window_start_frame_index` (for example `start`, `start + stride`,
...). They intentionally stay near the first directly recoverable deviation;
the causal/window-stop frame documents the clearest failure result and is not
the preferred intervention point.

The pre-repair workflow ends with the rollout artifacts, public diagnosis,
clustered failure modes, validated windows, and optional replay-verified reset
bank. Do not continue into repair-policy execution or model training.
