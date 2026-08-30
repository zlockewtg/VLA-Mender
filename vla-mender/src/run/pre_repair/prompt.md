# VLA pre-repair task workflow

You are the coordinator for one VLA/LIBERO pre-repair run. Complete the full
pre-repair workflow in this prompt: execute or verify the VLA rollout, preserve
all successful and failed trajectories, prepare public evidence, diagnose each
failure, cluster failure modes, and define recoverable failure windows. Stop
only after the validated diagnosis has been materialized into a complete,
replay-verified `repair_handoff/` bundle. Producing `diagnosis.json` alone is
not completion. Do not train a model, synthesize repair code, or execute a
repair policy.

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
- Reset candidate selection: `{{RESET_CANDIDATE_SELECTION}}`
{{RESET_PREVENTION_CONTRACT}}
- Reset candidates per failure: {{RESET_CANDIDATE_CONTRACT}}
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

Phase-align references by observable milestones rather than absolute frame
number, because randomized scenes can shift phase timing. For every failed
episode, cite the successful episode indices used for comparison and state the
specific observable transition that remains normal in the reference but first
becomes abnormal in the failure.

## Stage 3 — diagnose each failed trajectory

Apply these two evidence gates before assigning a failure phase or failure
window:

1. **Causal-prerequisite gate.** Before assigning a failure phase, verify from
   public evidence that every prerequisite of that phase was observably
   completed. Diagnose the first unsupported transition in the task's causal
   sequence. A downstream phase cannot be causal when an earlier prerequisite
   was never satisfied. For example, later transport, alignment, placement, or
   release motion is not evidence of failure in those phases unless the required
   object or mechanism was first observably engaged and remained engaged up to
   that phase. End-effector motion, gripper commands, and arrival at a downstream
   location are not by themselves proof that the prerequisite interaction
   occurred.
2. **Evidence-resolution gate.** Thumbnails, sparse montages, and contact sheets
   may be used only to locate candidate events. Any semantic claim about target
   identity, contact, attachment, retention, release, containment, or task state
   must be confirmed using the original-resolution neighboring frames in every
   relevant camera view. Inspect enough frames before and after the candidate
   event to establish temporal continuity. If the available view does not
   resolve the required fact, acquire or inspect better public evidence and
   lower confidence instead of assigning a label from robot motion alone.

For every failed episode, in episode-index order:

1. Inspect the sparse timeline and both camera views.
2. Construct the observable prerequisite sequence for the episode, verify it
   chronologically, and identify the first transition whose required result is
   not supported by public evidence. Explicitly track the relevant task entity
   or interaction across that transition; do not substitute robot motion for
   evidence that the interaction succeeded.
3. Identify the semantic `failure_phase`, such as approach, alignment,
   grasp/contact acquisition, manipulation, transport, release, recovery, or
   timeout, from that earliest unsupported transition. Use the most specific
   phase supported by public evidence.
4. Identify `first_causal_frame_index`: the first frame in the failed subtask
   phase where the executed trajectory becomes observably wrong or departs from
   phase-aligned successful references. This is the failure window's endpoint.
   Select the onset of the erroneous transition, not a later frame where its
   consequence is merely clearer, not a downstream collision/rebound, and not
   the timeout frame.
   Require a consecutive boundary: frame `first_causal_frame_index - 1` must
   still be observably consistent with the cited, phase-aligned successful
   references, while `first_causal_frame_index` is the first frame that is not.
5. Identify `recoverable_window_start_frame_index`: the earliest directly
   correctable frame **inside the same semantic subtask phase that fails**. The
   failed phase's observable prerequisites must already be satisfied, and the
   phase itself must already have begun, but the erroneous transition must not
   yet have occurred. Do not move the start backward into the preceding phase
   to enlarge the window. For example, if pick succeeds and transport fails,
   the start must be the earliest recoverable transport frame after the pick is
   observably complete; it must not be a pick or grasp-acquisition frame. If
   placement fails, the start must be the earliest recoverable placement-phase
   frame, not a generic earlier transport frame. Compare both camera views,
   contact/engagement, task phase, and public state/action motion against
   phase-aligned successful trajectories.
6. Define one inclusive failure window
   `[recoverable_window_start_frame_index,
   recoverable_window_stop_frame_index]`, and require
   `recoverable_window_stop_frame_index == first_causal_frame_index`. The
   window therefore runs from the earliest correctable state in the failed
   subtask phase to the first observable erroneous transition in that same
   phase. Do not default to the whole trajectory or cross a subtask-phase
   boundary.
7. Require an evidence-derived failure window for each episode. Do not impose a
   fixed minimum span and do not reuse one fixed start/stop interval across all
   episodes. Semantic phase boundaries and original-resolution public evidence,
   rather than a target frame count, determine each episode's window.
8. Record short evidence statements that are directly observable in the
   original-resolution videos, state, or executed actions. The evidence must
   state which prerequisite was last confirmed, where the failed subtask phase
   begins, which transition first failed, and what is visibly wrong at the
   causal/window-stop frame. Lower confidence when evidence is ambiguous.
9. Record `successful_reference_episode_indices` and
   `successful_reference_comparison` for every failure. The comparison must
   name the last reference-consistent observation, the first divergent
   observation, and why the cited successful trajectory supports that boundary.
10. When reset selection is `failed_stage_entry_only`, also record
   `intervention_stage` and `intervention_stage_start_frame_index`. Use the
   broad behavior stage requested by the experiment, not merely the narrow
   contact subphase where the consequence becomes visible. The stage-entry
   frame must be derived from temporal evidence and precede the causal frame.
11. When reset selection is `per_episode_stage_entry_only`, follow the complete
   task-independent stage-discovery and entry-localization contract below.

{{PER_EPISODE_STAGE_DISCOVERY}}

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
  "observable_stage_graph": [
    {
      "stage_name": "task-specific stage name",
      "observable_entry_condition": "observable transition into the stage",
      "observable_exit_condition": "observable condition completing the stage",
      "required_prerequisites": ["observable prerequisite"],
      "relevant_entities": ["task entity"],
      "supporting_successful_episode_indices": [0, 1]
    }
  ],
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
      "successful_reference_episode_indices": [0],
      "successful_reference_comparison": "frame 101 remains phase-aligned with successful episode 0; frame 102 is the first observable divergence",
      "intervention_stage": "task-specific behavior stage",
      "intervention_stage_start_frame_index": 60,
      "stage_entry_evidence": {
        "preceding_stage": "task-specific preceding stage",
        "inspected_frame_indices": [57, 58, 59, 60, 61, 62, 63],
        "camera_evidence": {
          "wide": "failure_diagnosis/stage_boundary_evidence/episode_000003_wide_000060.png",
          "wrist": "failure_diagnosis/stage_boundary_evidence/episode_000003_wrist_000060.png"
        },
        "contact_sheet": "failure_diagnosis/stage_boundary_evidence/episode_000003_000060.png",
        "observable_transition": "observable task-specific stage transition at frame 60",
        "state_action_transition": "public state/action evidence for the change-point",
        "persistence_evidence": "the new behavior persists after frame 60",
        "successful_reference_episode_indices": [0, 1],
        "why_previous_frame_is_too_early": "frame 59 still belongs to the preceding stage",
        "why_later_frame_is_too_late": "the new stage is already active after frame 60"
      },
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

## Stage 6 — required reset-bank materialization and repair handoff

This stage is mandatory. After `diagnosis.json` is complete and schema-valid,
the trusted coordinator must run:

```bash
PYTHONPATH=vla-mender/src {{OPENPI_ENVIRONMENT}}/bin/python -m workflow.pipeline materialize \
  --settings {{SETTINGS_PATH}} \
  --output {{OUTPUT_DIR}} \
  --diagnosis {{OUTPUT_DIR}}/failure_diagnosis/diagnosis.json
```

{{RESET_MATERIALIZATION_POLICY}}

The required final layout is:

```text
{{OUTPUT_DIR}}/repair_handoff/
├── manifest.json
├── private_reset_states/
│   └── episode_<episode>_frame_<frame>.npz
└── agent_views/
    └── episode_<episode>_frame_<frame>.png
```

`manifest.json` is the single JSON contract consumed by repair. It merges the
validated diagnosis, reset selection, repair jobs, public reset metadata, and
replay-verification reports; do not split those into `repair_jobs.json`,
`public_reset_bank.json`, or `replay_verification.json`. It has this schema:

```json
{
  "schema_version": 1,
  "artifact_type": "vla_mender.repair_handoff",
  "complete": true,
  "settings_fingerprint": "exact experiment fingerprint",
  "source": {
    "run_root": "absolute pre-repair run path",
    "rollout_dir": "absolute rollout path",
    "diagnosis_working_file": "absolute diagnosis.json path",
    "rollout_settings_fingerprint": "fingerprint recorded by the frozen rollout",
    "reused_frozen_rollout": false
  },
  "diagnosis": {
    "schema_version": 1,
    "successful_reference_episodes": [
      {"episode_index": 0, "reason": "observable successful phase reference"}
    ],
    "failure_modes": [
      {
        "failure_mode_id": "FM-01",
        "label": "short root-cause label",
        "category": "alignment",
        "episode_indices": [3],
        "frequency": 1,
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
        "confidence": 0.8
      }
    ]
  },
{{RESET_SELECTION_JSON}}
  "summary": {
    "failure_episode_count": 1,
    "failure_mode_count": 1,
    "reset_count": 3,
    "replay_verified_count": 3,
    "all_replays_verified": true
  },
  "resets": [
    {
      "job_id": "e000003-f000078",
      "episode_index": 3,
      "candidate_rank": 0,
      "intervention_point": "window_start",
      "requested_frame_index": 78,
      "reset_frame_index": 78,
      "failure_phase": "approach/alignment",
      "failure_mode_id": "FM-01",
      "failure_category": "alignment",
      "failure_mode": "short root-cause label",
      "window_start": 78,
      "window_stop": 102,
      "verified": true,
      "replayed_action_count": 78,
      "max_public_state_error": 0.0,
      "public_tolerance": 0.0001,
      "source_control_space": "osc",
      "target_control_space": "osc",
      "reset_dynamics": "preserve_full_state",
      "dynamics_audit": {},
      "private_state_sha256": "reset-state content hash",
      "reset_state": "private_reset_states/episode_000003_frame_000078.npz",
      "reset_state_file_sha256": "npz file hash",
      "agent_view": "agent_views/episode_000003_frame_000078.png",
      "agent_view_sha256": "visible image content hash",
      "agent_view_file_sha256": "png file hash"
    }
  ]
}
```

The numeric values above are examples; the materializer must use this resolved
experiment's configured reset selection, spaces, dynamics, errors, and hashes.
Every failed episode must remain present in `diagnosis.episodes`. Every
published reset must have a unique `job_id`, match one of the intervention
points declared by `selection`, have `verified=true`, and reference files
beneath `repair_handoff/`. All reset and view files must exist and match their
declared file hashes.
When a frozen rollout is reused after changing only reset settings,
`rollout_settings_fingerprint` must retain the source rollout fingerprint and
`reused_frozen_rollout` must be true; any rollout-affecting contract drift is
forbidden.

Before declaring pre-repair complete, read back `manifest.json` and verify all
of the following:

1. `artifact_type`, `schema_version`, `complete`, and `settings_fingerprint`
   match this experiment.
2. {{RESET_COMPLETION_REQUIREMENT}}
3. `summary.reset_count == summary.replay_verified_count == len(resets)` and
   `summary.all_replays_verified` is true.
4. Every reset's `.npz` and `.png` exists below `repair_handoff/`, its hashes
   match, and replay error is within `public_tolerance`.
5. No failed replay, incomplete candidate set, missing attachment, schema
   mismatch, or infrastructure exception is represented as a task failure or
   silently skipped. Fix the cause and rerun materialization into a fresh,
   non-conflicting handoff directory.

The pre-repair workflow ends with the rollout artifacts, public diagnosis,
clustered failure modes, validated windows, and the complete replay-verified
`repair_handoff/` bundle. Do not stop at failure diagnosis, and do not continue
into repair-policy execution or model training.
