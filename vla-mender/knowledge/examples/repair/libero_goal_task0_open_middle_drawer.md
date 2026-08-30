# Repair strategy: LIBERO Goal task 0 open the middle drawer

## Exact identity and evidence boundary

- Suite/task: `libero_goal:0`
- Language: `open the middle drawer of the cabinet`
- Randomized-50 baseline: 4/50
- Failure bank: 46 failed trajectories
- Canonical reference program SHA-256:
  `ea211e89a43df4b0faad728bf87a3ce2ea4294fd382934a5ab10d6dec5c87555`
- Code-expert evidence: 46/46 on the same 46 diagnosed repair resets
- Best confirmed trained checkpoint: step 1000, 48/50 under the official
  protocol

The 46/46 result is seen-reset repair coverage, not unseen generalization. The
randomized baseline and official checkpoint evaluation are different protocols
and must not be combined into one before/after rate.

## Diagnosed mechanisms and priority

| Mode | Count | Mechanism | Repair objective |
| --- | ---: | --- | --- |
| `FM-03` | 15/46 | Contact miss followed by runaway recovery | Stop, minimally retreat, reacquire once, then abstain. |
| `FM-01` | 13/46 | Excessive descent / cabinet collision | Restore handle-height clearance before contact. |
| `FM-02` | 10/46 | Shallow or lateral handle miss | Center on the middle handle and use guarded insertion. |
| `FM-04` | 8/46 | Near-handle stall / ineffective pull | Apply bounded outward pull while checking retained contact and progress. |

## Observable entry gate and handoff selection

Use the successful pregrasp phase, not one global frame and not a constant
offset from the eventual failure:

1. The reference successful pose occurs around frame 55.
2. For each failed trajectory, search the recoverable interval around frames
   45--65 for the public EEF-proxy state closest to that successful pose.
3. The prepared candidate frames are the designated match, `+5`, and `+10`.
4. Admit a reset only after exact prefix replay meets the configured public
   state tolerance. Use the saved reset inventory; do not recompute frames from
   a nearby trajectory.

At takeover there must still be room to localize all three handles, stage above
the cabinet, acquire the middle handle, and execute a reference-scale pull.

## Best control structure

### 1. Structural middle-handle localization

1. Require native `256x256` RGB/depth and the pinned camera geometry; fail
   closed on malformed inputs.
2. Query multiple handle/cabinet prompts in both wide and wrist views.
3. Form candidates from three vertically regular handle layers and choose the
   middle member of the best triplet. Use cross-view agreement to estimate the
   horizontal handle axis.
4. Derive the cabinet outward/front normal from the observed handle axis and
   the current EEF side. Do not encode a scene-specific simulator fixture pose.

### 2. Clearance-aware staging and acquisition

1. Normal staging: about `0.15 m` outward and `0.10 m` upward from the handle.
2. If that side is unreachable, flip the observed normal. The compact fallback
   stages about `0.08 m` outward and `0.06 m` upward.
3. Interpolate orientation smoothly before entering the handle corridor.
4. Move to a pregrasp about `0.06 m` outward, then approximately `0.015 m` into
   the handle. The canonical align70 reference uses a `-0.003 m` grasp-Z
   adjustment; nearby `-0.0015 m` and align100 variants also solved all 46 seen
   resets, so the exact Z offset is not established as uniquely best.
5. Use `grasp_if_unheld`. Reject empty closure or weak acquisition when the
   normalized gripper width is below `0.04`.

### 3. Bounded pull and progress gate

1. Normal branch: six outward steps of about `0.043 m` each.
2. Compact/adaptive branch: nine outward steps of about `0.025 m` each.
3. Keep the grasp quaternion and cabinet normal consistent during the pull.
4. Require observable outward EEF travel of at least `0.09 m` in the normal
   branch or `0.075 m` in the compact branch.
5. If contact is lost, the handle becomes ambiguous, or progress is below the
   gate, stop and report failure. Never reproduce the baseline's runaway
   recovery.

## Training-data strategy with the strongest downstream evidence

The best confirmed downstream result did not use every repair suffix
indiscriminately. It used a curated set of 30 successful repairs with their
matching VLA prefixes:

- trajectory composition: original prefix `[0, restart_frame)` plus valid real
  repair rows;
- 30 episodes, 1,578 prefix frames, 3,308 repair frames;
- native `256x256`, 20 Hz, action horizon 50;
- repair-only horizontal image correction recorded in the build manifest;
- no zero-arm loss mask, so gripper transitions remain supervised;
- native sampling was allowed to cross the splice only after this exact dataset
  passed continuity validation;
- training config used peak learning rate `2e-5`; step 1000 reached 48/50 under
  the official protocol.

Do not generalize cross-boundary sampling to a newly built dataset until strict
simulator signature, model identity, row-0 state/hash, image orientation,
runtime/media compatibility, and boundary visual continuity have all passed.

## Rejected shortcuts

- A fixed takeover frame for all scenes.
- Taking over exactly 35 frames before each causal failure; this can be far too
  late and drifted as late as frame 178 in the diagnosis.
- Selecting a handle from one mask without checking the three-layer cabinet
  structure.
- Retrying large recovery motions after the handle leaves both views.
- Treating 46/46 seen-reset code-expert coverage as unseen policy performance.
- Assuming that all 46 successful repair suffixes must outperform the curated
  30-example mixture during fine-tuning.

## Canonical artifacts

- Stable knowledge snapshot:
  `programs/libero_goal_task0_open_middle_drawer_align70.py`
- Reference program:
  `/mnt/public/tgy/capx-aspire/aspire/vla_mender/policies/task0_verified/open_middle_drawer_align70_v1.py`
- Failure diagnosis:
  `/mnt/public/tgy/data/libero_eval/libero_goal_task000_randomized50_20260810T052115Z/failure_diagnosis/diagnosis.json`
- 46/46 align70 batch:
  `/mnt/public/tgy/data/libero_eval/libero_goal_task000_randomized50_20260810T052115Z/failure_diagnosis/repair_observation_only/batch_numeric_strict79_repair46_align70_v1/batch_summary.json`
- Curated training dataset:
  `/mnt/public/tgy/datasets/pi0_libero_goal_task0_randomized50_vla_prefix_handle_alignment_v4_osc_repair46_basefm_lowest30_native256_fliprepair_guardpre20_post5_boundarymasked_20hz_20260811_v1`
- Dataset validation:
  `/mnt/public/tgy/datasets/pi0_libero_goal_task0_randomized50_vla_prefix_handle_alignment_v4_osc_repair46_basefm_lowest30_native256_fliprepair_guardpre20_post5_boundarymasked_20hz_20260811_v1/meta/validation_report.json`
- Best confirmed evaluation summary:
  `/mnt/public/tgy/data/libero_eval/open_middle_drawer_1000/summary.json`
