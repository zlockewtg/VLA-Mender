# Repair strategy: LIBERO-90 task 21 stove and frying pan

## Exact identity and evidence boundary

- Suite/task: `libero_90:21`
- Language: `turn on the stove and put the frying pan on it`
- Diagnosed mode: `FM-01`, post-stove transition / pan-grasp acquisition
- Baseline evidence: 0/50 over scene-model seeds 100000--100049
- Prepared handoff frames: per-trajectory frames 97--138, median 113
- Best observed program: SHA-256
  `6b82861de4f51d6a551ba6f9e8584d000d7c743d0dc6491512e53d343d156315`
- Best observed coverage: 21/30 validation; a later debug subset was 18/20
- Status: partial. Do not describe this strategy as 90% generalization and do
  not publish failed suffixes as demonstrations.

The prepared `stove_pan_scene3_repair_v2` campaign has no completed candidate
results as of this reference. Seed a new campaign from the canonical program
only after verifying its SHA, then use the v2 smoke/video/expansion workflow.

## Observable entry gate

Take over only at the prepared per-trajectory reset for `FM-01`, when the prefix
has reached the post-stove transition and the pan has not been effectively
picked up. Do not replace the verified handoff with one fixed frame. Before any
motion, require usable native wide RGB/depth and camera calibration. Abstain or
stop the candidate when the pan, its actionable handle region, or the target
burner cannot be localized consistently.

The prefix is expected to have attempted or completed stove activation. Treat
the stove state as a task condition to preserve/verify, not as permission to
blindly repeat a toggle-like knob motion.

## Best control structure

### 1. Commit task-relevant instances

1. Segment the frying pan with multiple synonymous prompts such as `frying
   pan`, `pan`, and `skillet`.
2. Split actionable geometry into a pan-body center and a handle tail. Favor a
   low-Z, elongated handle attached to the pan body; reject high-Z candidates
   from the moka pot or stove hardware.
3. Segment and commit the intended burner/placement mask. Estimate a grounded
   target center and top height from its depth points.
4. Re-observe the handle in the wrist camera before grasping. Derive the
   top-down gripper yaw from the handle's long axis instead of forcing one yaw
   for every scene.

### 2. Guarded handle acquisition

1. Move through a collision-clear pregrasp above the handle, then descend to a
   handle-centerline grasp.
2. Use `grasp_if_unheld` or the current pinned API's equivalent semantic guard.
3. Inspect grasp status and gripper aperture. The canonical program treats
   normalized width below `0.140` as a weak handle grasp and retries from a
   higher/centered pose.
4. Allow only bounded, mechanism-specific retries. A Cartesian convergence
   failure or repeated false handle localization must return a policy failure,
   not trigger an unbounded workspace search.

### 3. Lift, transport, and live correction

1. Lift in stages to approximately `z=0.46` before translating across the
   workspace.
2. Preserve the observed grasp-to-pan-body offset. Compute the gripper's burner
   target from the desired pan-body center rather than placing the grasp point
   directly over the burner.
3. When the wide view still localizes the carried pan, re-estimate its body
   center and apply one bounded XY correction at the safe carry height.
4. If the pan is visibly lost, stop. Do not continue to the release phase with
   only commanded-pose evidence.

### 4. Placement, release, and stove condition

1. Descend in stages over the committed burner mask.
2. Use guarded release and inspect object-target geometry. Permit at most two
   bounded XY/Z corrections when release is blocked.
3. Once the evaluator-relevant placement is visibly achieved, avoid extra
   motion that can destabilize the pan.
4. Actuate the stove knob only when current observable evidence says stove
   activation is still missing. The canonical reference calls its knob routine
   after a successful release; this branch must be re-audited for the current
   reset rather than copied blindly.

## Residual mechanisms to target next

Prioritize changes in this order, one mechanism per candidate:

1. grasp pose reachability / Cartesian convergence near the handle;
2. wrist false-handle selection and weak, off-center grasp;
3. loss of the pan during high transport;
4. incorrect pan-body-to-burner offset or blocked release;
5. only then, conditional stove-state completion.

Inspect both wide and wrist videos for every failed seed before choosing which
mechanism applies. Record reset or service failures separately from policy
failures.

## Rejected shortcuts

- Fixed global XY/Z grasp offsets: the best tested offset variant solved only
  1/4; the other three variants solved 0/4.
- Pure push-based pan movement: tested batches solved 0/3 and 0/1.
- One forced gripper yaw for every scene.
- Unconditional knob actuation after a prefix that already activated the stove.
- Including failed, truncated, weak-grasp, or blocked-release suffixes in a
  training dataset.

## Canonical artifacts

- Stable knowledge snapshot:
  `programs/libero90_task21_stove_pan_best_observed.py`
- Program:
  `/mnt/public/tgy/VLA-Mender/outputs/stove_pan_scene3_repair/program_store/6b82861de4f51d6a551ba6f9e8584d000d7c743d0dc6491512e53d343d156315/repair_program.py`
- Validation:
  `/mnt/public/tgy/VLA-Mender/outputs/stove_pan_scene3_repair/fm01_validation_summary.json`
- Debug subset:
  `/mnt/public/tgy/VLA-Mender/outputs/stove_pan_scene3_repair/fm01_debug_summary.json`
- Prepared v2 campaign:
  `/mnt/public/tgy/VLA-Mender/outputs/stove_pan_scene3_repair_v2/repair_resolved.yaml`
- Prepared reset inventory:
  `/mnt/public/tgy/VLA-Mender/outputs/stove_pan_scene3_repair_v2/repair_resolved.yaml`
