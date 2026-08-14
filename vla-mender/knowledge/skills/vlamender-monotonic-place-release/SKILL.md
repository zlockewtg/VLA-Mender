---
name: vlamender-monotonic-place-release
description: Stabilize a VLA-Mender placement suffix when a held object drifts in XY during descent, moves upward or backward, switches IK branches between nearby poses, repeats placement waypoints, or reaches guarded release with a small residual. Use for observation-grounded placement onto an open horizontal support; route containers, insertion, pushing, and collision-constrained approaches to their specialized motion patterns.
---

# Monotonic Place and Release

Preserve the completed grasp and replace a noisy placement suffix with four bounded phases:
high alignment, locked-XY descent, guarded release, and fail-closed hold.

## Preconditions

Before issuing motion:

1. Call `get_robot_state(obs)` and `estimate_grasp_state(obs, object_prompts)`.
2. Continue only when grasp state is `held`; keep `unknown` in `ambiguous_hold`.
3. Select the semantic placement instance, call `commit_target_mask`, then call
   `ground_placement_target` before the held object occludes it.
4. Use only current observation geometry. Never embed a seed, reset identifier, fixed world pose,
   fixed joint vector, or action replay.

## Choose the motion family

Use this skill as the default for a held rigid object above an open horizontal support.

- For a walled container or insertion, preserve rim/clearance and insertion waypoints.
- For pushing or sweeping, use a contact-aware Cartesian interpolation pattern.
- For a demonstrated collision constraint, preserve the minimum clearance waypoint.
- For a measured controller stall, split only the failing segment while retaining locked XY and a
  continuous IK reference.
- Use pre-probe or joint replay only after traces show a repeatable far-workspace branch stall.

Do not add waypoints solely because the requested Z change exceeds a fixed distance.

## Phase 1: Align high

Move to a collision-safe pose above the committed target without first lifting higher unless current
geometry requires clearance. After orientation and high pose settle:

1. Re-observe the held object.
2. Compute the object-to-target XY residual from fresh geometry.
3. Clamp each correction to a bounded, observation-relative step.
4. Correct only at the safe high pose and re-observe after each motion.
5. Stop when the residual is comfortably inside the release gate; bound the number of corrections.

Do not convert one noisy estimate into an unbounded world-coordinate offset. Do not alternate between
independently solved high and low Cartesian targets.

## Phase 2: Lock XY and descend once

Read `get_robot_state(obs)["motion_target_position"]` after high alignment. Save its XY and keep the
current settled orientation. Derive release Z from current target/object geometry, then issue one
descending target:

```python
robot = get_robot_state(obs)
locked_xy = np.asarray(robot["motion_target_position"], dtype=np.float64)[:2].copy()
goto_pose(
    np.array([locked_xy[0], locked_xy[1], release_z], dtype=np.float64),
    np.asarray(robot["eef_quaternion_wxyz"], dtype=np.float64),
    z_approach=0.0,
)
```

Never feed `eef_position` back into `goto_pose`; it is the observed panda-hand link frame, whereas
`motion_target_position` is the TCP frame accepted by the controller. Keep Z non-increasing within
this phase. If the single command genuinely fails to converge, insert the fewest descending Z
targets needed, all with the same locked XY and orientation.

## Phase 3: Release through the guard

Call:

```python
result = guarded_open_gripper(
    obs,
    object_prompts,
    target_prompts,
    "guarded_release",
    target_commit=target,
)
```

Advance only for `opened` or `already_released`. Never claim completion from a reward flag or an open
gripper alone.

Treat the returned `placement_geometry` as the authoritative release diagnostic. The runtime guard
uses a configured total `xy_limit` of 0.025 m between the live held-object estimate and committed
target geometry. `base_xy_limit` remains a geometry diagnostic and `xy_relaxation` reports the
additional allowance up to the configured total. Never recreate the overlap test in generated
policy code or bypass the guard. Never hard-code a threshold copied from one rollout in generated
policy code. Align comfortably inside the gate; use the returned values to explain a decision, not
to replace it.

If the result is blocked, allow at most one correction only when all of the following hold:

- the object is still `held`;
- the residual is lateral and bounded;
- the correction can stay at the current `motion_target_position[2]`.

`vertical_contact_ready` and Z clearance are diagnostic only and do not block guarded release.

Retry the guard once after that constant-height correction. Otherwise enter `failed_safe_hold` and
preserve the gripper command. Never restart high alignment or replay the whole placement suffix.

After `opened` or `already_released`, collect a bounded number of passive settle observations and
finish by default. Do not add a fixed-distance post-release retreat merely because release
succeeded: that motion can dominate upward travel, lengthen the joint path, and disturb an already
valid placement. Add a retreat only when fresh observations show continued gripper contact,
dragging/collision risk, or the task explicitly requires clearance. Model it as a separate minimal
motion phase and verify that the placed object remains stable.

## Validated refinement evidence

The terminal rule above is supported by one LIBERO Spatial reset-suffix comparison on seed 009. Both
candidates completed with reward 1 and an executed guarded release. Removing only the fixed 0.10 m
post-release retreat reduced motion calls from 5 to 4, simulator trajectory steps from 257 to 187,
EEF path length from 0.415 m to 0.307 m, upward travel from 0.124 m to 0.0155 m, and EEF direction
reversals from 2 to 1.

The same experiment also showed why XY tolerance belongs to the runtime guard rather than generated
policy code. The guard owns the configured threshold and reports the measured residual, so a policy
can respond to a block without embedding rollout-specific constants or bypassing release safety.

This is single-reset evidence. It validates the terminal-motion reduction and guard routing, but it
does not establish cross-seed success or solve joint-space smoothness in general. Preserve the
successful-parent comparison and revalidate on different objects, support geometries, and reset
states before promoting task-specific waypoint parameters.

## Verify the observed trajectory

Judge motion from achieved robot observations, not raw Cartesian requests. Compare the candidate with
its successful parent using:

- task completion and `guarded_open_gripper.executed` as invariants;
- final-descent upward travel;
- endpoint and maximum transient XY departure during descent;
- EEF direction reversals;
- maximum joint-space jump and total joint path;
- environment motion steps and user-invoked motion calls.

Reject a shorter trajectory if it loses real release or task success. Do not post-process recorded
actions with an unvalidated smoothing filter; replay any controller or trajectory-filter change and
recheck contact geometry.

## Anti-patterns

- Replaying `high -> mid -> target` after a blocked release.
- Correcting XY at contact height when the same correction was available at the high pose.
- Re-segmenting an occluded placement target instead of using the committed handle.
- Repeatedly solving adjacent poses from unrelated or stale joint seeds.
- Treating a multi-waypoint descent as protection against an unfixed IK branch discontinuity.
- Hard-coding a rollout's measured XY gate instead of using `guarded_open_gripper`.
- Adding an unconditional fixed-distance retreat after a successful release.
- Opening directly, retrying raw open calls, or retreating before guarded release succeeds.
