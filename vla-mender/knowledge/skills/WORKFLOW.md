# VLA-Mender Skill Catalog Workflow

This catalog contains both runtime repair guidance and offline data-workflow guidance. Before a
repair experiment, inspect the manifest, search the catalog directly, and read every `SKILL.md`
plausibly relevant to the exact task, action needs, and failure mode. List the relevant files and
ideas in the prompt-required transient Markdown checklist. The checklist is not persisted or used
as a runtime gate. Skills document pinned contracts; they are not callables or snippets to copy
without adaptation.

The following rules apply to every reset-suffix manipulation skill selected with
`scope: repair_runtime`:

1. Decode observations only with `get_robot_state(obs)`.
2. Ground the task-language object and call `estimate_grasp_state(obs, object_prompts)` before phase
   routing or any grasp/release action.
3. Treat `unknown` as `ambiguous_hold`: preserve the current gripper command and issue no motion.
4. Acquire only through `grasp_if_unheld(...)`. `held` means the completed grasp prefix is preserved.
5. Release only through `guarded_open_gripper(...)` in the graph's release handler.
6. For open-support placement, align from fresh held-object geometry at a safe high pose, lock
   `motion_target_position` XY, descend monotonically, and never replay the whole placement suffix
   after a blocked release. Read `vlamender-monotonic-place-release` for the exception routing.
7. After `guarded_open_gripper` reports `opened` or `already_released`, settle with bounded passive
   observations and finish by default. Add a post-release retreat only when fresh observations or
   the task semantics demonstrate a clearance need.
8. Direct generated-code calls to `open_gripper()` and `close_gripper()` are forbidden in
   VLA-Mender runs.

Runtime API docstrings and `aspire/agent_workspace/api-reference.md` are authoritative. Skills may
choose task-language prompts and observation-relative poses, but may not change safety semantics.

Skills with `scope: dataset_pipeline` are selected only for offline dataset construction or
validation. They are not part of an online repair policy's callable API.
