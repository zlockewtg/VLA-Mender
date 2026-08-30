# Workspace-verified repair strategy references

This directory records task-specific repair strategies that were derived from
local VLA-Mender campaigns and checked against their saved evaluator artifacts.
They are separate from the public NVIDIA Task Gallery in the parent directory.

This is the local repair-winner catalog selected by the parent
`knowledge/examples/manifest.json` index. Start with `manifest.json`, require an
exact suite/task/language match, and then
read the selected strategy card. A strategy is a parent/reference for a new
candidate, not permission to declare a seed repaired. The current repair
campaign must still follow its smoke, video-analysis, expansion, and promotion
workflow; evaluator success remains the only coverage signal.

`programs/` contains byte-identical snapshots of the currently selected source
programs so a later agent can inspect a stable parent even when an output tree
moves. The manifest records both the original artifact and snapshot hash.

## Available strategies

| ID | Exact task | Current evidence boundary |
| --- | --- | --- |
| `libero90-task21-stove-pan` | `libero_90:21` — turn on the stove and put the frying pan on it | Best observed program is partial: 21/30 validation and 18/20 debug subset. |
| `libero-goal-task0-open-middle-drawer` | `libero_goal:0` — open the middle drawer of the cabinet | Code expert solved 46/46 seen repair resets; its best confirmed trained checkpoint solved 48/50 official rollouts. |

## Use rules

1. Match `suite`, `task_id`, task language, failure phase, and observable entry
   gate. A nearby task or a different LIBERO suite is not an exact match.
2. Verify the referenced source program SHA-256 before borrowing it. Adapt calls
   only to the pinned runtime API and current public observations.
3. Never expose private simulator state to policy code. Re-localize task
   geometry from the current wide and wrist observations.
4. Preserve bounded motion, target-loss stops, contact/progress checks, and
   abstention branches. Do not keep retrying after the observable preconditions
   disappear.
5. Treat all reported coverage literally. Seen-reset success, debug-subset
   success, randomized evaluation, and official evaluation are different
   protocols and must not be merged into one rate.
6. When building training data, publish only successful, non-truncated repair
   rows after validating full simulator/model/media continuity at the splice.
   Robot state alone is not sufficient evidence of scene continuity.
