---
name: vlamender-grasp-state-estimation
description: Estimate held, not-held, or unknown from gripper and wide/wrist object geometry.
---

# Grasp-State Estimation

Call:

```python
grasp = estimate_grasp_state(obs, object_prompts)
```

The estimator combines the validated gripper aperture with SAM3/depth geometry near the EEF in the
agent and wrist cameras. SAM3 candidates below the runtime confidence floor are ignored. A fully
open Franka parallel-jaw gripper is `not_held`, even when it is positioned beside the source object;
do not replace the complete estimator with an ad-hoc width threshold.

- `held`: preserve the grasp prefix and route to lift/transport/alignment.
- `not_held`: acquisition may be attempted through `grasp_if_unheld`.
- `unknown`: enter `ambiguous_hold`; do not move, open, close, or seek another instance.

Internally detected masks are state-estimation evidence only. A manipulation target still requires a
normal SAM3 selection and `commit_target_mask`.
