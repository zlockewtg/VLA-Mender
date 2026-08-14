---
name: vlamender-guarded-grasp-acquisition
description: Prevent re-grasping when a reset observation already contains a held object.
---

# Guarded Grasp Acquisition

Every acquisition handler must use:

```python
result = grasp_if_unheld(
    obs,
    object_prompts,
    pregrasp_position,
    grasp_position,
    grasp_quaternion_wxyz,
)
```

Interpret status as follows:

- `already_held`: issue no acquisition motion; advance to the appropriate held-object suffix.
- `ambiguous_hold`: preserve the gripper and reobserve locally.
- `grasped`: reobserve and classify the next phase.
- `grasp_unverified` or `motion_failed`: do not claim lift/transport completion.

Do not add a separate raw open or close command around this API.
