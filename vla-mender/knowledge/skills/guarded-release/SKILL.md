---
name: vlamender-guarded-release
description: Open only in a release phase with current object-target geometry confirming readiness.
---

# Guarded Release

Every release handler must use:

```python
result = guarded_open_gripper(obs, object_prompts, target_prompts, phase_id)
```

Advance only when status is `opened` or `already_released`. Any `blocked_*` result preserves the
current phase or enters `ambiguous_hold`.

The API requires an explicit release phase, held-object evidence near the EEF, and current geometric
alignment over the semantic target. It fails closed on missing or conflicting perception.
Broad target masks, insufficient footprint containment, and an unheld object away from the target
all return `blocked_*`; `vertical_contact_ready` and Z clearance remain diagnostics and do not block
release. `already_released` requires visible object-on-target geometry rather than an open gripper
alone.

The runtime guard uses a fixed total XY release limit of 0.025 m between the live held-object
estimate and the committed target. Read `placement_geometry.xy_distance` and `xy_limit` for the
decision evidence; do not recreate or bypass this check in generated policy code.
