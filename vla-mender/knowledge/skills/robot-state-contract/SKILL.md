---
name: vlamender-robot-state-contract
description: Use the versioned robot-state adapter instead of decoding LIBERO observation arrays.
---

# Robot State Contract

Call `state = get_robot_state(obs)` and use its named fields. Never access or index
`obs["robot_cartesian_pos"]` directly.

The returned fields are:

- `eef_position`
- `motion_target_position`
- `eef_quaternion_wxyz`
- `arm_joint_positions`
- `gripper_width_normalized`
- `gripper_aperture_state`
- `schema_version`

`eef_position` is the observed panda-hand link position. `motion_target_position` is the
corresponding TCP-frame position accepted by `goto_pose`; derive observation-relative waypoints from
the latter so the controller offset is not applied twice.

The LIBERO raw layout is XYZ, four quaternion values, then gripper width. Index 6 is a quaternion
component. The adapter validates shape, finiteness, quaternion normalization, and gripper range.
