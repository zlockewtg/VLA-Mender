"""Native OSC control surface for the LIBERO reduced skill library."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from knowledge.api.env_protocol import BaseEnv
from knowledge.api.franka.common import apply_tcp_offset
from knowledge.api.franka.libero_reduced import CartesianPoseConvergenceError
from knowledge.api.franka.libero_reduced_skill_library import (
    FrankaLiberoApiReducedSkillLibrary,
)


def _normalized_quaternion_wxyz(value: np.ndarray, *, name: str) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.all(np.isfinite(quaternion)) or norm <= 1e-12:
        raise ValueError(f"{name} must be a finite, non-zero WXYZ quaternion")
    return quaternion / norm


def _osc_pose_error_action(
    current_position: np.ndarray,
    current_quaternion_wxyz: np.ndarray,
    target_position: np.ndarray,
    target_quaternion_wxyz: np.ndarray,
    gripper_action: float,
    *,
    gain: float = 1.0,
    position_action_scale_m: float = 0.05,
    rotation_action_scale_rad: float = 0.5,
) -> tuple[np.ndarray, float, float]:
    """Convert Cartesian feedback error to one normalized OSC action."""
    current_pos = np.asarray(current_position, dtype=np.float64).reshape(3)
    target_pos = np.asarray(target_position, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(current_pos)) or not np.all(np.isfinite(target_pos)):
        raise ValueError("current and target OSC positions must be finite")
    current_quat = _normalized_quaternion_wxyz(
        current_quaternion_wxyz, name="current_quaternion_wxyz"
    )
    target_quat = _normalized_quaternion_wxyz(
        target_quaternion_wxyz, name="target_quaternion_wxyz"
    )
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("OSC feedback gain must be a finite positive value")
    if not np.isfinite(position_action_scale_m) or position_action_scale_m <= 0.0:
        raise ValueError("position_action_scale_m must be finite and positive")
    if not np.isfinite(rotation_action_scale_rad) or rotation_action_scale_rad <= 0.0:
        raise ValueError("rotation_action_scale_rad must be finite and positive")
    if not np.isfinite(gripper_action):
        raise ValueError("gripper_action must be finite")

    current_rotation = SciRotation.from_quat(np.roll(current_quat, -1)).as_matrix()
    target_rotation = SciRotation.from_quat(np.roll(target_quat, -1)).as_matrix()
    position_error = target_pos - current_pos
    rotation_error = SciRotation.from_matrix(
        target_rotation @ current_rotation.T
    ).as_rotvec()
    command = np.concatenate(
        [
            np.clip(gain * position_error / position_action_scale_m, -1.0, 1.0),
            np.clip(gain * rotation_error / rotation_action_scale_rad, -1.0, 1.0),
            [float(np.clip(gripper_action, -1.0, 1.0))],
        ]
    )
    return (
        command,
        float(np.linalg.norm(position_error)),
        float(np.linalg.norm(rotation_error)),
    )


def _minimum_jerk(progress: float) -> float:
    """Quintic time scaling with zero endpoint velocity and acceleration."""
    value = float(np.clip(progress, 0.0, 1.0))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def _interpolate_pose_minimum_jerk(
    start_position: np.ndarray,
    start_quaternion_wxyz: np.ndarray,
    target_position: np.ndarray,
    target_quaternion_wxyz: np.ndarray,
    progress: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate Cartesian position and orientation with minimum-jerk timing."""
    scale = _minimum_jerk(progress)
    start_pos = np.asarray(start_position, dtype=np.float64).reshape(3)
    target_pos = np.asarray(target_position, dtype=np.float64).reshape(3)
    start_quat = _normalized_quaternion_wxyz(
        start_quaternion_wxyz, name="start_quaternion_wxyz"
    )
    target_quat = _normalized_quaternion_wxyz(
        target_quaternion_wxyz, name="target_quaternion_wxyz"
    )
    start_rotation = SciRotation.from_quat(np.roll(start_quat, -1))
    target_rotation = SciRotation.from_quat(np.roll(target_quat, -1))
    relative_rotvec = (target_rotation * start_rotation.inv()).as_rotvec()
    interpolated_rotation = SciRotation.from_rotvec(scale * relative_rotvec) * start_rotation
    interpolated_quaternion = np.roll(interpolated_rotation.as_quat(), 1)
    return (
        start_pos + scale * (target_pos - start_pos),
        _normalized_quaternion_wxyz(
            interpolated_quaternion, name="interpolated_quaternion_wxyz"
        ),
    )


def _slew_limit_arm_command(
    command: np.ndarray,
    previous_command: np.ndarray,
    max_delta_norm: float,
) -> np.ndarray:
    """Bound one-frame arm-command changes without changing the gripper command."""
    limited = np.asarray(command, dtype=np.float64).copy()
    previous = np.asarray(previous_command, dtype=np.float64).reshape(6)
    delta = limited[:6] - previous
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > max_delta_norm:
        limited[:6] = previous + delta * (max_delta_norm / delta_norm)
    return limited


def _gripper_action_from_observed_width(width_normalized: float) -> float:
    """Recover the hold direction after reset from public gripper state.

    ``_gripper_fraction`` is a controller latch and can be stale after a
    state-only handoff.  An intermediate observed aperture must be held closed
    because it is the normal signature of an object between the fingers.
    """
    width = float(width_normalized)
    if not np.isfinite(width) or width < -0.05 or width > 1.05:
        raise ValueError("observed gripper width must be finite and normalized")
    return (
        -1.0
        if width >= FrankaLiberoApiReducedSkillLibrary._VLAMENDER_GRIPPER_OPEN_THRESHOLD
        else 1.0
    )


class FrankaLiberoApiReducedOscSkillLibrary(FrankaLiberoApiReducedSkillLibrary):
    """Reduced LIBERO skill library backed by native operational-space control."""

    _OSC_POSITION_ACTION_SCALE_M = 0.05
    _OSC_ROTATION_ACTION_SCALE_RAD = 0.5
    _OSC_MAX_POSITION_COMMAND_GLOBAL_M = 0.05
    _OSC_MAX_POSITION_COMMAND_HANDLE_M = 0.025
    # Profile timing follows the observed LIBERO OSC bandwidth, not the
    # controller's nominal action scale.  Using 0.05 m / 0.5 rad as if the EEF
    # achieved them in one frame made the virtual target run far ahead, then
    # produced a long saturated chase and a visible corrective rebound.
    _OSC_PROFILE_POSITION_STEP_M = 0.030
    _OSC_PROFILE_ROTATION_STEP_RAD = 0.180
    _OSC_PROFILE_MIN_STEPS = 8
    # Supervision windows are a dataset concern: every boundary keeps the
    # surrounding +/-5 frames in the loss.  Holding every internal waypoint
    # at zero for five simulator frames made a continuous path visibly stop at
    # each corner.  One observed-stop frame is sufficient for a waypoint; the
    # following motion frames complete the post-boundary supervision window.
    _OSC_BOUNDARY_SUPERVISION_WINDOW_STEPS = 5
    _OSC_WAYPOINT_HOLD_STEPS = 1
    _OSC_LEGACY_CONVERGED_STEPS = 2
    _OSC_MAX_ARM_COMMAND_DELTA_NORM = 0.20
    _OSC_BRAKE_COMMAND_DECAY = 0.50
    _OSC_ZERO_COMMAND_ARM_NORM_THRESHOLD = 0.08
    # At 20 Hz these correspond to 10 mm/s translation and 0.2 rad/s
    # orientation.  They are below one normalized OSC command quantum and
    # tolerate the small contact vibration of a held pan.
    _OSC_STOP_POSITION_DELTA_M = 5.0e-4
    _OSC_STOP_ORIENTATION_DELTA_RAD = 1.0e-2
    _WORKSPACE_LOWER = np.array([-0.1, -0.5, 0.005], dtype=np.float64)
    _WORKSPACE_UPPER = np.array([0.75, 0.5, 0.9], dtype=np.float64)

    def __init__(self, env: BaseEnv) -> None:
        if not hasattr(env, "step_osc_pose"):
            raise TypeError(
                "FrankaLiberoApiReducedOscSkillLibrary requires "
                "an environment implementing step_osc_pose"
            )
        super().__init__(env, enable_ik=False)
        self._handle_alignment_active = False

    def functions(self) -> dict[str, Any]:
        fns = super().functions()
        fns["get_osc_controller_spec"] = self.get_osc_controller_spec
        fns["osc_step"] = self.osc_step
        fns["goto_pose_osc"] = self.goto_pose_osc
        fns["open_gripper_observed"] = self.open_gripper_observed
        fns["close_gripper_observed"] = self.close_gripper_observed
        fns["settle_task_completion_observed"] = (
            self.settle_task_completion_observed
        )
        fns["defer_task_completion_until_program_end"] = (
            self.defer_task_completion_until_program_end
        )
        # Compatibility for existing code-policy programs.  On this API the
        # familiar name is intentionally routed to native OSC, never IK.
        fns["goto_pose"] = self.goto_pose
        return fns

    def defer_task_completion_until_program_end(self) -> None:
        """Allow a code teacher to finish stop/release frames after first success."""
        defer = getattr(self._env, "defer_task_completion_until_program_end", None)
        if not callable(defer):
            raise RuntimeError("the environment cannot defer task-completion interruption")
        defer()

    def get_osc_controller_spec(self) -> dict[str, Any]:
        """Return the exact normalized action contract used by this runtime.

        Returns:
            A dictionary describing ``OSC_POSE`` action order, bounds, physical
            scaling and gripper signs.  The dispatched seven-dimensional
            action is the action saved by the rollout recorder; no joint-space
            waypoint or IK result is generated on this control path.
        """
        return {
            "controller": "OSC_POSE",
            "action_order": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
            "normalized_bounds": [-1.0, 1.0],
            "position_scale_m": self._OSC_POSITION_ACTION_SCALE_M,
            "max_position_command_global_m": self._OSC_MAX_POSITION_COMMAND_GLOBAL_M,
            "max_position_command_handle_m": self._OSC_MAX_POSITION_COMMAND_HANDLE_M,
            "rotation_scale_rad": self._OSC_ROTATION_ACTION_SCALE_RAD,
            "rotation_representation": "world_frame_rotation_vector",
            "gripper_open": -1.0,
            "gripper_close": 1.0,
            "saved_action": "normalized_action_dispatched_to_libero",
            "uses_inverse_kinematics": False,
            "trajectory_time_scaling": "minimum_jerk_quintic",
            "trajectory_profiles": [
                "minimum_jerk",
                "direct_slew",
                "legacy_direct",
            ],
            "minimum_profile_steps": self._OSC_PROFILE_MIN_STEPS,
            "deceleration_frames": "4-8",
            "boundary_supervision_window_steps": (
                self._OSC_BOUNDARY_SUPERVISION_WINDOW_STEPS
            ),
            "waypoint_zero_hold_steps": self._OSC_WAYPOINT_HOLD_STEPS,
            "zero_requires_observed_ee_stop": True,
        }

    def open_gripper_observed(self) -> dict[str, Any]:
        """Open with an observation-bounded wait instead of a fixed 30 frames."""
        self._env._set_gripper(1.0)
        stable_steps = 0
        executed_steps = 0
        last_width: float | None = None
        for _ in range(12):
            self._env._step_once()
            executed_steps += 1
            robot = self.get_robot_state(self.get_observation())
            last_width = float(robot["gripper_width_normalized"])
            stable_steps = stable_steps + 1 if robot["gripper_aperture_state"] == "open" else 0
            if executed_steps >= 3 and stable_steps >= 2:
                break
        audit = {
            "executed_steps": executed_steps,
            "maximum_steps": 12,
            "stable_steps": stable_steps,
            "last_width_normalized": last_width,
            "reason": "observed_open" if stable_steps >= 2 else "maximum_settle_fallback",
        }
        print("[vlamender_open_settle] " + repr(audit), flush=True)
        return audit

    def _vlamender_open_raw(self) -> None:
        """Route guarded releases through the observation-bounded OSC wait."""
        self.open_gripper_observed()

    def close_gripper_observed(self) -> dict[str, Any]:
        """Close with the existing observation-bounded 12-frame grasp wait."""
        self._vlamender_close_raw()
        return dict(getattr(self, "_vlamender_last_close_audit", {}))

    def settle_task_completion_observed(self, max_steps: int = 12) -> dict[str, Any]:
        """Hold the released pose only until the public task predicate succeeds."""
        if isinstance(max_steps, bool) or int(max_steps) <= 0 or int(max_steps) > 30:
            raise ValueError("max_steps must be an integer in [1, 30]")
        completed = False
        executed_steps = 0
        task_completed = getattr(self._env, "task_completed", None)
        if not callable(task_completed):
            raise RuntimeError("the environment cannot report task completion")
        self._env._set_gripper(1.0)
        for _ in range(int(max_steps)):
            self._env._step_once()
            executed_steps += 1
            if bool(task_completed()):
                completed = True
                break
        audit = {
            "task_completed": completed,
            "executed_steps": executed_steps,
            "maximum_steps": int(max_steps),
            "reason": "observed_task_completed" if completed else "maximum_settle_fallback",
        }
        print("[vlamender_task_settle] " + repr(audit), flush=True)
        return audit

    def osc_step(self, action: np.ndarray) -> dict[str, Any]:
        """Execute one bounded native OSC control frame.

        Prefer ``goto_pose_osc`` for normal motion.  Use this low-level API only
        for an explicitly bounded visual-servo correction.

        Args:
            action: Seven normalized values ``[dpos(3), drotvec(3), gripper]``.
                Translation ``1`` means 0.05 m, rotation ``1`` means 0.5 rad,
                gripper ``-1`` opens and ``+1`` closes.  Values are clipped to
                ``[-1, 1]`` before both execution and trajectory recording.

        Returns:
            A fresh public robot/camera observation after the control frame.
        """
        self._env.step_osc_pose(action)
        return self.get_observation()

    def goto_pose_osc(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
        *,
        gain: float = 1.0,
        max_steps: int = 120,
        hold_steps: int | None = None,
        position_tolerance_m: float | None = None,
        profile: str = "minimum_jerk",
        gripper_action: float | None = None,
    ) -> None:
        """Move to a Cartesian pose with closed-loop native OSC and no IK.

        A minimum-jerk virtual Cartesian target is tracked from the latest
        public robot observation.  The last half of its 8-or-more-frame timing
        curve supplies 4-or-more deceleration frames.  A zero arm command is
        emitted only after both Cartesian error and observation-derived EEF
        velocity are small.  Internal waypoints hold that zero for one frame;
        the dataset retains the five motion/braking frames on both sides of
        the boundary instead of turning the supervision window into a pause.

        Args:
            position: Three-value motion-target XYZ in the same frame returned
                by ``get_robot_state(obs)["motion_target_position"]``.
            quaternion_wxyz: Four-value world-frame WXYZ unit quaternion.
            z_approach: Optional non-negative world-Z approach offset in meters.
            gain: Positive Cartesian feedback gain.  The default is recommended.
            max_steps: Positive per-stage simulator-frame budget.
            hold_steps: Number of zero-command frames after the observed stop.
                Defaults to one for continuous waypoint transitions.
            position_tolerance_m: Optional observed terminal tolerance.  Contact
                waypoints may use a slightly wider bounded tolerance when the
                object physically prevents further free-space convergence.
            profile: ``"minimum_jerk"`` for free-space motion, or
                ``"direct_slew"`` for a contact insertion that needs sustained
                Cartesian authority, or ``"legacy_direct"`` for a narrowly
                scoped contact path whose evaluator-confirmed behavior depends
                on the historical direct feedback controller.  The first two
                profiles retain the same per-frame arm-command slew limit and
                observed-stop admission.
            gripper_action: Optional fixed command in ``[-1, 1]``.  When
                omitted, the command is inferred from observed aperture as
                before.  A near-zero value can preserve a transport-stage
                intermediate aperture without repeatedly squeezing an object.

        Returns:
            None.

        Raises:
            CartesianPoseConvergenceError: The observed EEF pose remains outside
                the same Cartesian tolerances used by the historical API.
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(pos)):
            raise ValueError("goto_pose_osc position must contain three finite values")
        quat = _normalized_quaternion_wxyz(
            quaternion_wxyz, name="goto_pose_osc quaternion_wxyz"
        )
        if not np.isfinite(z_approach) or z_approach < 0.0:
            raise ValueError("z_approach must be a finite non-negative value")
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be a positive integer")
        if profile not in {"minimum_jerk", "direct_slew", "legacy_direct"}:
            raise ValueError(
                "profile must be 'minimum_jerk', 'direct_slew', or 'legacy_direct'"
            )
        if gripper_action is not None and (
            not np.isfinite(gripper_action) or abs(float(gripper_action)) > 1.0
        ):
            raise ValueError("gripper_action must be finite and in [-1, 1]")
        if hold_steps is None:
            hold_steps = self._OSC_WAYPOINT_HOLD_STEPS
        if isinstance(hold_steps, bool) or int(hold_steps) <= 0:
            raise ValueError("hold_steps must be a positive integer")
        if position_tolerance_m is None:
            position_tolerance_m = self._GOTO_POSITION_TOLERANCE_M
        if (
            not np.isfinite(position_tolerance_m)
            or position_tolerance_m <= 0.0
            or position_tolerance_m > 0.05
        ):
            raise ValueError("position_tolerance_m must be in (0, 0.05]")

        if z_approach > 0.0:
            approach = pos.copy()
            approach[2] += float(z_approach)
            self._goto_pose_osc_stage(
                approach,
                quat,
                stage="approach",
                gain=gain,
                max_steps=int(max_steps),
                hold_steps=int(hold_steps),
                position_tolerance_m=float(position_tolerance_m),
                profile=profile,
                gripper_action=gripper_action,
            )
        self._goto_pose_osc_stage(
            pos,
            quat,
            stage="final",
            gain=gain,
            max_steps=int(max_steps),
            hold_steps=int(hold_steps),
            position_tolerance_m=float(position_tolerance_m),
            profile=profile,
            gripper_action=gripper_action,
        )

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
    ) -> None:
        """Compatibility alias for ``goto_pose_osc`` on the OSC runtime.

        This method never invokes ``solve_ik`` or ``move_to_joints``.  New OSC
        programs should call ``goto_pose_osc`` explicitly; the alias lets
        existing high-level skill programs migrate without code changes.
        """
        self.goto_pose_osc(position, quaternion_wxyz, z_approach=z_approach)

    def _vlamender_goto_grasp_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> None:
        """Continue a partial acquisition directly, without a new time-profile reset."""
        self.goto_pose_osc(
            position,
            quaternion_wxyz,
            profile="direct_slew",
            position_tolerance_m=self._VLAMENDER_GRASP_CONTACT_READY_TOLERANCE_M,
        )

    def _goto_pose_osc_stage(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        stage: str,
        gain: float,
        max_steps: int,
        hold_steps: int,
        position_tolerance_m: float,
        profile: str,
        gripper_action: float | None,
    ) -> None:
        commanded_position = np.clip(
            np.asarray(position, dtype=np.float64).reshape(3),
            self._WORKSPACE_LOWER,
            self._WORKSPACE_UPPER,
        )
        target_eef_position = apply_tcp_offset(
            commanded_position, quaternion_wxyz, self._TCP_OFFSET
        )
        start_observation = self.get_observation()
        start_cartesian = np.asarray(
            start_observation.get("robot_cartesian_pos", []), dtype=np.float64
        ).reshape(-1)
        if start_cartesian.size != 8 or not np.all(np.isfinite(start_cartesian)):
            raise RuntimeError(
                "goto_pose_osc requires robot_cartesian_pos as "
                "XYZ + quaternion_wxyz + gripper_width"
            )
        gripper_action = (
            _gripper_action_from_observed_width(start_cartesian[-1])
            if gripper_action is None
            else float(gripper_action)
        )
        if profile == "legacy_direct":
            self._goto_pose_osc_stage_legacy_direct(
                target_eef_position,
                quaternion_wxyz,
                gripper_action=gripper_action,
                stage=stage,
                gain=gain,
                max_steps=max_steps,
                position_tolerance_m=position_tolerance_m,
            )
            return
        start_position = start_cartesian[:3].copy()
        start_quaternion = _normalized_quaternion_wxyz(
            start_cartesian[3:7], name="observed_start_quaternion_wxyz"
        )
        start_rotation = SciRotation.from_quat(np.roll(start_quaternion, -1))
        target_rotation = SciRotation.from_quat(np.roll(quaternion_wxyz, -1))
        translation_distance = float(np.linalg.norm(target_eef_position - start_position))
        rotation_distance = float(
            np.linalg.norm((target_rotation * start_rotation.inv()).as_rotvec())
        )
        max_position_step = (
            self._OSC_MAX_POSITION_COMMAND_HANDLE_M
            if getattr(self, "_handle_alignment_active", False)
            else self._OSC_MAX_POSITION_COMMAND_GLOBAL_M
        )
        # The peak derivative of quintic minimum-jerk timing is 1.875.  This
        # duration is based on observed closed-loop bandwidth and reserves at
        # least four frames for the deceleration half.
        profile_steps = max(
            self._OSC_PROFILE_MIN_STEPS,
            int(
                np.ceil(
                    1.875 * translation_distance / self._OSC_PROFILE_POSITION_STEP_M
                )
            ),
            int(
                np.ceil(
                    1.875 * rotation_distance / self._OSC_PROFILE_ROTATION_STEP_RAD
                )
            ),
        )
        stable_steps = 0
        settle_started = False
        elapsed_steps = 0
        position_error = float("inf")
        orientation_error = float("inf")
        position_delta = float("inf")
        orientation_delta = float("inf")
        terminal_arm_norm = float("inf")
        previous_arm_command = np.zeros(6, dtype=np.float64)
        previous_observed_position = start_position.copy()
        previous_observed_quaternion = start_quaternion.copy()

        for step_index in range(max_steps):
            cartesian = np.asarray(
                self.get_observation().get("robot_cartesian_pos", []), dtype=np.float64
            ).reshape(-1)
            if cartesian.size != 8 or not np.all(np.isfinite(cartesian[:7])):
                raise RuntimeError(
                    "goto_pose_osc requires robot_cartesian_pos as "
                    "XYZ + quaternion_wxyz + gripper_width"
                )
            observed_quaternion = _normalized_quaternion_wxyz(
                cartesian[3:7], name="observed_quaternion_wxyz"
            )
            position_delta = float(np.linalg.norm(cartesian[:3] - previous_observed_position))
            previous_rotation = SciRotation.from_quat(np.roll(previous_observed_quaternion, -1))
            observed_rotation = SciRotation.from_quat(np.roll(observed_quaternion, -1))
            orientation_delta = float(
                np.linalg.norm((observed_rotation * previous_rotation.inv()).as_rotvec())
            )
            observed_stopped = (
                elapsed_steps > 0
                and position_delta <= self._OSC_STOP_POSITION_DELTA_M
                and orientation_delta <= self._OSC_STOP_ORIENTATION_DELTA_RAD
            )

            if profile == "direct_slew":
                progress = 1.0
                virtual_position = target_eef_position
                virtual_quaternion = quaternion_wxyz
            else:
                progress = min(1.0, float(step_index + 1) / float(profile_steps))
                virtual_position, virtual_quaternion = _interpolate_pose_minimum_jerk(
                    start_position,
                    start_quaternion,
                    target_eef_position,
                    quaternion_wxyz,
                    progress,
                )
            command, _, _ = _osc_pose_error_action(
                cartesian[:3],
                observed_quaternion,
                virtual_position,
                virtual_quaternion,
                gripper_action,
                gain=gain,
                position_action_scale_m=self._OSC_POSITION_ACTION_SCALE_M,
                rotation_action_scale_rad=self._OSC_ROTATION_ACTION_SCALE_RAD,
            )
            position_norm = float(np.linalg.norm(command[:3]))
            max_position_command = max_position_step / self._OSC_POSITION_ACTION_SCALE_M
            if position_norm > max_position_command:
                command[:3] *= max_position_command / position_norm
            command = _slew_limit_arm_command(
                command,
                previous_arm_command,
                self._OSC_MAX_ARM_COMMAND_DELTA_NORM,
            )

            # Evaluate the real terminal error, not merely the virtual profile
            # error.  A large residual command can never be replaced directly
            # by zero, even if the coarse pose tolerance is already satisfied.
            terminal_command, position_error, orientation_error = _osc_pose_error_action(
                cartesian[:3],
                observed_quaternion,
                target_eef_position,
                quaternion_wxyz,
                gripper_action,
                gain=gain,
                position_action_scale_m=self._OSC_POSITION_ACTION_SCALE_M,
                rotation_action_scale_rad=self._OSC_ROTATION_ACTION_SCALE_RAD,
            )
            terminal_arm_norm = float(np.linalg.norm(terminal_command[:6]))
            terminal_stable = (
                progress >= 1.0
                and position_error <= position_tolerance_m
                and orientation_error <= self._GOTO_ORIENTATION_TOLERANCE_RAD
                and float(np.linalg.norm(previous_arm_command)) <= 0.12
            )
            ready_for_zero = (
                terminal_stable
                and observed_stopped
                and float(np.linalg.norm(previous_arm_command))
                <= self._OSC_ZERO_COMMAND_ARM_NORM_THRESHOLD
            )
            if ready_for_zero:
                command[:6] = 0.0
                settle_started = True
            else:
                if progress >= 1.0:
                    if (
                        position_error <= position_tolerance_m
                        and orientation_error <= self._GOTO_ORIENTATION_TOLERANCE_RAD
                        # A grasp/contact waypoint can be physically stopped
                        # inside tolerance while static contact leaves a
                        # non-zero pose-error command.  Brake that observed
                        # stop instead of chasing contact until timeout.
                        and (terminal_arm_norm <= 0.20 or observed_stopped)
                    ):
                        # Once the target is genuinely close, decay the prior
                        # velocity command over roughly 5-7 frames.  Continuing
                        # full terminal feedback here creates a small limit
                        # cycle and eventually forces a timeout-to-zero jump.
                        command[:6] = _slew_limit_arm_command(
                            np.concatenate(
                                [
                                    self._OSC_BRAKE_COMMAND_DECAY
                                    * previous_arm_command,
                                    [gripper_action],
                                ]
                            ),
                            previous_arm_command,
                            self._OSC_MAX_ARM_COMMAND_DELTA_NORM,
                        )[:6]
                    else:
                        command = terminal_command
                        position_norm = float(np.linalg.norm(command[:3]))
                        if position_norm > max_position_command:
                            command[:3] *= max_position_command / position_norm
                        command = _slew_limit_arm_command(
                            command,
                            previous_arm_command,
                            self._OSC_MAX_ARM_COMMAND_DELTA_NORM,
                        )
            if settle_started and terminal_stable:
                stable_steps += 1
            elif settle_started:
                # The first zero was issued only after an observed stop.  Keep
                # the boundary window alive across a tiny corrective settle
                # frame, but restart if the pose actually leaves tolerance.
                if (
                    position_error > position_tolerance_m
                    or orientation_error > self._GOTO_ORIENTATION_TOLERANCE_RAD
                ):
                    settle_started = False
                    stable_steps = 0
                else:
                    stable_steps += 1
            else:
                stable_steps = 0
            self._env.step_osc_pose(command)
            elapsed_steps += 1
            previous_observed_position = cartesian[:3].copy()
            previous_observed_quaternion = observed_quaternion.copy()
            previous_arm_command = command[:6].copy()
            if stable_steps >= hold_steps:
                return

        raise CartesianPoseConvergenceError(
            "goto_pose_osc failed Cartesian convergence: "
            f"stage={stage} position_error_m={position_error:.6f} "
            f"position_tolerance_m={position_tolerance_m:.6f} "
            f"orientation_error_rad={orientation_error:.6f} "
            f"orientation_tolerance_rad={self._GOTO_ORIENTATION_TOLERANCE_RAD:.6f} "
            f"observed_position_delta_m={position_delta:.6f} "
            f"observed_orientation_delta_rad={orientation_delta:.6f} "
            f"previous_arm_command_norm={float(np.linalg.norm(previous_arm_command)):.6f} "
            f"terminal_arm_command_norm={terminal_arm_norm:.6f}"
        )

    def _goto_pose_osc_stage_legacy_direct(
        self,
        target_eef_position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        gripper_action: float,
        stage: str,
        gain: float,
        max_steps: int,
        position_tolerance_m: float,
    ) -> None:
        """Reproduce the historical direct OSC loop for a proven contact path."""
        consecutive = 0
        position_error = float("inf")
        orientation_error = float("inf")
        max_position_step = (
            self._OSC_MAX_POSITION_COMMAND_HANDLE_M
            if getattr(self, "_handle_alignment_active", False)
            else self._OSC_MAX_POSITION_COMMAND_GLOBAL_M
        )
        max_position_command = max_position_step / self._OSC_POSITION_ACTION_SCALE_M
        for _ in range(max_steps):
            cartesian = np.asarray(
                self.get_observation().get("robot_cartesian_pos", []), dtype=np.float64
            ).reshape(-1)
            if cartesian.size != 8 or not np.all(np.isfinite(cartesian[:7])):
                raise RuntimeError(
                    "goto_pose_osc requires robot_cartesian_pos as "
                    "XYZ + quaternion_wxyz + gripper_width"
                )
            command, position_error, orientation_error = _osc_pose_error_action(
                cartesian[:3],
                cartesian[3:7],
                target_eef_position,
                quaternion_wxyz,
                gripper_action,
                gain=gain,
                position_action_scale_m=self._OSC_POSITION_ACTION_SCALE_M,
                rotation_action_scale_rad=self._OSC_ROTATION_ACTION_SCALE_RAD,
            )
            position_norm = float(np.linalg.norm(command[:3]))
            if position_norm > max_position_command:
                command[:3] *= max_position_command / position_norm
            if (
                position_error <= position_tolerance_m
                and orientation_error <= self._GOTO_ORIENTATION_TOLERANCE_RAD
            ):
                consecutive += 1
                command[:6] = 0.0
            else:
                consecutive = 0
            self._env.step_osc_pose(command)
            if consecutive >= self._OSC_LEGACY_CONVERGED_STEPS:
                return
        raise CartesianPoseConvergenceError(
            "goto_pose_osc failed legacy direct Cartesian convergence: "
            f"stage={stage} position_error_m={position_error:.6f} "
            f"position_tolerance_m={position_tolerance_m:.6f} "
            f"orientation_error_rad={orientation_error:.6f} "
            f"orientation_tolerance_rad={self._GOTO_ORIENTATION_TOLERANCE_RAD:.6f}"
        )


__all__ = [
    "FrankaLiberoApiReducedOscSkillLibrary",
    "_interpolate_pose_minimum_jerk",
    "_minimum_jerk",
    "_osc_pose_error_action",
    "_slew_limit_arm_command",
]
