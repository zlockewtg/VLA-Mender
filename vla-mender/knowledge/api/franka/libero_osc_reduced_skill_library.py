"""Native OSC control surface for the LIBERO reduced skill library."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import BaseEnv
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


class FrankaLiberoApiReducedOscSkillLibrary(FrankaLiberoApiReducedSkillLibrary):
    """Reduced LIBERO skill library backed by native operational-space control."""

    _OSC_POSITION_ACTION_SCALE_M = 0.05
    _OSC_ROTATION_ACTION_SCALE_RAD = 0.5
    _OSC_MAX_POSITION_COMMAND_GLOBAL_M = 0.05
    _OSC_MAX_POSITION_COMMAND_HANDLE_M = 0.025
    _OSC_CONVERGED_STEPS = 2
    _WORKSPACE_LOWER = np.array([-0.1, -0.5, 0.005], dtype=np.float64)
    _WORKSPACE_UPPER = np.array([0.75, 0.5, 0.9], dtype=np.float64)

    def __init__(self, env: BaseEnv) -> None:
        if not hasattr(env, "step_osc_pose"):
            raise TypeError(
                "FrankaLiberoApiReducedOscSkillLibrary requires "
                "capx.envs.simulators.libero.FrankaLiberoOscEnv"
            )
        super().__init__(env, enable_ik=False)
        self._handle_alignment_active = False

    def functions(self) -> dict[str, Any]:
        fns = super().functions()
        fns["get_osc_controller_spec"] = self.get_osc_controller_spec
        fns["osc_step"] = self.osc_step
        fns["goto_pose_osc"] = self.goto_pose_osc
        # Compatibility for existing code-policy programs.  On this API the
        # familiar name is intentionally routed to native OSC, never IK.
        fns["goto_pose"] = self.goto_pose
        return fns

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
        }

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
    ) -> None:
        """Move to a Cartesian pose with closed-loop native OSC and no IK.

        Each simulator frame recomputes position and rotation error from the
        latest public robot observation and dispatches one normalized native
        OSC action.  This produces larger, feedback-driven Cartesian actions
        like a base OSC policy; it does not expand a joint waypoint into many
        slow ``JOINT_POSITION`` replay steps.

        Args:
            position: Three-value motion-target XYZ in the same frame returned
                by ``get_robot_state(obs)["motion_target_position"]``.
            quaternion_wxyz: Four-value world-frame WXYZ unit quaternion.
            z_approach: Optional non-negative world-Z approach offset in meters.
            gain: Positive Cartesian feedback gain.  The default is recommended.
            max_steps: Positive per-stage simulator-frame budget.

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

        if z_approach > 0.0:
            approach = pos.copy()
            approach[2] += float(z_approach)
            self._goto_pose_osc_stage(
                approach, quat, stage="approach", gain=gain, max_steps=int(max_steps)
            )
        self._goto_pose_osc_stage(
            pos, quat, stage="final", gain=gain, max_steps=int(max_steps)
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

    def _goto_pose_osc_stage(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        stage: str,
        gain: float,
        max_steps: int,
    ) -> None:
        commanded_position = np.clip(
            np.asarray(position, dtype=np.float64).reshape(3),
            self._WORKSPACE_LOWER,
            self._WORKSPACE_UPPER,
        )
        target_eef_position = apply_tcp_offset(
            commanded_position, quaternion_wxyz, self._TCP_OFFSET
        )
        gripper_action = float(
            np.clip(1.0 - 2.0 * float(self._env._gripper_fraction), -1.0, 1.0)
        )
        consecutive = 0
        position_error = float("inf")
        orientation_error = float("inf")

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
            max_position_step = (self._OSC_MAX_POSITION_COMMAND_HANDLE_M
                                 if getattr(self, "_handle_alignment_active", False)
                                 else self._OSC_MAX_POSITION_COMMAND_GLOBAL_M)
            position_norm = float(np.linalg.norm(command[:3]))
            max_position_command = max_position_step / self._OSC_POSITION_ACTION_SCALE_M
            if position_norm > max_position_command:
                command[:3] *= max_position_command / position_norm
            if (
                position_error <= self._GOTO_POSITION_TOLERANCE_M
                and orientation_error <= self._GOTO_ORIENTATION_TOLERANCE_RAD
            ):
                consecutive += 1
                # A zero position delta reanchors the position goal while a
                # zero rotation delta retains robosuite's current orientation
                # goal.  Holding while checking a second frame prevents a
                # final correction from leaving an already accepted pose.
                command[:6] = 0.0
            else:
                consecutive = 0
            self._env.step_osc_pose(command)
            if consecutive >= self._OSC_CONVERGED_STEPS:
                return

        raise CartesianPoseConvergenceError(
            "goto_pose_osc failed Cartesian convergence: "
            f"stage={stage} position_error_m={position_error:.6f} "
            f"position_tolerance_m={self._GOTO_POSITION_TOLERANCE_M:.6f} "
            f"orientation_error_rad={orientation_error:.6f} "
            f"orientation_tolerance_rad={self._GOTO_ORIENTATION_TOLERANCE_RAD:.6f}"
        )


__all__ = [
    "FrankaLiberoApiReducedOscSkillLibrary",
    "_osc_pose_error_action",
]
