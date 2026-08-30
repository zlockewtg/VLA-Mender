"""Native LIBERO exact-reset backend for generated code-policy programs."""

from __future__ import annotations

import contextlib
import io
import traceback
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import viser.transforms as vtf

from knowledge.api.utils.serve_utils import ToolServiceError
from workflow.libero_runtime import LiberoRuntime, SIM_STATE_TOLERANCE
from workflow.rollout.action_noise import OscActionNoise

from .policy import execute_program, validate_program
from .util import atomic_write_json, atomic_write_text, sha256_bytes, sha256_file, utc_now


class ResetVerificationError(RuntimeError):
    """Prepared reset bytes or restored simulator observations are inconsistent."""


class RepairTaskCompleted(Exception):
    """Internal control flow used to stop immediately at native task success."""


def _execute_repair_program(
    source: str,
    *,
    functions: dict[str, Any],
    observation: dict[str, Any],
) -> str | None:
    """Execute policy code while keeping tool outages out of policy results."""
    try:
        execute_program(source, functions=functions, observation=observation)
    except RepairTaskCompleted:
        return None
    except ToolServiceError:
        raise
    except Exception:
        return traceback.format_exc()
    return None


class NativeLiberoPolicyEnv:
    """Small public adapter consumed by VLA-Mender's local knowledge API."""

    gripper_metric_length = 0.04

    def __init__(
        self,
        env: Any,
        runtime: LiberoRuntime,
        observation: dict[str, Any],
        max_steps: int,
        *,
        action_noise: dict[str, Any] | None = None,
        post_success_steps: int = 0,
    ):
        self.env = env
        self.runtime = runtime
        self._current_obs = observation
        self.max_steps = int(max_steps)
        self._sim_step_count = 0
        self._current_reward = 0.0
        self._current_done = bool(env.check_success())
        self._current_info: dict[str, Any] = {}
        self._interrupt_on_task_completion = True
        self._requested_post_success_steps = int(post_success_steps)
        self._deferred_post_success_limit: int | None = None
        width = float(np.asarray(observation.get("robot0_gripper_qpos", [0.04])).reshape(-1)[0])
        self._gripper_fraction = float(np.clip(width / self.gripper_metric_length, 0.0, 1.0))
        self.output_dir: Path | None = None
        self.states: list[list[float]] = []
        self.actions: list[list[float]] = []
        self.nominal_actions: list[list[float]] = []
        self.sampled_action_noises: list[list[float]] = []
        self.applied_action_noises: list[list[float]] = []
        self.rewards: list[float] = []
        self.success_flags: list[bool] = []
        self.wide_frames: list[np.ndarray] = []
        self.wrist_frames: list[np.ndarray] = []
        if action_noise is not None and runtime.control_space != "osc":
            raise ValueError("action noise is currently supported only for OSC control")
        self._action_noise = (
            OscActionNoise(dict(action_noise)) if action_noise is not None else None
        )
        self.action_noise_config = (
            dict(self._action_noise.config) if self._action_noise is not None else None
        )

    @property
    def low_level_env(self) -> NativeLiberoPolicyEnv:
        return self

    def _raise_if_limit(self) -> None:
        if self._sim_step_count >= self.max_steps:
            raise RuntimeError(
                f"ROLLOUT_STEP_LIMIT_EXCEEDED: repair used {self.max_steps} simulator steps"
            )

    def _record_pre_action(
        self,
        action: np.ndarray,
        *,
        nominal_action: np.ndarray,
        sampled_noise: np.ndarray,
        applied_noise: np.ndarray,
    ) -> None:
        wide, wrist = self.runtime.observation_images(self._current_obs)
        self.wide_frames.append(wide)
        self.wrist_frames.append(wrist)
        self.states.append(self.runtime.public_state(self._current_obs).tolist())
        self.actions.append(np.asarray(action, dtype=np.float32).tolist())
        if self._action_noise is not None:
            self.nominal_actions.append(
                np.asarray(nominal_action, dtype=np.float32).tolist()
            )
            self.sampled_action_noises.append(
                np.asarray(sampled_noise, dtype=np.float32).tolist()
            )
            self.applied_action_noises.append(
                np.asarray(applied_noise, dtype=np.float32).tolist()
            )

    def _dispatch(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._raise_if_limit()
        command = np.asarray(action, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(command)):
            raise ValueError("repair action must contain finite values")
        low, high = self.env.env.action_spec
        if command.shape != np.asarray(low).shape:
            raise ValueError(
                f"repair action has shape {command.shape}; expected {np.asarray(low).shape}"
            )
        nominal_command = np.clip(command, np.asarray(low), np.asarray(high))
        sampled_noise = np.zeros_like(nominal_command, dtype=np.float64)
        if self._action_noise is not None:
            sampled_noise = self._action_noise.sample(nominal_command.size)
        command = np.clip(
            nominal_command + sampled_noise, np.asarray(low), np.asarray(high)
        )
        applied_noise = command - nominal_command
        self._record_pre_action(
            command,
            nominal_action=nominal_command,
            sampled_noise=sampled_noise,
            applied_noise=applied_noise,
        )
        observation, reward, done, info = self.env.step(command.tolist())
        self._current_obs = observation
        self._current_reward = float(reward)
        self._current_done = bool(done)
        self._current_info = dict(info)
        self._sim_step_count += 1
        self.rewards.append(float(reward))
        self.success_flags.append(bool(self.env.check_success()))
        if self.success_flags[-1]:
            if self._interrupt_on_task_completion:
                raise RepairTaskCompleted
            if self._deferred_post_success_limit is not None:
                first_success = int(
                    np.flatnonzero(np.asarray(self.success_flags, dtype=bool))[0]
                )
                recorded = len(self.success_flags) - first_success - 1
                if recorded >= self._deferred_post_success_limit:
                    raise RepairTaskCompleted
        return observation, float(reward), bool(done), dict(info)

    def step_osc_pose(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        command = np.asarray(action, dtype=np.float64).reshape(-1)
        if command.size != 7:
            raise ValueError("OSC action must contain seven values")
        command = np.clip(command, -1.0, 1.0)
        self._gripper_fraction = float((1.0 - command[-1]) / 2.0)
        return self._dispatch(command)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, reward, done, info = self._dispatch(np.asarray(action, dtype=np.float64))
        return observation, reward, done, False, info

    def _set_gripper(self, fraction: float) -> None:
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))

    def _step_once(self) -> None:
        low, _ = self.env.env.action_spec
        command = np.zeros_like(np.asarray(low), dtype=np.float64)
        command[-1] = 1.0 - 2.0 * self._gripper_fraction
        self._dispatch(command)

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.01, max_steps: int = 120
    ) -> None:
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        for _ in range(int(max_steps)):
            current = np.asarray(self._current_obs["robot0_joint_pos"], dtype=np.float64)
            if float(np.linalg.norm(current - target)) < float(tolerance):
                return
            action = np.concatenate(
                [np.clip((target - current) * self.runtime.control_frequency_hz, -1.0, 1.0),
                 [1.0 - 2.0 * self._gripper_fraction]]
            )
            self._dispatch(action)

    def get_observation(self) -> dict[str, Any]:
        from robosuite.utils.camera_utils import get_real_depth_map

        observation: dict[str, Any] = {}
        sim = self.env.env.sim
        base_id = sim.model.body_name2id("robot0_base")
        eef_id = sim.model.body_name2id("gripper0_eef")
        base_pose = vtf.SE3(
            wxyz_xyz=np.concatenate([sim.data.xquat[base_id], sim.data.xpos[base_id]])
        )
        base_inverse = base_pose.inverse()
        height = int(np.asarray(self._current_obs["agentview_image"]).shape[0])
        width = int(np.asarray(self._current_obs["agentview_image"]).shape[1])
        for camera_name in ("agentview", "robot0_eye_in_hand"):
            camera_world = vtf.SE3(
                wxyz_xyz=np.concatenate(
                    [
                        vtf.SO3.from_matrix(sim.data.get_camera_xmat(camera_name)).wxyz,
                        sim.data.get_camera_xpos(camera_name),
                    ]
                )
            )
            camera_robot = (
                base_inverse
                @ camera_world
                @ vtf.SE3.from_rotation_and_translation(
                    vtf.SO3.from_rpy_radians(0.0, np.pi, 0.0), np.zeros(3)
                )
                @ vtf.SE3.from_rotation_and_translation(
                    vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi), np.zeros(3)
                )
            )
            camera_id = sim.model.camera_name2id(camera_name)
            fovy = float(sim.model.cam_fovy[camera_id])
            focal = 0.5 * height / np.tan(fovy * np.pi / 360.0)
            images: dict[str, Any] = {
                "rgb": np.asarray(self._current_obs[f"{camera_name}_image"], dtype=np.uint8)[::-1]
            }
            depth_key = f"{camera_name}_depth"
            if depth_key in self._current_obs:
                images["depth"] = get_real_depth_map(
                    sim, np.asarray(self._current_obs[depth_key])[::-1]
                )
            observation[camera_name] = {
                "pose": np.concatenate(
                    [camera_robot.translation(), camera_robot.rotation().wxyz]
                ),
                "pose_mat": camera_robot.as_matrix(),
                "intrinsics": np.array(
                    [[focal, 0.0, 0.5 * width], [0.0, focal, 0.5 * height], [0.0, 0.0, 1.0]]
                ),
                "images": images,
            }
        eef_robot = (
            base_inverse
            @ vtf.SE3(wxyz_xyz=np.concatenate([sim.data.xquat[eef_id], sim.data.xpos[eef_id]]))
            @ vtf.SE3.from_rotation_and_translation(
                vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                np.array([0.0, 0.0, -0.107]),
            )
        )
        gripper_width = float(
            np.asarray(self._current_obs["robot0_gripper_qpos"]).reshape(-1)[0]
            / self.gripper_metric_length
        )
        observation["robot_joint_pos"] = np.concatenate(
            [np.asarray(self._current_obs["robot0_joint_pos"]), [gripper_width]]
        )
        observation["robot_cartesian_pos"] = np.concatenate(
            [eef_robot.translation(), eef_robot.rotation().wxyz, [gripper_width]]
        )
        return observation

    def get_current_time_s(self) -> float:
        return self._sim_step_count / self.runtime.control_frequency_hz

    def task_completed(self) -> bool:
        return bool(self.env.check_success())

    def defer_task_completion_until_program_end(self) -> None:
        """Record success flags but let a teacher finish braking and release."""
        self._interrupt_on_task_completion = False
        # A deferred teacher may contain its own observed settle loop.  Stop it
        # once the job's requested tail is complete so that such a loop cannot
        # silently record more post-success transitions than the dataset
        # protocol permits.
        self._deferred_post_success_limit = self._requested_post_success_steps

    def ensure_post_success_steps(self, requested_steps: int) -> dict[str, Any]:
        """Record exactly ``requested_steps`` real transitions after first success.

        The normal repair control flow interrupts the policy on the first native
        task success.  Dataset collection jobs may explicitly request a bounded
        post-success tail.  Missing tail transitions are executed as neutral arm
        commands while preserving the observed gripper command; they still pass
        through ``_dispatch`` and therefore are simulator transitions recorded at
        the same pre-action point as every policy command.
        """
        if isinstance(requested_steps, bool) or not 0 <= int(requested_steps) <= 100:
            raise ValueError("post_success_steps must be an integer in [0, 100]")
        requested = int(requested_steps)
        success_indices = np.flatnonzero(np.asarray(self.success_flags, dtype=bool))
        if success_indices.size == 0:
            return {
                "requested_steps": requested,
                "first_success_frame": None,
                "recorded_steps": 0,
                "all_post_success_flags_true": False,
                "command_source": "native_policy_no_success",
            }
        first_success = int(success_indices[0])
        existing = len(self.success_flags) - first_success - 1
        if existing > requested:
            raise RuntimeError(
                "policy already recorded more post-success transitions than requested: "
                f"existing={existing}, requested={requested}"
            )
        previous_interrupt = self._interrupt_on_task_completion
        previous_deferred_limit = self._deferred_post_success_limit
        self._interrupt_on_task_completion = False
        self._deferred_post_success_limit = None
        try:
            for _ in range(requested - existing):
                self._step_once()
        finally:
            self._interrupt_on_task_completion = previous_interrupt
            self._deferred_post_success_limit = previous_deferred_limit
        recorded = len(self.success_flags) - first_success - 1
        post_flags = self.success_flags[first_success + 1 :]
        if recorded != requested:
            raise RuntimeError(
                f"post-success transition count mismatch: {recorded} != {requested}"
            )
        return {
            "requested_steps": requested,
            "first_success_frame": first_success,
            "recorded_steps": recorded,
            "all_post_success_flags_true": bool(post_flags) and all(post_flags),
            "command_source": "real_neutral_env_steps_preserving_current_gripper",
        }


def _restore_gripper(env: Any, state: np.ndarray | None) -> None:
    if state is None:
        return
    gripper = env.env.robots[0].gripper
    desired = np.asarray(state, dtype=np.float64).reshape(-1)
    gripper.current_action = desired.copy()


def _api_for(env: NativeLiberoPolicyEnv, control_space: str) -> Any:
    if control_space == "osc":
        from knowledge.api.franka.libero_osc_reduced_skill_library import (
            FrankaLiberoApiReducedOscSkillLibrary,
        )

        return FrankaLiberoApiReducedOscSkillLibrary(env)
    from knowledge.api.franka.libero_reduced_skill_library import (
        FrankaLiberoApiReducedSkillLibrary,
    )

    return FrankaLiberoApiReducedSkillLibrary(env)


def execute_repair_job(
    job: dict[str, Any],
    *,
    program_path: str | Path,
    attempt_dir: str | Path,
    libero_root: str | Path,
    max_steps: int,
) -> dict[str, Any]:
    """Restore one prepared reset and execute one frozen policy program."""

    source = Path(program_path).read_text(encoding="utf-8")
    violations = validate_program(source)
    if violations:
        return {
            "schema_version": 1,
            "outcome": "policy_invalid",
            "success": False,
            "violations": violations,
            "job_id": job["job_id"],
        }
    control_space = str(job["target_control_space"])
    runtime = LiberoRuntime(
        str(job["suite"]),
        int(job["task_id"]),
        control_space,
        int(job["control_frequency_hz"]),
        libero_root=Path(libero_root),
    )
    raw_env = runtime.new_env(int(job["scene_model_seed"]), camera_depths=True)
    output = Path(attempt_dir)
    output.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    try:
        reset_path = Path(str(job["reset_state"]))
        if sha256_file(reset_path) != str(job["reset_file_sha256"]):
            raise ResetVerificationError(f"prepared reset file hash changed: {reset_path}")
        with np.load(reset_path, allow_pickle=False) as payload:
            state = np.asarray(payload["sim_state"], dtype=np.float64).reshape(-1)
            gripper = (
                np.asarray(payload["gripper_controller_state"], dtype=np.float64)
                if "gripper_controller_state" in payload.files
                else None
            )
        if runtime.state_hash(state) != str(job["reset_hash"]):
            raise ResetVerificationError(f"prepared simulator state hash changed: {reset_path}")
        runtime.set_sim_state(raw_env, state)
        _restore_gripper(raw_env, gripper)
        current_state = np.asarray(raw_env.get_sim_state(), dtype=np.float64).reshape(-1)
        restore_error = float(np.max(np.abs(current_state - state)))
        if restore_error > SIM_STATE_TOLERANCE:
            raise ResetVerificationError(
                f"exact reset mismatch {restore_error} > {SIM_STATE_TOLERANCE}"
            )
        observation = raw_env.regenerate_obs_from_state(current_state)
        actual_wide, _ = runtime.observation_images(observation)
        if sha256_file(str(job["agent_view"])) != str(job["agent_view_file_sha256"]):
            raise ResetVerificationError(
                f"prepared public reset image hash changed: {job['agent_view']}"
            )
        expected_wide = np.asarray(imageio.imread(str(job["agent_view"])), dtype=np.uint8)
        expected_pixel_hash = sha256_bytes(np.ascontiguousarray(expected_wide).tobytes())
        if expected_pixel_hash != str(job["agent_view_sha256"]):
            raise ResetVerificationError(
                f"prepared public reset image pixels changed: {job['agent_view']}"
            )
        if actual_wide.shape != expected_wide.shape:
            raise ResetVerificationError(
                f"restored public image shape mismatch {actual_wide.shape} != {expected_wide.shape}"
            )
        image_mae = float(
            np.mean(np.abs(actual_wide.astype(np.int16) - expected_wide.astype(np.int16)))
        )
        if image_mae > 2.0:
            raise ResetVerificationError(
                f"restored public image mismatch MAE={image_mae}"
            )
        policy_env = NativeLiberoPolicyEnv(
            raw_env,
            runtime,
            observation,
            max_steps,
            action_noise=job.get("action_noise"),
            post_success_steps=int(job.get("post_success_steps", 0)),
        )
        policy_env.output_dir = output
        api = _api_for(policy_env, control_space)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            program_error = _execute_repair_program(
                source,
                functions=api.functions(),
                observation=api.get_observation(),
            )
            post_success_audit = policy_env.ensure_post_success_steps(
                int(job.get("post_success_steps", 0))
            )
        success = policy_env.task_completed() and program_error is None
        if policy_env.wide_frames:
            fps = int(job["control_frequency_hz"])
            imageio.mimsave(output / "wide.mp4", policy_env.wide_frames, fps=fps)
            imageio.mimsave(output / "wrist.mp4", policy_env.wrist_frames, fps=fps)
        trajectory = {
            "schema_version": 1,
            "states": policy_env.states,
            "actions": policy_env.actions,
            "rewards": policy_env.rewards,
            "success_flags": policy_env.success_flags,
            "post_success_audit": post_success_audit,
        }
        if policy_env.action_noise_config is not None:
            trajectory.update(
                {
                    "action_noise": policy_env.action_noise_config,
                    "nominal_actions": policy_env.nominal_actions,
                    "sampled_action_noises": policy_env.sampled_action_noises,
                    "applied_action_noises": policy_env.applied_action_noises,
                }
            )
        atomic_write_json(output / "trajectory.json", trajectory)
        atomic_write_json(
            output / "terminal_observation.json",
            {
                "public_state": runtime.public_state(policy_env._current_obs).tolist(),
                "task_completed": bool(policy_env.task_completed()),
            },
        )
        # Task-local rollout diagnostics only; these private simulator fields
        # are never copied into dataset episodes or exposed to code policies.
        # They make failed placement geometry auditable when the target is
        # visually occluded by the basket at the terminal frame.
        sim = raw_env.env.sim
        terminal_bodies = {}
        for body_id in range(int(sim.model.nbody)):
            body_name = sim.model.body_id2name(body_id) or ""
            lowered = body_name.lower()
            if "alphabet_soup" in lowered or "basket" in lowered:
                terminal_bodies[body_name] = {
                    "position": np.asarray(sim.data.body_xpos[body_id]).tolist(),
                    "quaternion_wxyz": np.asarray(sim.data.body_xquat[body_id]).tolist(),
                }
        atomic_write_json(
            output / "terminal_sim_diagnostics.json",
            {
                "task_completed": bool(policy_env.task_completed()),
                "bodies": terminal_bodies,
            },
        )
        atomic_write_text(output / "stdout.log", stdout.getvalue())
        # program_error is authoritative in result.json; stderr.log contains
        # only text actually emitted on stderr.
        atomic_write_text(output / "stderr.log", stderr.getvalue())
        return {
            "schema_version": 1,
            "outcome": "success" if success else "policy_failure",
            "success": success,
            "task_completed": bool(policy_env.task_completed()),
            "job_id": job["job_id"],
            "task_key": job["task_key"],
            "failure_mode_id": job["failure_mode_id"],
            "program_error": program_error,
            "simulator_steps": policy_env._sim_step_count,
            "post_success_audit": post_success_audit,
            "action_noise": policy_env.action_noise_config,
            "restore_state_max_abs_error": restore_error,
            "restore_image_mae": image_mae,
            "started_at": started_at,
            "finished_at": utc_now(),
        }
    finally:
        raw_env.close()
