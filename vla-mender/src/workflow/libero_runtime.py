"""Small, task-agnostic LIBERO/MuJoCo runtime bridge.

All imports of LIBERO and robosuite are lazy so configuration validation and
diagnosis evidence preparation remain usable on CPU-only machines.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .parameters import ControlSpace, ResetDynamics

PUBLIC_REPLAY_TOLERANCE = 2e-4
SIM_STATE_TOLERANCE = 1e-12
STABILIZATION_STEPS = 10
DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


_LIBERO_PATH_OVERRIDE: dict[str, str] | None = None


def _configured_libero_path(key: str) -> str:
    if _LIBERO_PATH_OVERRIDE is None or key not in _LIBERO_PATH_OVERRIDE:
        raise KeyError(f"unknown configured LIBERO path: {key}")
    return _LIBERO_PATH_OVERRIDE[key]


def _libero_path_override(libero_root: Path) -> dict[str, str]:
    root = libero_root.expanduser().resolve()
    paths = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(root.parent / "datasets"),
        "assets": str(root / "assets"),
    }
    missing = [
        key
        for key in ("benchmark_root", "bddl_files", "init_states", "assets")
        if not Path(paths[key]).is_dir()
    ]
    if missing:
        raise ValueError(
            f"invalid LIBERO resource root {root}; missing path keys {missing}"
        )
    return paths


def libero_imports(
    libero_root: Path | None = None,
) -> tuple[Any, Any, Any]:
    """Import LIBERO and optionally override all resource paths in memory.

    LIBERO normally reads ``${LIBERO_CONFIG_PATH}/config.yaml``. A configured
    ``backend.libero_root`` replaces that process-global file lookup with a
    run-contract path resolver before benchmark modules bind ``get_libero_path``.
    """

    global _LIBERO_PATH_OVERRIDE
    errors: list[ImportError] = []
    for package_name in ("libero.libero", "libero"):
        try:
            package = importlib.import_module(package_name)
        except ImportError as exc:
            if exc.name and not (
                exc.name == package_name or exc.name.startswith(package_name + ".")
            ):
                raise
            errors.append(exc)
            continue
        if libero_root is not None:
            _LIBERO_PATH_OVERRIDE = _libero_path_override(libero_root)
            package.get_libero_path = _configured_libero_path
            try:
                utils = importlib.import_module(package_name + ".utils")
            except ImportError:
                utils = None
            if utils is not None:
                utils.get_libero_path = _configured_libero_path
        try:
            benchmark = importlib.import_module(package_name + ".benchmark")
            envs = importlib.import_module(package_name + ".envs")
        except ImportError as exc:
            if exc.name and not exc.name.startswith(package_name + "."):
                raise
            errors.append(exc)
            continue
        return benchmark, envs.OffScreenRenderEnv, package.get_libero_path
    cause = errors[-1] if errors else None
    raise RuntimeError(
        "LIBERO and robosuite are required for simulator stages"
    ) from cause


def controller_name(control_space: ControlSpace) -> str:
    return "OSC_POSE" if control_space == "osc" else "JOINT_POSITION"


class LiberoRuntime:
    """Build environments and perform exact state/controller operations."""

    def __init__(
        self,
        suite: str,
        task_id: int,
        control_space: ControlSpace,
        control_frequency_hz: int,
        *,
        libero_root: Path | None = None,
    ):
        self.suite = suite
        self.task_id = task_id
        self.control_space = control_space
        self.control_frequency_hz = control_frequency_hz
        self.libero_root = libero_root
        benchmark, offscreen_env, get_libero_path = libero_imports(libero_root)
        suite_obj = benchmark.get_benchmark_dict()[suite]()
        task = suite_obj.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self._env_cls = offscreen_env
        self._bddl = bddl

    def task_definition(self) -> tuple[Any, Any]:
        benchmark, _, _ = libero_imports(self.libero_root)
        suite = benchmark.get_benchmark_dict()[self.suite]()
        return suite, suite.get_task(self.task_id)

    def task_description(self) -> str:
        """Return the canonical LIBERO language instruction for this task."""

        _, task = self.task_definition()
        return str(task.language)

    def official_initial_states(self) -> np.ndarray:
        suite, _ = self.task_definition()
        return np.asarray(suite.get_task_init_states(self.task_id), dtype=np.float64)

    def new_env(
        self,
        seed: int,
        *,
        camera_depths: bool = False,
        camera_segmentations: str | None = None,
    ) -> Any:
        env = self._env_cls(
            bddl_file_name=self._bddl,
            camera_heights=256,
            camera_widths=256,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            control_freq=self.control_frequency_hz,
            controller=controller_name(self.control_space),
        )
        env.seed(int(seed))
        env.reset()
        actual = int(getattr(getattr(env, "env", env), "control_freq", self.control_frequency_hz))
        if actual != self.control_frequency_hz:
            raise RuntimeError(f"requested {self.control_frequency_hz} Hz but LIBERO created {actual} Hz")
        return env

    @staticmethod
    def neutral_action(env: Any) -> np.ndarray:
        low, high = env.env.action_spec
        action = np.zeros_like(np.asarray(low, dtype=np.float32))
        action[-1] = np.clip(-1.0, np.asarray(low)[-1], np.asarray(high)[-1])
        return action

    @staticmethod
    def public_state(obs: dict[str, Any]) -> np.ndarray:
        """Return the policy-visible state, without private simulator fields."""

        if "robot0_eef_pos" in obs:
            quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
            quat[3] = np.clip(quat[3], -1.0, 1.0)
            denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
            axis_angle = (np.zeros(3, dtype=np.float64) if math.isclose(denominator, 0.0)
                          else quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator)
            values = [obs["robot0_eef_pos"], axis_angle, obs.get("robot0_gripper_qpos", [])]
            return np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values])
        if "robot0_joint_pos" in obs:
            return np.concatenate(
                [np.asarray(obs["robot0_joint_pos"], dtype=np.float64).reshape(-1),
                 np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float64).reshape(-1)]
            )
        raise KeyError("LIBERO observation has neither robot0_eef_pos nor robot0_joint_pos")

    @staticmethod
    def observation_images(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        def image(*keys: str) -> np.ndarray:
            for key in keys:
                if key in obs:
                    value = np.asarray(obs[key], dtype=np.uint8)
                    return np.ascontiguousarray(value[::-1, ::-1])
            raise KeyError(f"missing camera observation; tried {keys}")

        return image("agentview_image", "agentview_rgb"), image("robot0_eye_in_hand_image", "robot0_wrist_image")

    @staticmethod
    def capture_gripper_controller(env: Any) -> np.ndarray | None:
        try:
            return np.asarray(env.env.robots[0].gripper.current_action, dtype=np.float64).copy()
        except (AttributeError, KeyError):
            return None

    @staticmethod
    def _sync_controller(env: Any) -> dict[str, Any]:
        controller = getattr(env.env.robots[0], "controller", None)
        if controller is None:
            return {"controller_synchronized": False, "reason": "controller_unavailable"}
        if hasattr(controller, "update"):
            controller.update(force=True)
        robot = env.env.robots[0]
        current_value = getattr(robot, "joint_positions", None)
        if current_value is None:
            current_value = getattr(robot, "_joint_positions")
        current = np.asarray(current_value, dtype=np.float64)
        if hasattr(controller, "update_initial_joints"):
            controller.update_initial_joints(current)
        if hasattr(controller, "update_initial_joints_goal"):
            controller.update_initial_joints_goal(current)
        goal_error = None
        if hasattr(controller, "goal_pos") and hasattr(controller, "ee_pos"):
            goal_error = float(np.linalg.norm(np.asarray(controller.goal_pos) - np.asarray(controller.ee_pos)))
        return {"controller_synchronized": True, "controller_goal_error": goal_error}

    @staticmethod
    def apply_reset_dynamics(env: Any, dynamics: ResetDynamics) -> dict[str, Any]:
        """Apply reset policy without advancing simulation time."""

        if dynamics == "preserve_full_state":
            return {"dynamics": dynamics, "qvel_zeroed": False, "cleared_fields": {}}
        if dynamics != "quiescent_osc":
            raise ValueError(f"unknown reset dynamics: {dynamics}")
        sim = getattr(env, "sim", None)
        if sim is None:
            sim = getattr(getattr(env, "env", None), "sim", None)
        if sim is None:
            raise RuntimeError("LIBERO environment does not expose MuJoCo sim")
        qvel = np.asarray(sim.data.qvel)
        qvel[...] = 0.0
        cleared: dict[str, float] = {}
        for field in ("ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied"):
            value = getattr(sim.data, field, None)
            if value is not None:
                array = np.asarray(value)
                cleared[field] = float(np.max(np.abs(array))) if array.size else 0.0
                value[...] = 0.0
        if hasattr(env, "regenerate_obs_from_state"):
            env.regenerate_obs_from_state(np.asarray(env.get_sim_state(), dtype=np.float64))
        sync = LiberoRuntime._sync_controller(env)
        return {"dynamics": dynamics, "qvel_zeroed": True, "cleared_fields": cleared, **sync}

    @staticmethod
    def set_sim_state(env: Any, state: np.ndarray) -> dict[str, Any]:
        vector = np.asarray(state, dtype=np.float64)
        if hasattr(env, "set_sim_state"):
            env.set_sim_state(vector)
            if hasattr(env, "regenerate_obs_from_state"):
                env.regenerate_obs_from_state(vector)
        elif hasattr(env, "set_init_state"):
            env.set_init_state(vector)
        else:
            raise RuntimeError("LIBERO environment exposes neither set_sim_state nor set_init_state")
        return LiberoRuntime._sync_controller(env)

    @staticmethod
    def state_hash(state: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(state, dtype=np.float64).view(np.uint8)).hexdigest()

    @staticmethod
    def write_private_state(path: str | Path, *, sim_state: np.ndarray, gripper_state: np.ndarray | None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {"sim_state": np.asarray(sim_state, dtype=np.float64)}
        if gripper_state is not None:
            arrays["gripper_controller_state"] = np.asarray(gripper_state, dtype=np.float64)
        np.savez_compressed(destination, **arrays)

    @staticmethod
    def write_json(path: str | Path, value: dict[str, Any]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(destination)
