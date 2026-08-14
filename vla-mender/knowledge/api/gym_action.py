from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from knowledge.api.base_api import ApiBase


class GymActionApi(ApiBase):
    """Low-dimensional action helpers for Gym/Gymnasium-style robot environments."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        self._last_step: dict[str, Any] | None = None

    def functions(self) -> dict[str, Any]:
        return {
            "observe": self.observe,
            "step_action": self.step_action,
            "run_actions": self.run_actions,
            "get_action_space": self.get_action_space,
            "get_last_info": self.get_last_info,
        }

    def observe(self) -> dict[str, Any]:
        """Return the current normalized environment observation.

        Returns:
            A dictionary with the latest environment state. Common keys are:
            - state: raw simulator state or a state dictionary.
            - robot_proprio: robot-only proprioception when available.
            - rgb: mapping from camera name to RGB image arrays.
            - depth: mapping from camera name to depth arrays when available.
            - cameras: mapping from camera name to camera dictionaries.
            - reward, success, done, action_space.
        """
        return self._env.get_observation()

    def step_action(self, action: Iterable[float] | np.ndarray) -> dict[str, Any]:
        """Step the environment once with a low-dimensional action.

        Args:
            action:
                Numeric action matching the environment action space. For franka_sim this is
                usually shape (4,) [dx, dy, dz, gripper] or shape (7,) when orientation
                actions are enabled. For Metaworld this is shape (4,).

        Returns:
            A dictionary containing obs, reward, terminated, truncated, done, success, and info.
        """
        self._log_step("step_action", f"Stepping action {np.asarray(action).tolist()}")
        obs, reward, terminated, truncated, info = self._env.step(np.asarray(action, dtype=np.float32))
        result = {
            "obs": obs,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done": bool(terminated or truncated),
            "success": bool(getattr(self._env, "task_completed", lambda: False)()),
            "info": info,
        }
        self._last_step = result
        image = self._first_image(obs)
        if image is not None:
            self._log_step_update(images=image)
        return result

    def run_actions(
        self,
        actions: Iterable[Iterable[float] | np.ndarray],
        *,
        repeat: int = 1,
    ) -> list[dict[str, Any]]:
        """Run a sequence of low-dimensional actions.

        Args:
            actions:
                Iterable of numeric actions.
            repeat:
                Number of times to repeat each action before moving to the next.

        Returns:
            A list with one step result per simulator step. Stops early if the task is done.
        """
        repeat = max(1, int(repeat))
        results: list[dict[str, Any]] = []
        for action in actions:
            for _ in range(repeat):
                result = self.step_action(action)
                results.append(result)
                if result["done"]:
                    return results
        return results

    def get_action_space(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the low-level action space."""
        space = getattr(self._env, "action_space", None)
        if space is None:
            return {}
        desc: dict[str, Any] = {"type": type(space).__name__}
        shape = getattr(space, "shape", None)
        if shape is not None:
            desc["shape"] = tuple(shape)
        dtype = getattr(space, "dtype", None)
        if dtype is not None:
            desc["dtype"] = str(dtype)
        for attr in ("low", "high"):
            value = getattr(space, attr, None)
            if value is not None:
                desc[attr] = np.asarray(value).tolist()
        return desc

    def get_last_info(self) -> dict[str, Any]:
        """Return metadata from the latest step_action call."""
        if self._last_step is None:
            return {}
        return dict(self._last_step.get("info", {}))

    def _first_image(self, obs: dict[str, Any]) -> np.ndarray | None:
        rgb = obs.get("rgb")
        if isinstance(rgb, dict) and rgb:
            return np.asarray(rgb[next(iter(rgb))])
        cameras = obs.get("cameras")
        if isinstance(cameras, dict) and cameras:
            cam = cameras[next(iter(cameras))]
            images = cam.get("images", {}) if isinstance(cam, dict) else {}
            if "rgb" in images:
                return np.asarray(images["rgb"])
        return None
