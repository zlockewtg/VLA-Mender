"""Structural environment type used by local knowledge APIs.

The knowledge layer only needs an object with the documented public robot
methods. Keeping this as a protocol removes the historical import-time CaP-X
dependency while retaining compatibility with CaP-X environments.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseEnv(Protocol):
    _gripper_fraction: float

    def get_observation(self) -> dict[str, Any]: ...

    def step(self, action: Any) -> Any: ...

    def step_osc_pose(self, action: Any) -> Any: ...

    def move_to_joints_blocking(self, joints: Any, **kwargs: Any) -> Any: ...

    def _set_gripper(self, fraction: float) -> None: ...

    def _step_once(self) -> None: ...
