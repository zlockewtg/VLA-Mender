"""Stateful action perturbations applied during simulator rollouts.
Noise belongs at the rollout boundary: callers sample it immediately before an
environment step and persist the resulting executed command as the action
label. Dataset builders must never synthesize noisy labels after the fact.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class OscActionNoise:
    """Deterministic, temporally smooth noise for normalized OSC arm commands.

    The first six dimensions are perturbed and the seventh (gripper) dimension
    is always zero. ``standard_deviation`` and ``maximum_absolute`` therefore
    each contain exactly six values.
    """

    def __init__(self, config: dict[str, Any]):
        kind = str(config.get("type", "ornstein_uhlenbeck"))
        if kind != "ornstein_uhlenbeck":
            raise ValueError("OSC action noise type must be 'ornstein_uhlenbeck'")
        seed = config.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("OSC action noise seed must be an integer")
        rho = float(config.get("rho", 0.85))
        if not np.isfinite(rho) or not 0.0 <= rho < 1.0:
            raise ValueError("OSC action noise rho must be in [0, 1)")
        standard_deviation = np.asarray(
            config.get("standard_deviation"), dtype=np.float64
        ).reshape(-1)
        maximum_absolute = np.asarray(
            config.get("maximum_absolute"), dtype=np.float64
        ).reshape(-1)
        if standard_deviation.shape != (6,) or not np.all(
            np.isfinite(standard_deviation) & (standard_deviation > 0.0)
        ):
            raise ValueError(
                "OSC action noise standard_deviation must contain six positive values"
            )
        if maximum_absolute.shape != (6,) or not np.all(
            np.isfinite(maximum_absolute) & (maximum_absolute > 0.0)
        ):
            raise ValueError(
                "OSC action noise maximum_absolute must contain six positive values"
            )
        if np.any(maximum_absolute < standard_deviation):
            raise ValueError(
                "OSC action noise maximum_absolute must be at least one standard deviation"
            )
        self.config = {
            "type": kind,
            "seed": int(seed),
            "rho": rho,
            "standard_deviation": standard_deviation.tolist(),
            "maximum_absolute": maximum_absolute.tolist(),
            "dimensions": [0, 1, 2, 3, 4, 5],
            "gripper_perturbed": False,
        }
        self._rho = rho
        self._standard_deviation = standard_deviation
        self._maximum_absolute = maximum_absolute
        self._rng = np.random.default_rng(int(seed))
        self._state: np.ndarray | None = None

    def sample(self, action_size: int) -> np.ndarray:
        """Return one seven-dimensional sample with zero gripper noise."""
        if int(action_size) != 7:
            raise ValueError("OSC action noise requires a seven-dimensional action")
        innovation = self._rng.normal(0.0, self._standard_deviation, size=6)
        if self._state is None:
            self._state = innovation
        else:
            stationary_scale = float(np.sqrt(1.0 - self._rho * self._rho))
            self._state = self._rho * self._state + stationary_scale * innovation
        self._state = np.clip(
            self._state, -self._maximum_absolute, self._maximum_absolute
        )
        return np.concatenate([self._state.copy(), np.zeros(1, dtype=np.float64)])
