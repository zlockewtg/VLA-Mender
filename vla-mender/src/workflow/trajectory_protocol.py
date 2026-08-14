"""Public trajectory contract shared by rollout and failure diagnosis.

The protocol deliberately contains only observations and executed actions.  A
rollout may retain simulator-private state in a separate, coordinator-owned
artifact, but that data is never part of the diagnosis handoff.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

TRAJECTORY_PROTOCOL_NAME = "vla-mender.libero.openpi"
TRAJECTORY_PROTOCOL_VERSION = 2


def protocol_metadata() -> dict[str, Any]:
    return {
        "name": TRAJECTORY_PROTOCOL_NAME,
        "version": TRAJECTORY_PROTOCOL_VERSION,
        "state_timing": "pre_action",
        "transition": "state_t_and_action_t_lead_to_state_t_plus_1",
        "visibility": "public_observation_only",
    }


def _finite(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def validate_episode(record: Mapping[str, Any]) -> None:
    protocol = record.get("trajectory_protocol")
    if protocol != protocol_metadata():
        raise ValueError("episode does not use the supported trajectory protocol")
    states = record.get("states")
    actions = record.get("actions")
    successes = record.get("successes")
    rewards = record.get("rewards")
    if not all(isinstance(value, list) for value in (states, actions, successes, rewards)):
        raise ValueError("trajectory arrays must be JSON lists")
    lengths = {len(states), len(actions), len(successes), len(rewards)}
    if len(lengths) != 1:
        raise ValueError("states/actions/successes/rewards must have equal lengths")
    if not all(_finite(value) for value in (states, actions, rewards)):
        raise ValueError("trajectory contains non-finite state/action/reward values")
    if int(record.get("num_steps", -1)) != len(actions):
        raise ValueError("num_steps does not match trajectory length")
    if not isinstance(record.get("episode_index"), int):
        raise ValueError("episode_index must be an integer")


def validate_rollout_contract(summary: Mapping[str, Any], settings_fingerprint: str | None = None) -> None:
    if summary.get("trajectory_protocol") != protocol_metadata():
        raise ValueError("rollout summary does not use the supported trajectory protocol")
    fingerprint = summary.get("settings_fingerprint")
    if settings_fingerprint is not None and fingerprint != settings_fingerprint:
        raise ValueError(
            "rollout settings fingerprint does not match the supplied experiment settings"
        )
    episodes = summary.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("rollout summary must contain episodes[]")
    indices = [int(item["episode_index"]) for item in episodes]
    if len(indices) != len(set(indices)):
        raise ValueError("rollout summary contains duplicate episode indices")


def diagnosis_evidence_metadata(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_protocol": summary["trajectory_protocol"],
        "settings_fingerprint": summary.get("settings_fingerprint"),
        "transition_semantics": "row t is state_t and action_t; action_t leads to state_(t+1)",
    }
