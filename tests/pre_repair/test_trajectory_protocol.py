from __future__ import annotations

import pytest

from workflow.trajectory_protocol import protocol_metadata, validate_rollout_contract


def _summary() -> dict:
    return {
        "trajectory_protocol": protocol_metadata(),
        "settings_fingerprint": "fingerprint",
        "overall": {
            "episodes": 2,
            "successes": 1,
            "failures": 1,
        },
        "episodes": [
            {"episode_index": 0, "success": True},
            {"episode_index": 1, "success": False},
        ],
    }


def test_rollout_contract_accepts_exact_outcome_counts():
    validate_rollout_contract(_summary(), "fingerprint")


def test_rollout_contract_requires_failure_count():
    summary = _summary()
    del summary["overall"]["failures"]

    with pytest.raises(ValueError, match="episodes/successes/failures"):
        validate_rollout_contract(summary)


def test_rollout_contract_rejects_counts_that_disagree_with_episodes():
    summary = _summary()
    summary["overall"]["successes"] = 2
    summary["overall"]["failures"] = 0

    with pytest.raises(ValueError, match="do not match episode outcomes"):
        validate_rollout_contract(summary)
