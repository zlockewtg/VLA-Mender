from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def repair_settings(tmp_path: Path) -> Path:
    repo = Path(__file__).resolve().parents[2]
    run_root = tmp_path / "prepared_task"
    failure_root = run_root / "failure_diagnosis"
    states = failure_root / "private_reset_states"
    views = failure_root / "agent_views"
    episodes = run_root / "rollout" / "episodes"
    for directory in (states, views, episodes):
        directory.mkdir(parents=True, exist_ok=True)
    (run_root / "experiment.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "task": {
                    "suite": "libero_goal",
                    "task_id": 3,
                    "task_description": "put the cup on the plate",
                },
                "rollout": {"control_frequency_hz": 20},
            }
        ),
        encoding="utf-8",
    )
    _json(
        run_root / "rollout" / "summary.json",
        {"task": {"description": "put the cup on the plate"}},
    )
    _json(run_root / "rollout" / "successful_episodes.json", {"episodes": [9]})
    _json(failure_root / "diagnosis.json", {"failure_modes": [{"id": "FM-01"}]})

    jobs = []
    resets = []
    for index in range(5):
        frame = 20 + index
        state_name = f"episode_{index:06d}_frame_{frame:06d}.npz"
        view_name = f"episode_{index:06d}_frame_{frame:06d}.png"
        state_path = states / state_name
        view_path = views / view_name
        state_path.write_bytes(f"state-{index}".encode())
        view_path.write_bytes(f"view-{index}".encode())
        file_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
        view_hash = hashlib.sha256(view_path.read_bytes()).hexdigest()
        jobs.append(
            {
                "job_id": f"e{index:06d}-f{frame:06d}",
                "episode_index": index,
                "reset_frame_index": frame,
                "reset_state": state_name,
                "agent_view": view_name,
                "target_control_space": "osc",
                "reset_dynamics": "preserve_full_state",
                "failure_phase": "grasp",
                "failure_mode_id": "FM-01",
                "failure_category": "contact",
                "failure_mode": "missed cup grasp",
            }
        )
        resets.append(
            {
                "episode_index": index,
                "requested_frame_index": frame,
                "verified": True,
                "private_state_sha256": f"state-vector-{index}",
                "agent_view_sha256": view_hash,
                "private_state_file_sha256": file_hash,
            }
        )
        _json(
            episodes / f"episode_{index:06d}.json",
            {"episode_index": index, "scene_model_seed": 1000 + index},
        )
    _json(failure_root / "repair_jobs.json", {"schema_version": 1, "jobs": jobs})
    _json(failure_root / "public_reset_bank.json", {"schema_version": 1, "resets": resets})
    _json(failure_root / "replay_verification.json", {"schema_version": 1, "reports": resets})

    libero_root = tmp_path / "libero"
    libero_root.mkdir()
    output = tmp_path / "repair_campaign"
    settings = tmp_path / "repair_example.yaml"
    settings.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "campaign": {
                    "name": "test_repair",
                    "output_dir": str(output),
                    "parallel_tasks": 1,
                },
                "project": {
                    "root": str(repo),
                    "source_root": "vla-mender/src",
                    "knowledge_root": "vla-mender/knowledge",
                },
                "environment": {
                    "python": sys.executable,
                    "working_directory": str(repo),
                    "libero_root": str(libero_root),
                    "env": {},
                },
                "resources": {
                    "gpus": [0],
                    "gpus_per_task": 1,
                    "workers_per_gpu": 2,
                    "services": {"profile": "minimal", "manage": False},
                },
                "runtime": {
                    "max_steps": 20,
                    "job_timeout_s": 30,
                    "infrastructure_retries": 1,
                    "resume": True,
                    "backend": "fake",
                },
                "repair": {"initial_split": {"debug": 2, "validation": 3}},
                "tasks": [{"run_root": str(run_root)}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return settings


@pytest.fixture
def repair_settings_v2(repair_settings: Path) -> Path:
    value = yaml.safe_load(repair_settings.read_text(encoding="utf-8"))
    value["schema_version"] = 2
    value["campaign"]["output_dir"] = str(repair_settings.parent / "repair_campaign_v2")
    value["repair"] = {
        "budget": {"soft_task_hours": 4},
        "smoke": {"min_seeds": 3, "max_seeds": 8},
        "exploration_review": {
            "consecutive_no_gain_candidates": 3,
            "per_seed_policy_attempts": 8,
        },
        "allow_abandon": True,
    }
    path = repair_settings.parent / "repair_v2.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path
