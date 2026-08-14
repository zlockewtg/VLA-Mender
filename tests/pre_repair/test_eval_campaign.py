from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from workflow.rollout.runner import policy_seed
from workflow.rollout.state_provider import (
    InitialStateBundle,
    array_hash,
    existing_randomized_cache,
    load_custom_initial_states,
)


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "vla-mender" / "scripts" / "eval"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EVAL_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


campaign_eval = _load("vla_mender_campaign_eval", "eval.py")
worker_eval = _load("vla_mender_eval_runtime", "_runtime.py")


def _config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "campaign.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _base_yaml() -> str:
    return """\
schema_version: 1
checkpoint: ./global-checkpoint
config_name: global-config
openpi_source: ./openpi
python: /usr/bin/python3
tasks:
  - key: first
    suite: libero_goal
    task_id: 0
    checkpoint: ./task-checkpoint
    initial_states:
      count: 2
    evaluation:
      max_steps: 111
  - suite: libero_spatial
    task_id: 9
initial_states:
  provider: official
  count: 4
  indices: null
evaluation:
  max_steps: 300
resources:
  gpus: [0, 1]
  num_envs: 2
"""


def test_cli_overrides_task_local_values_and_resolves_relative_paths(tmp_path):
    config = _config(tmp_path, _base_yaml())
    args = campaign_eval.parse_args(
        [
            "--config",
            str(config),
            "--checkpoint",
            "./cli-checkpoint",
            "--state-count",
            "3",
            "--max-steps",
            "222",
            "--gpus",
            "4,5",
        ]
    )
    campaign = campaign_eval.resolve_campaign(args)

    assert [task.key for task in campaign.tasks] == ["first", "libero_spatial-task009"]
    assert {task.checkpoint for task in campaign.tasks} == {
        (tmp_path / "cli-checkpoint").resolve()
    }
    assert [task.initial_states["count"] for task in campaign.tasks] == [3, 3]
    assert all(task.initial_states["indices"] is None for task in campaign.tasks)
    assert [task.evaluation["max_steps"] for task in campaign.tasks] == [222, 222]
    assert campaign.resources["gpus"] == [4, 5]
    assert campaign.tasks[0].openpi_source == (tmp_path / "openpi").resolve()


def test_repeated_task_replaces_yaml_tasks_and_indices_clear_count(tmp_path):
    config = _config(tmp_path, _base_yaml())
    campaign = campaign_eval.resolve_campaign(
        campaign_eval.parse_args(
            [
                "--config",
                str(config),
                "--task",
                "libero_object:3",
                "--task",
                "libero_10:8",
                "--initial-state-indices",
                "1,4,7",
            ]
        )
    )

    assert [(task.suite, task.task_id) for task in campaign.tasks] == [
        ("libero_object", 3),
        ("libero_10", 8),
    ]
    assert all(task.initial_states["count"] is None for task in campaign.tasks)
    assert all(task.initial_states["indices"] == [1, 4, 7] for task in campaign.tasks)


def test_relative_output_is_resolved_from_yaml_directory(tmp_path):
    config = _config(tmp_path, _base_yaml())
    campaign = campaign_eval.resolve_campaign(
        campaign_eval.parse_args(["--config", str(config), "--output", "relative/run"])
    )
    assert campaign.output == (tmp_path / "relative" / "run").resolve()


@pytest.mark.parametrize(
    "fragment, error",
    [
        (
            """\
schema_version: 1
checkpoint: ./checkpoint
openpi_source: ./openpi
tasks: [{suite: libero_goal, task_id: 0}]
""",
            "provider must be explicitly selected",
        ),
        (
            """\
schema_version: 1
checkpoint: ./checkpoint
openpi_source: ./openpi
tasks: [{suite: libero_goal, task_id: 0}]
initial_states: {provider: official, count: 2, indices: [0, 1]}
""",
            "mutually exclusive",
        ),
        (
            """\
schema_version: 1
checkpoint: ./checkpoint
openpi_source: ./openpi
tasks:
  - {key: same, suite: libero_goal, task_id: 0}
  - {key: same, suite: libero_spatial, task_id: 1}
initial_states: {provider: official, count: 2, indices: null}
""",
            "task keys must be unique",
        ),
    ],
)
def test_invalid_campaign_contracts_fail_before_runtime(tmp_path, fragment, error):
    config = _config(tmp_path, fragment)
    with pytest.raises(ValueError, match=error):
        campaign_eval.resolve_campaign(
            campaign_eval.parse_args(["--config", str(config)])
        )


def test_official_state_boundary_and_videos_resume_are_rejected(tmp_path):
    config = _config(tmp_path, _base_yaml())
    with pytest.raises(ValueError, match="official initial-state indices"):
        campaign_eval.resolve_campaign(
            campaign_eval.parse_args(
                ["--config", str(config), "--initial-state-indices", "49,50"]
            )
        )
    with pytest.raises(ValueError, match="videos_only does not support resume"):
        campaign_eval.resolve_campaign(
            campaign_eval.parse_args(
                [
                    "--config",
                    str(config),
                    "--artifact-mode",
                    "videos_only",
                    "--output",
                    str(tmp_path / "run"),
                    "--resume",
                ]
            )
        )


def test_balanced_contiguous_shards_gpu_round_robin_and_contract_identity(tmp_path):
    config = _config(tmp_path, _base_yaml())
    campaign = campaign_eval.resolve_campaign(
        campaign_eval.parse_args(["--config", str(config), "--state-count", "5"])
    )
    task = campaign.tasks[0]
    workers = campaign_eval.build_workers(campaign, task, manifest=None)

    assert campaign_eval.split_indices(list(range(11)), 4) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10],
    ]
    assert [worker["indices"] for worker in workers] == [[0, 1, 2], [3, 4]]
    assert [worker["gpu"] for worker in workers] == [0, 1]
    left = {"generated_at": "a", "output": "/old", "gpus": [0, 1], "workers": 2}
    right = {"generated_at": "b", "output": "/new", "gpus": [6, 7], "workers": 2}
    assert campaign_eval._contract_identity(left) == campaign_eval._contract_identity(
        right
    )
    right["workers"] = 3
    assert campaign_eval._contract_identity(left) != campaign_eval._contract_identity(
        right
    )


def test_policy_seed_is_independent_of_worker_shards():
    expected = {
        (trial, index): 7 + 100 + trial * 50 + index
        for trial in range(2)
        for index in range(50)
    }
    shards = campaign_eval.split_indices(list(range(50)), 7)
    actual = {
        (trial, index): policy_seed(
            seed=7,
            offset=100,
            trial_index=trial,
            state_count=50,
            initial_state_index=index,
        )
        for shard in shards
        for trial in range(2)
        for index in shard
    }
    assert actual == expected


def test_worker_adapter_executes_the_shared_batch_core(tmp_path, monkeypatch):
    class FakeEnv:
        def __init__(self):
            self.env = self
            self.action_spec = (np.full(7, -1.0), np.full(7, 1.0))

        def seed(self, _seed):
            pass

        def reset(self):
            return {"step": 0}

        def set_init_state(self, _state):
            return {"step": 0}

        def step(self, _action):
            return {"step": 1}, 1.0, True, {}

        def close(self):
            pass

    class FakeRuntime:
        def __init__(self, *_args):
            pass

        @staticmethod
        def task_description():
            return "open the middle drawer"

        @staticmethod
        def neutral_action(_env):
            return np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)

        @staticmethod
        def observation_images(_obs):
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            return image, image.copy()

        @staticmethod
        def public_state(_obs):
            return np.zeros(8, dtype=np.float32)

        @staticmethod
        def new_env(_seed):
            return FakeEnv()

    class FakePolicy:
        @staticmethod
        def infer(_observation):
            return {
                "actions": np.zeros((1, 7), dtype=np.float32),
                "policy_timing": {"infer_ms": 1.0},
            }

    class FakeBackend:
        def __init__(self, *_args, **_kwargs):
            self.policy = FakePolicy()

    class FakeDataset:
        def __init__(self):
            self.meta = SimpleNamespace(total_episodes=0)
            self.frames = []
            self.stopped = False

        def add_frame(self, frame, *, task):
            self.frames.append((frame, task))

        def save_episode(self):
            self.meta.total_episodes += 1

        def stop_image_writer(self):
            self.stopped = True

    dataset = FakeDataset()

    def fake_open_output(args, *, contract, task_description, specs):
        assert specs == [(0, 0), (0, 1)]
        return dataset, worker_eval._new_results(
            contract, task_description=task_description, output=args.output
        )

    monkeypatch.setattr(worker_eval, "LiberoRuntime", FakeRuntime)
    monkeypatch.setattr(worker_eval, "OpenPIBackend", FakeBackend)
    monkeypatch.setattr(worker_eval, "_open_output", fake_open_output)
    monkeypatch.setattr(
        worker_eval,
        "resolve_evaluation_initial_states",
        lambda **_kwargs: InitialStateBundle(
            states=np.zeros((2, 3), dtype=np.float64),
            entries=None,
            kind="official",
            array_sha256="state-array-hash",
        ),
    )
    monkeypatch.setattr("workflow.rollout.evaluator._seed_policy", lambda _seed: None)
    args = worker_eval.parse_worker_args(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--suites",
            "libero_goal",
            "--task-ids",
            "0",
            "--episodes-per-task",
            "2",
            "--max-steps",
            "1",
            "--num-steps-wait",
            "0",
            "--action-chunk",
            "1",
            "--num-inference-steps",
            "1",
            "--output",
            str(tmp_path / "dataset"),
            "--repo-id",
            "local/test",
        ]
    )

    results = worker_eval.evaluate(args)

    assert [episode["policy_seed"] for episode in results["episodes"]] == [7, 8]
    assert [episode["initial_state_index"] for episode in results["episodes"]] == [
        0,
        1,
    ]
    assert dataset.meta.total_episodes == 2
    assert len(dataset.frames) == 2
    assert dataset.stopped is True


def test_custom_manifest_validates_schema_array_and_per_state_hashes(tmp_path):
    states = np.arange(12, dtype=np.float64).reshape(3, 4)
    np.save(tmp_path / "states.npy", states)
    entries = [
        {
            "custom_initial_state_index": index,
            "state_vector_index": index,
            "simulator_state_sha256": array_hash(state),
            "scene_model_seed": 100 + index,
            "placement_seed": 100 + index,
            "suite": "libero_goal",
            "task_id": 0,
            "task": "open the middle drawer",
        }
        for index, state in enumerate(states)
    ]
    manifest = {
        "schema_version": 1,
        "kind": "custom_bddl_sampler_initial_states",
        "suite": "libero_goal",
        "task_id": 0,
        "task": "open the middle drawer",
        "count": 3,
        "state_file": "states.npy",
        "state_shape": [3, 4],
        "control_frequency": 20,
        "entries": entries,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "validation_report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "valid": True,
                "count": 3,
                "state_array_sha256": array_hash(states),
            }
        ),
        encoding="utf-8",
    )

    loaded_manifest, loaded_states, loaded_entries, digest = load_custom_initial_states(
        manifest_path
    )
    assert loaded_manifest["suite"] == "libero_goal"
    np.testing.assert_array_equal(loaded_states, states)
    assert sorted(loaded_entries) == [0, 1, 2]
    assert digest == array_hash(states)

    manifest["entries"][1]["simulator_state_sha256"] = "bad"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="state hash mismatch"):
        load_custom_initial_states(manifest_path)


def test_random_scene_cache_reuse_and_contract_mismatch(tmp_path):
    output = tmp_path / "cache"
    output.mkdir()
    states = np.arange(8, dtype=np.float64).reshape(2, 4)
    np.save(output / "states.npy", states)
    expected = {
        "schema_version": 1,
        "kind": "custom_bddl_sampler_initial_states",
        "suite": "libero_goal",
        "task_id": 0,
        "count": 2,
        "placement_seed_start": 100000,
        "control_frequency": 20,
        "non_training_stabilization_steps_baked_into_state": 10,
        "validation_hold_steps": 5,
        "allowed_maximum_stabilization_drift": 0.05,
    }
    manifest = {
        **expected,
        "state_shape": [2, 4],
        "entries": [
            {
                "custom_initial_state_index": index,
                "state_vector_index": index,
                "simulator_state_sha256": array_hash(state),
            }
            for index, state in enumerate(states)
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output / "validation_report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "valid": True,
                "count": 2,
                "unique_state_hashes": 2,
                "all_distinct_from_official_states": True,
                "all_initial_predicates_false": True,
                "all_sampler_replays_exact": True,
                "all_explicit_restores_exact": True,
                "maximum_stabilization_drift": 0.01,
                "allowed_maximum_stabilization_drift": 0.05,
                "state_array_sha256": array_hash(states),
            }
        ),
        encoding="utf-8",
    )

    assert existing_randomized_cache(output, expected) == manifest
    with pytest.raises(FileExistsError, match="contract mismatch"):
        existing_randomized_cache(output, {**expected, "control_frequency": 10})


def test_resume_requires_exact_unique_prefix():
    specs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    worker_eval.validate_resume_prefix(
        [
            {"trial_index": 0, "initial_state_index": 0},
            {"trial_index": 0, "initial_state_index": 1},
        ],
        specs,
    )
    with pytest.raises(ValueError, match="contiguous rollout prefix"):
        worker_eval.validate_resume_prefix(
            [
                {"trial_index": 0, "initial_state_index": 0},
                {"trial_index": 0, "initial_state_index": 0},
            ],
            specs,
        )


def test_worker_resume_validates_contract_metadata_and_loads_safe_prefix(
    tmp_path, monkeypatch
):
    output = tmp_path / "worker"
    (output / "meta").mkdir(parents=True)
    contract = {"videos_only": False, "checkpoint": "/checkpoint"}
    results = worker_eval._new_results(contract, task_description="task", output=output)
    results["episodes"] = [
        {"trial_index": 0, "initial_state_index": 0, "success": True}
    ]
    worker_eval._write_json(output / "results.json", results)
    (output / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 1}), encoding="utf-8"
    )

    class FakeDataset:
        def __init__(self, *, repo_id, root):
            assert repo_id == "local/test"
            assert root == output
            self.meta = SimpleNamespace(total_episodes=1)

    modules = {
        "lerobot": types.ModuleType("lerobot"),
        "lerobot.common": types.ModuleType("lerobot.common"),
        "lerobot.common.datasets": types.ModuleType("lerobot.common.datasets"),
        "lerobot.common.datasets.lerobot_dataset": types.ModuleType(
            "lerobot.common.datasets.lerobot_dataset"
        ),
    }
    modules["lerobot.common.datasets.lerobot_dataset"].LeRobotDataset = FakeDataset
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    args = SimpleNamespace(
        output=output,
        overwrite=False,
        videos_only=False,
        resume=True,
        repo_id="local/test",
    )
    dataset, resumed = worker_eval._open_output(
        args,
        contract=contract,
        task_description="task",
        specs=[(0, 0), (0, 1)],
    )
    assert dataset.meta.total_episodes == 1
    assert resumed["episodes"] == results["episodes"]

    with pytest.raises(ValueError, match="saved worker contract"):
        worker_eval._open_output(
            args,
            contract={**contract, "checkpoint": "/changed"},
            task_description="task",
            specs=[(0, 0), (0, 1)],
        )
    (output / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 2}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="metadata/results episode counts differ"):
        worker_eval._open_output(
            args,
            contract=contract,
            task_description="task",
            specs=[(0, 0), (0, 1)],
        )


def test_videos_only_output_writes_results_without_lerobot(tmp_path):
    output = tmp_path / "videos-worker"
    args = SimpleNamespace(
        output=output,
        overwrite=False,
        videos_only=True,
        resume=False,
    )
    contract = {"videos_only": True}
    dataset, results = worker_eval._open_output(
        args, contract=contract, task_description="task", specs=[(0, 0)]
    )
    assert dataset is None
    assert results["dataset_root"] is None
    assert (output / "results.json").is_file()
    assert (output / "videos" / "agentview").is_dir()
    assert (output / "videos" / "wrist").is_dir()


def test_task_summary_detects_missing_and_duplicate_episode_keys(tmp_path):
    config = _config(tmp_path, _base_yaml())
    campaign = campaign_eval.resolve_campaign(
        campaign_eval.parse_args(["--config", str(config), "--state-count", "2"])
    )
    task = campaign.tasks[0]
    workers = campaign_eval.build_workers(campaign, task, manifest=None)
    for worker in workers:
        worker["results"].parent.mkdir(parents=True)
    workers[0]["results"].write_text(
        json.dumps(
            {
                "episodes": [
                    {"initial_state_index": 0, "trial_index": 0, "success": True}
                ]
            }
        ),
        encoding="utf-8",
    )
    workers[1]["results"].write_text(
        json.dumps(
            {
                "episodes": [
                    {"initial_state_index": 0, "trial_index": 0, "success": False}
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = campaign_eval.summarize_task(task, workers, {0: 0, 1: 0})
    assert summary["duplicate_episode_keys"] is True
    assert summary["failed_workers"] == [1]


def test_dry_run_does_not_write_output(tmp_path):
    config = _config(tmp_path, _base_yaml())
    output = tmp_path / "must-not-exist"
    campaign = campaign_eval.resolve_campaign(
        campaign_eval.parse_args(
            ["--config", str(config), "--output", str(output), "--dry-run"]
        )
    )

    assert campaign_eval.run_campaign(campaign) == 0
    assert not output.exists()


def test_task_failure_is_recorded_and_later_tasks_still_run(tmp_path, monkeypatch):
    config = _config(tmp_path, _base_yaml())
    output = tmp_path / "serial-run"
    campaign = campaign_eval.resolve_campaign(
        campaign_eval.parse_args(["--config", str(config), "--output", str(output)])
    )
    calls: list[str] = []

    def fake_run_task(_campaign, task):
        calls.append(task.key)
        if task.key == "first":
            raise RuntimeError("intentional first-task failure")
        requested = len(campaign_eval.selected_indices(task))
        return {}, {
            "schema_version": 1,
            "key": task.key,
            "suite": task.suite,
            "task_id": task.task_id,
            "requested_episodes": requested,
            "episodes": requested,
            "successes": requested,
            "success_rate": 1.0,
            "failed_workers": [],
        }

    monkeypatch.setattr(campaign_eval, "run_task", fake_run_task)
    status = campaign_eval.run_campaign(campaign)

    assert status == 1
    assert calls == ["first", "libero_spatial-task009"]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_tasks"] == ["first"]
    assert summary["tasks"][1]["success_rate"] == 1.0
