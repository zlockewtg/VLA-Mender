from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from workflow.dataset.builder import build_dataset, valid_start_indices
from workflow.dataset.config import load_config
from workflow.dataset.continuity import (
    flow_metrics,
    model_numeric_sha256,
    signature_mismatches,
    simulator_state_sha256,
    validate_simulator_evidence,
)


def _png(value: int) -> dict[str, object]:
    rgb = np.full((8, 8, 3), value, dtype=np.uint8)
    ok, payload = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return {"bytes": payload.tobytes(), "path": None}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _synthetic_inputs(tmp_path: Path) -> Path:
    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    output_schema = pa.schema(
        [
            pa.field("image", image_type),
            pa.field("state", pa.list_(pa.float32(), 2)),
            pa.field("actions", pa.list_(pa.float32(), 2)),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
    )
    reference = tmp_path / "reference"
    reference_data = reference / "data/chunk-000"
    reference_data.mkdir(parents=True)
    reference_table = pa.Table.from_pydict(
        {
            "image": [_png(10)],
            "state": [[0.0, 0.0]],
            "actions": [[0.0, 0.0]],
            "timestamp": [0.0],
            "frame_index": [0],
            "episode_index": [0],
            "index": [0],
            "task_index": [3],
        },
        schema=output_schema,
    )
    pq.write_table(reference_table, reference_data / "episode_000000.parquet")
    _write_json(
        reference / "meta/info.json",
        {
            "chunks_size": 1000,
            "features": {
                "state": {"dtype": "float32", "shape": [2], "names": ["state"]},
                "actions": {"dtype": "float32", "shape": [2], "names": ["actions"]},
            },
            "fps": 20,
            "total_tasks": 1,
        },
    )
    (reference / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 3, "task": "synthetic task"}) + "\n"
    )

    prefix = tmp_path / "prefix.parquet"
    prefix_table = pa.Table.from_pydict(
        {
            "image": [_png(10), _png(20), _png(30), _png(40)],
            "state": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            "actions": [[0.1, 0.0]] * 4,
            "timestamp": [0.0, 0.05, 0.1, 0.15],
            "frame_index": [0, 1, 2, 3],
            "episode_index": [0] * 4,
            "index": [0, 1, 2, 3],
            "task_index": [3] * 4,
        },
        schema=output_schema,
    )
    pq.write_table(prefix_table, prefix)

    repair = tmp_path / "repair.parquet"
    repair_table = pa.table(
        {
            "repair_image": pa.array([_png(40), _png(50), _png(60), _png(60)], type=image_type),
            "observation.state": pa.array(
                [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
                type=pa.list_(pa.float32(), 2),
            ),
            "action": pa.array([[0.2, 0.0]] * 4, type=pa.list_(pa.float32(), 2)),
            "action_is_valid": [True, True, True, False],
            "next.reward": [0.0, 0.0, 1.0, 0.0],
            "next.done": [False, False, True, True],
        }
    )
    pq.write_table(repair_table, repair)

    manifest = tmp_path / "episodes.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "episodes": [
                {
                    "source_episode_id": "source-a",
                    "restart_frame": 3,
                    "task_index": 3,
                    "task": "synthetic task",
                    "prefix": {
                        "parquet": str(prefix),
                        "images": {"image": {"column": "image"}},
                    },
                    "repair": {
                        "parquet": str(repair),
                        "images": {"image": {"column": "repair_image"}},
                    },
                    "metadata": {"selector": "unit-test"},
                }
            ],
        },
    )
    config = tmp_path / "build.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "output": str(tmp_path / "output"),
                "reference_dataset": str(reference),
                "episodes_manifest": str(manifest),
                "dataset_source": "unit-test",
                "fps": 20,
                "action_horizon": 2,
                "pre_guard_frames": 1,
                "post_guard_frames": 1,
                "cameras": {
                    "image": {
                        "prefix_column": "image",
                        "width": 8,
                        "height": 8,
                    }
                },
                "action": {"state_dim": 2, "action_dim": 2},
                "continuity": {
                    "require_simulator_evidence": False,
                    "max_flow_median_px": 0.01,
                    "max_flow_p90_px": 0.01,
                },
            },
            sort_keys=False,
        )
    )
    return config


def test_end_to_end_embedded_build(tmp_path: Path) -> None:
    config = load_config(_synthetic_inputs(tmp_path))
    source = pq.read_table(tmp_path / "prefix.parquet")
    report = build_dataset(config)
    assert report["valid"] is True
    assert report["episode_count"] == 1
    assert report["frame_count"] == 6
    assert report["vla_prefix_frame_count"] == 3
    assert report["repair_suffix_frame_count"] == 3
    assert report["valid_start_count"] == 1
    assert report["max_splice_state_abs_error"] == 0.0
    assert report["demo_video_count"] == 1
    assert (config.output / "meta/provenance/build_config.yaml").is_file()
    demo_manifest = json.loads(
        (config.output / "meta/visualization/trajectory_demos/manifest.json").read_text()
    )
    assert demo_manifest["demo_count"] == 1
    assert (config.output / "meta/visualization/trajectory_demos/demo_01_episode_000.mp4").is_file()
    output = pq.read_table(config.output / "data/chunk-000/episode_000000.parquet")
    assert output["image"][0].as_py() == source["image"][0].as_py()


def test_repair_only_episode_has_no_splice_or_intervention_guard(tmp_path: Path) -> None:
    config_path = _synthetic_inputs(tmp_path)
    manifest_path = tmp_path / "episodes.json"
    manifest = json.loads(manifest_path.read_text())
    row = manifest["episodes"][0]
    row["mode"] = "repair_only"
    row.pop("restart_frame")
    row.pop("prefix")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    config = load_config(config_path)
    report = build_dataset(config)
    assert report["valid"] is True
    assert report["episode_count"] == 1
    assert report["spliced_episode_count"] == 0
    assert report["repair_only_episode_count"] == 1
    assert report["repair_only_frame_count"] == 3
    assert report["frame_count"] == 3
    assert report["vla_prefix_frame_count"] == 0
    assert report["valid_start_count"] == 1

    build = json.loads((config.output / "meta/build_manifest.json").read_text())
    episode = build["episodes"][0]
    assert episode["episode_mode"] == "repair_only"
    assert episode["splice_frame_index"] is None
    assert episode["splice_state_max_abs_error"] is None
    assert episode["source_prefix_parquet"] is None
    trainable = json.loads(
        (config.output / "meta/trainable_index_manifest.json").read_text()
    )
    assert [row["trainable"] for row in trainable["frames"]] == [True, True, False]
    assert all(not row["intervention_guard"] for row in trainable["frames"])
    assert all(not row["splice_boundary_before"] for row in trainable["frames"])
    assert all(row["segment_role"] == "repair_only" for row in trainable["frames"])


def test_native_splice_training_allows_action_chunks_to_cross_boundary(
    tmp_path: Path,
) -> None:
    config_path = _synthetic_inputs(tmp_path)
    value = yaml.safe_load(config_path.read_text())
    value["pre_guard_frames"] = 0
    value["post_guard_frames"] = 0
    value["allow_splice_crossing_action_chunks"] = True
    config_path.write_text(yaml.safe_dump(value, sort_keys=False))

    config = load_config(config_path)
    report = build_dataset(config)
    assert report["valid"] is True
    assert report["valid_start_count"] == 4

    trainable = json.loads(
        (config.output / "meta/trainable_index_manifest.json").read_text()
    )
    assert [row["trainable"] for row in trainable["frames"]] == [
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert len({row["continuous_segment_id"] for row in trainable["frames"]}) == 1
    assert valid_start_indices(trainable["frames"], action_horizon=2) == [0, 1, 2, 3]

    build = json.loads((config.output / "meta/build_manifest.json").read_text())
    assert build["policies"]["splice_crossing_action_chunks_trainable"] is True


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = _synthetic_inputs(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["typo_policy"] = True
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="unknown top-level keys"):
        load_config(path)


def test_build_rejects_action_width_different_from_reference(tmp_path: Path) -> None:
    path = _synthetic_inputs(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["action"]["action_dim"] = 3
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(
        RuntimeError,
        match=r"configured actions width 3 differs from reference width 2",
    ):
        build_dataset(load_config(path))


def test_model_numeric_hash_detects_static_scene_motion() -> None:
    class Model:
        body_pos = np.zeros((2, 3), dtype=np.float64)

    model = Model()
    before = model_numeric_sha256(model, fields=("body_pos",))
    model.body_pos[1, 0] = 0.25
    assert model_numeric_sha256(model, fields=("body_pos",)) != before


def test_signature_comparison_rejects_fields_missing_on_both_sides() -> None:
    assert signature_mismatches({}, {}, ("model_numeric_sha256",)) == {
        "model_numeric_sha256": ("<missing>", "<missing>")
    }


def test_identical_images_have_zero_flow() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    metrics = flow_metrics(image, image.copy())
    assert metrics["mae"] == 0.0
    assert metrics["flow_p90_px"] == 0.0


def test_source_runtime_check_preserves_virtualenv_boundary(tmp_path: Path) -> None:
    runtime = tmp_path / "venv"
    python = runtime / "bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path("/usr/bin/python3"))
    robosuite = runtime / "lib/python3.12/site-packages/robosuite/__init__.py"
    robosuite.parent.mkdir(parents=True)
    robosuite.write_text("")
    signature = {
        "nq": 1,
        "python_executable": str(python),
        "robosuite_module_path": str(robosuite),
    }
    reset = tmp_path / "reset.npz"
    row0 = tmp_path / "row0.npz"
    state = np.asarray([1.0], dtype=np.float64)
    np.savez_compressed(reset, simulator_state=state)
    np.savez_compressed(row0, simulator_state=state)
    digest = simulator_state_sha256(state)
    descriptor = tmp_path / "descriptor.json"
    attempt = tmp_path / "attempt.json"
    result = tmp_path / "result.json"
    _write_json(
        descriptor,
        {
            "sim_signature": signature,
            "private_simulator_state_path": str(reset),
        },
    )
    _write_json(
        attempt,
        {
            "transfer": {
                "sim_signature": signature,
                "strict_sim_signature_verified": True,
                "strict_sim_signature_fields": ["nq"],
            }
        },
    )
    _write_json(
        result,
        {
            "repair_row0_simulator_state": {
                "verified": True,
                "path": str(row0),
                "key": "simulator_state",
                "sha256": digest,
                "reset_sha256": digest,
            }
        },
    )
    report = validate_simulator_evidence(
        {
            "reset_descriptor": str(descriptor),
            "attempt_manifest": str(attempt),
            "result": str(result),
            "expected_source_python": str(python),
        },
        fields=("nq",),
        tolerance=1e-12,
    )
    assert report["expected_source_python"] == str(python)


def test_observation_only_public_restore_attestation(tmp_path: Path) -> None:
    state = np.asarray([1.0, 2.0], dtype=np.float64)
    digest = simulator_state_sha256(state)
    reset = tmp_path / "reset.npz"
    row0 = tmp_path / "row0.npz"
    np.savez_compressed(reset, simulator_state=state)
    np.savez_compressed(row0, simulator_state=state)
    signature = {"nq": 1, "state_width": 2}
    descriptor = tmp_path / "descriptor.json"
    attempt = tmp_path / "attempt.json"
    result = tmp_path / "result.json"
    _write_json(
        descriptor,
        {
            "suite": "libero_goal",
            "task_id": 0,
            "suite_episode_index": 3,
            "scene_model_seed": 100003,
            "restored_frame_index": 126,
            "sim_signature": signature,
            "private_simulator_state_path": str(reset),
        },
    )
    _write_json(
        attempt,
        {
            "public_restore_audit": {
                "suite": "libero_goal",
                "task_id": 0,
                "suite_episode_index": 3,
                "scene_model_seed": 100003,
                "frame_index": 126,
                "simulator_state_width": 2,
                "simulator_state_sha256": digest,
                "strict_sim_signature_verified": True,
                "strict_sim_signature_fields": ["nq", "state_width"],
            }
        },
    )
    _write_json(
        result,
        {
            "repair_row0_simulator_state": {
                "verified": True,
                "path": str(row0),
                "key": "simulator_state",
                "sha256": digest,
                "reset_sha256": digest,
            }
        },
    )
    report = validate_simulator_evidence(
        {
            "reset_descriptor": str(descriptor),
            "attempt_manifest": str(attempt),
            "result": str(result),
        },
        fields=("nq", "state_width"),
        tolerance=1e-12,
    )
    assert report["verified"] is True
    assert report["verification_mode"] == "trusted_runtime_attestation"


def test_observation_only_attestation_rejects_wrong_episode(tmp_path: Path) -> None:
    state = np.asarray([1.0], dtype=np.float64)
    digest = simulator_state_sha256(state)
    reset = tmp_path / "reset.npz"
    row0 = tmp_path / "row0.npz"
    np.savez_compressed(reset, simulator_state=state)
    np.savez_compressed(row0, simulator_state=state)
    descriptor = tmp_path / "descriptor.json"
    attempt = tmp_path / "attempt.json"
    result = tmp_path / "result.json"
    common = {
        "suite": "libero_goal",
        "task_id": 0,
        "scene_model_seed": 100003,
    }
    _write_json(
        descriptor,
        {
            **common,
            "suite_episode_index": 3,
            "restored_frame_index": 126,
            "sim_signature": {"nq": 1},
            "private_simulator_state_path": str(reset),
        },
    )
    _write_json(
        attempt,
        {
            "public_restore_audit": {
                **common,
                "suite_episode_index": 4,
                "frame_index": 126,
                "simulator_state_width": 1,
                "simulator_state_sha256": digest,
                "strict_sim_signature_verified": True,
                "strict_sim_signature_fields": ["nq"],
            }
        },
    )
    _write_json(
        result,
        {
            "repair_row0_simulator_state": {
                "verified": True,
                "path": str(row0),
                "sha256": digest,
                "reset_sha256": digest,
            }
        },
    )
    with pytest.raises(RuntimeError, match="public audit identity mismatch"):
        validate_simulator_evidence(
            {
                "reset_descriptor": str(descriptor),
                "attempt_manifest": str(attempt),
                "result": str(result),
            },
            fields=("nq",),
            tolerance=1e-12,
        )
