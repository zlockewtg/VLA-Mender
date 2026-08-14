from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "vla-mender/scripts/repair/run_observation_only_reset_repairs.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "observation_only_reset_repairs", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visible_image_audit_recognizes_materializer_axis_convention(
    tmp_path: Path,
) -> None:
    module = _module()
    actual = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    expected_path = tmp_path / "reset.png"
    Image.fromarray(actual[:, ::-1]).save(expected_path)

    class LowLevel:
        @staticmethod
        def get_observation():
            return {"agentview": {"images": {"rgb": actual}}}

    class Environment:
        low_level_env = LowLevel()

    audit = module._visible_image_audit(Environment(), expected_path)
    assert audit["evidence_axis_convention"] == "horizontal_flip"
    assert audit["mean_absolute_error"] == 0.0
    assert audit["exact_pixel_fraction"] == 1.0


def test_private_reset_translation_stays_under_coordinator_root(tmp_path: Path) -> None:
    module = _module()
    reset_bank = tmp_path / "bank"
    source = reset_bank / "private_reset_states/state.npz"
    source.parent.mkdir(parents=True)
    np.savez_compressed(
        source,
        sim_state=np.arange(6, dtype=np.float64),
        gripper_controller_state=np.asarray([-1.0]),
    )
    job = {
        "job_id": "e000003-window_start-f000126",
        "episode_index": 3,
        "reset_frame_index": 126,
        "reset_state": "private_reset_states/state.npz",
    }
    record = {
        "suite": "libero_goal",
        "task_id": 0,
        "episode_index": 3,
        "scene_model_seed": 100003,
    }
    descriptor = module._normalized_private_descriptor(
        reset_bank=reset_bank,
        private_root=tmp_path / "private",
        job=job,
        record=record,
        source_state=np.zeros(8, dtype=np.float64),
        sim_signature={"state_width": 6},
        tolerance=1e-2,
    )
    normalized = Path(descriptor["private_simulator_state_path"])
    assert normalized.is_relative_to(tmp_path / "private")
    with np.load(normalized, allow_pickle=False) as payload:
        assert set(payload.files) == {"simulator_state", "gripper_current_action"}
        np.testing.assert_array_equal(payload["simulator_state"], np.arange(6))
        np.testing.assert_array_equal(payload["gripper_current_action"], [-1.0])


def test_private_reset_translation_rejects_path_escape(tmp_path: Path) -> None:
    module = _module()
    outside = tmp_path / "outside.npz"
    np.savez_compressed(outside, sim_state=np.zeros(2))
    job = {
        "job_id": "bad",
        "episode_index": 0,
        "reset_frame_index": 0,
        "reset_state": "../outside.npz",
    }
    try:
        module._normalized_private_descriptor(
            reset_bank=tmp_path / "bank",
            private_root=tmp_path / "private",
            job=job,
            record={
                "suite": "libero_goal",
                "task_id": 0,
                "episode_index": 0,
                "scene_model_seed": 0,
            },
            source_state=np.zeros(8),
            sim_signature={},
            tolerance=1e-2,
        )
    except ValueError as exc:
        assert "escapes reset bank" in str(exc)
    else:
        raise AssertionError("path escape was accepted")
