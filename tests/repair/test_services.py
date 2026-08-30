from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from workflow.research.services import TaskServiceManager


def _load_contact_graspnet_launcher():
    repo = Path(__file__).resolve().parents[2]
    tools_dir = repo / "vla-mender" / "tools"
    sys.path.insert(0, str(tools_dir))
    launcher_path = tools_dir / "launch_contact_graspnet_server.py"
    spec = importlib.util.spec_from_file_location(
        "repair_contact_graspnet_launcher", launcher_path
    )
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = launcher
    spec.loader.exec_module(launcher)
    return launcher


def test_generated_local_service_config_matches_launcher(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    manager = TaskServiceManager(
        tmp_path,
        project_root=repo,
        python=sys.executable,
        gpu_id=3,
        gpu_slot=1,
        profile="default",
        port_base=14000,
        port_stride=100,
        manage=True,
        keep_alive=True,
        startup_timeout_s=1,
        extra_env={},
    )
    manager._write_config()
    launcher_path = repo / "vla-mender" / "tools" / "launch_servers.py"
    spec = importlib.util.spec_from_file_location("repair_service_launcher", launcher_path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = launcher
    spec.loader.exec_module(launcher)
    parsed = launcher.parse_servers_from_yaml(str(manager.config_path))
    assert [value["server"] for value in parsed] == ["sam3", "graspnet", "pyroki"]
    assert [value["port"] for value in parsed] == [14114, 14115, 14116]


def test_tool_checkpoints_use_fixed_repository_paths(monkeypatch) -> None:
    repo = Path(__file__).resolve().parents[2]
    tools_dir = repo / "vla-mender" / "tools"
    monkeypatch.syspath_prepend(str(tools_dir))

    import checkpoint_paths

    assert checkpoint_paths.SAM3_CHECKPOINT_PATH == (
        repo / "third_party" / "sam3" / "checkpoints" / "sam3.pt"
    )
    assert checkpoint_paths.CONTACT_GRASPNET_CHECKPOINT_PATH == (
        repo
        / "third_party"
        / "contact_graspnet_pytorch"
        / "checkpoints"
        / "contact_graspnet"
        / "checkpoints"
        / "model.pt"
    )


def test_service_commands_ignore_checkpoint_environment(monkeypatch) -> None:
    repo = Path(__file__).resolve().parents[2]
    launcher_path = repo / "vla-mender" / "tools" / "launch_servers.py"
    spec = importlib.util.spec_from_file_location(
        "repair_service_launcher_fixed_checkpoints", launcher_path
    )
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = launcher
    spec.loader.exec_module(launcher)

    monkeypatch.setenv("SAM3_CHECKPOINT_PATH", "/external/sam3.pt")
    monkeypatch.setenv("CONTACT_GRASPNET_CHECKPOINT_DIR", "/external/graspnet")
    sam3_cmd = launcher._build_cmd(
        {
            "server": "sam3",
            "extra_args": {"checkpoint_path": "/yaml/sam3.pt"},
        },
        workers=1,
    )
    graspnet_cmd = launcher._build_cmd(
        {
            "server": "graspnet",
            "extra_args": {"checkpoint_dir": "/yaml/graspnet"},
        },
        workers=1,
    )

    assert "--checkpoint-path" not in sam3_cmd
    assert "--checkpoint-dir" not in graspnet_cmd


def test_contact_graspnet_estimator_uses_vendor_constructor(
    tmp_path: Path, monkeypatch
) -> None:
    launcher = _load_contact_graspnet_launcher()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.pt").touch()
    config = {"MODEL": {"model": "contact_graspnet"}}
    expected_model = object()
    parsed_states = []

    class FakeEstimator:
        def __init__(self, cfg):
            assert cfg is config
            self.model = expected_model

    class FakeCheckpointIO:
        def __init__(self, *, checkpoint_dir, model):
            assert checkpoint_dir == str(tmp_path / "checkpoints")
            assert model is expected_model

        def parse_state_dict(self, state_dict):
            parsed_states.append(state_dict)

    state_dict = {"model": "weights"}
    monkeypatch.setattr(launcher.torch, "load", lambda *_args, **_kwargs: state_dict)

    estimator = launcher.build_grasp_estimator(
        config,
        str(checkpoint_dir),
        torch.device("cuda:0"),
        estimator_cls=FakeEstimator,
        checkpoint_io_cls=FakeCheckpointIO,
    )

    assert isinstance(estimator, FakeEstimator)
    assert parsed_states == [state_dict]


def test_contact_graspnet_checkpoint_error_aborts_startup(
    tmp_path: Path, monkeypatch
) -> None:
    launcher = _load_contact_graspnet_launcher()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.pt").touch()

    class FakeEstimator:
        def __init__(self, _cfg):
            self.model = object()

    class FailingCheckpointIO:
        def __init__(self, **_kwargs):
            pass

        def parse_state_dict(self, _state_dict):
            raise RuntimeError("incompatible checkpoint")

    monkeypatch.setattr(launcher.torch, "load", lambda *_args, **_kwargs: {})

    with pytest.raises(RuntimeError, match="incompatible checkpoint"):
        launcher.build_grasp_estimator(
            {},
            str(checkpoint_dir),
            torch.device("cuda:0"),
            estimator_cls=FakeEstimator,
            checkpoint_io_cls=FailingCheckpointIO,
        )
