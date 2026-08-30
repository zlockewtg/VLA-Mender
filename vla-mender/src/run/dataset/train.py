"""Preflight or launch OpenPI post-training from the dataset YAML."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any

import yaml

from workflow.dataset.run import prepare_dataset_run
from workflow.training.config import load_training_config, validate_training_inputs


def _training_settings(path: Path) -> Path:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and {"run", "source", "build"} <= set(raw):
        manifest = prepare_dataset_run(path)
        training = manifest.get("training_config")
        if not training:
            raise ValueError(f"dataset settings have no training section: {path}")
        return Path(str(training)).resolve()
    return path.resolve()


def _command(settings: Path) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    config = load_training_config(settings)
    preflight = validate_training_inputs(config)
    command = [
        str(config.openpi_environment / "bin/torchrun"),
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={len(config.gpus)}",
        "-m",
        "workflow.training.openpi_runner",
        "--settings",
        str(config.settings_path),
    ]
    env = os.environ.copy()
    project_src = str(Path(__file__).resolve().parents[2])
    openpi_src = str(config.openpi_source / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [project_src, openpi_src, *([existing] if existing else [])]
    )
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in config.gpus)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return command, env, preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all inputs and print the exact torchrun command without launching it.",
    )
    args = parser.parse_args(argv)
    settings = _training_settings(args.settings.expanduser().resolve())
    command, env, preflight = _command(settings)
    result = {
        **preflight,
        "settings": str(settings),
        "settings_sha256": hashlib.sha256(settings.read_bytes()).hexdigest(),
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "command": command,
        "command_shell": shlex.join(command),
        "launched": not args.dry_run,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
