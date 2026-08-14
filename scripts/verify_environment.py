#!/usr/bin/env python3
"""Verify the environment-only VLA-Mender installation without runtime assets."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _distribution_version(name: str) -> str:
    return importlib.metadata.version(name)


def _import_from_vendor(module_name: str, vendor_root: Path):
    module = importlib.import_module(module_name)
    module_file = Path(module.__file__).resolve() if module.__file__ else None
    if module_file is None or not module_file.is_relative_to(vendor_root.resolve()):
        raise RuntimeError(
            f"{module_name} resolved outside {vendor_root}: {module_file}"
        )
    return module


def _configure_isolated_libero() -> tempfile.TemporaryDirectory[str]:
    temp_dir = tempfile.TemporaryDirectory(prefix="vla-mender-libero-")
    config_root = Path(temp_dir.name)
    benchmark_root = ROOT / "third_party/LIBERO-PRO/libero/libero"
    config = "\n".join(
        [
            f"benchmark_root: {benchmark_root}",
            f"bddl_files: {benchmark_root / 'bddl_files'}",
            f"init_states: {benchmark_root / 'init_files'}",
            f"datasets: {benchmark_root.parent / 'datasets'}",
            f"assets: {benchmark_root / 'assets'}",
            "",
        ]
    )
    (config_root / "config.yaml").write_text(config, encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    return temp_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero",
        action="store_true",
        help="also verify packages installed by the libero extra",
    )
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"expected Python 3.12, got {sys.version.split()[0]}")

    # Do not allow a machine-level checkout to mask a broken editable install.
    os.environ.pop("PYTHONPATH", None)

    contact_root = ROOT / "third_party/contact_graspnet_pytorch"
    sam3_root = ROOT / "third_party/sam3"
    _import_from_vendor("contact_graspnet_pytorch", contact_root)
    _import_from_vendor("sam3", sam3_root)
    estimator_module = importlib.import_module(
        "contact_graspnet_pytorch.contact_grasp_estimator"
    )
    if not hasattr(estimator_module, "GraspEstimator"):
        raise RuntimeError("Contact-GraspNet does not export GraspEstimator")
    importlib.import_module("pyroki")

    checked = {
        "contact-graspnet-pytorch": _distribution_version(
            "contact-graspnet-pytorch"
        ),
        "sam3": _distribution_version("sam3"),
        "pyroki": _distribution_version("pyroki"),
    }

    if args.libero:
        isolated_config = _configure_isolated_libero()
        try:
            libero_root = ROOT / "third_party/LIBERO-PRO"
            robosuite_root = ROOT / "third_party/libero_dependencies/robosuite"
            _import_from_vendor("libero", libero_root)
            _import_from_vendor("robosuite", robosuite_root)
            importlib.import_module("libero.envs").OffScreenRenderEnv
            checked.update(
                {
                    "libero": _distribution_version("libero"),
                    "robosuite": _distribution_version("robosuite"),
                }
            )
        finally:
            isolated_config.cleanup()

    print("environment verification passed")
    for name, version in sorted(checked.items()):
        print(f"  {name}=={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

