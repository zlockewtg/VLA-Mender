#!/usr/bin/env python3
"""Check source-submodule assets and repository-local model weights."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "vla-mender" / "tools"
sys.path.insert(0, str(TOOLS))

from checkpoint_paths import (  # noqa: E402
    CONTACT_GRASPNET_CHECKPOINT_PATH,
    CONTACT_GRASPNET_CONFIG_PATH,
    SAM3_CHECKPOINT_PATH,
)


def main() -> int:
    checks = {
        "LIBERO assets": ROOT / "third_party/LIBERO-PRO/libero/libero/assets",
        "robosuite model assets": (
            ROOT
            / "third_party/libero_dependencies/robosuite/robosuite/models/assets"
        ),
        "SAM3 checkpoint": SAM3_CHECKPOINT_PATH,
        "Contact-GraspNet config": CONTACT_GRASPNET_CONFIG_PATH,
        "Contact-GraspNet model": CONTACT_GRASPNET_CHECKPOINT_PATH,
    }
    missing = False
    for label, path in checks.items():
        present = path.exists()
        status = "present" if present else "missing"
        print(f"{status:7} {label}: {path}")
        missing |= not present
    return int(missing)


if __name__ == "__main__":
    raise SystemExit(main())
