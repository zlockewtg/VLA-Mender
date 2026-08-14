#!/usr/bin/env python3
"""Check source-submodule assets and externally provisioned model weights."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _sam3_candidates() -> list[Path]:
    values: list[Path] = []
    if os.environ.get("SAM3_CHECKPOINT_PATH"):
        values.append(Path(os.environ["SAM3_CHECKPOINT_PATH"]).expanduser())
    if os.environ.get("HF_HOME"):
        values.append(Path(os.environ["HF_HOME"]).expanduser() / "hub/models--facebook--sam3")
    values.extend([Path("/mnt/public/tgy/ckpts/huggingface/hub/models--facebook--sam3"), Path.home() / ".cache/huggingface/hub/models--facebook--sam3"])
    return values

def _sam3_exists(path: Path) -> bool:
    return ((path.is_file() and path.name == "sam3.pt") or (path.is_dir() and ((path / "sam3.pt").is_file() or any((path / "snapshots").glob("*/sam3.pt")))))

def main() -> int:
    checks = {
        "LIBERO assets": ROOT / "third_party/LIBERO-PRO/libero/libero/assets",
        "robosuite model assets": ROOT / "third_party/libero_dependencies/robosuite/robosuite/models/assets",
        "Contact-GraspNet config": ROOT / "third_party/contact_graspnet_pytorch/checkpoints/contact_graspnet/config.yaml",
        "Contact-GraspNet model": ROOT / "third_party/contact_graspnet_pytorch/checkpoints/contact_graspnet/checkpoints/model.pt",
    }
    missing = False
    for label, path in checks.items():
        present = path.exists()
        status = "present" if present else "missing"
        print(f"{status:7} {label}: {path}")
        missing |= not present
    sam3 = next((p.resolve() for p in _sam3_candidates() if _sam3_exists(p)), None)
    status = "present" if sam3 else "missing"
    detail = str(sam3) if sam3 else "set SAM3_CHECKPOINT_PATH"
    print(f"{status:7} SAM3 checkpoint: {detail}")
    return int(missing or sam3 is None)

if __name__ == "__main__":
    raise SystemExit(main())
