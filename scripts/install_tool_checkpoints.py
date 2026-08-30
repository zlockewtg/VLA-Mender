#!/usr/bin/env python3
"""Provision tool checkpoints at their repository-local runtime paths."""

from __future__ import annotations

import os
import shutil
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


SAM3_REPO_ID = "facebook/sam3"
SAM3_FILENAME = "sam3.pt"
MIN_SAM3_BYTES = 1_000_000_000
MIN_CONTACT_GRASPNET_BYTES = 1_000_000


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.installing")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _valid_file(path: Path, minimum_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size >= minimum_bytes


def _cached_sam3_candidates() -> list[Path]:
    candidates = [
        Path("/mnt/public/tgy/ckpts/huggingface/hub/models--facebook--sam3"),
        Path.home() / ".cache" / "huggingface" / "hub" / "models--facebook--sam3",
    ]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.insert(
            0,
            Path(hf_home).expanduser() / "hub" / "models--facebook--sam3",
        )
    resolved: list[Path] = []
    for root in candidates:
        resolved.append(root / SAM3_FILENAME)
        resolved.extend(sorted((root / "snapshots").glob(f"*/{SAM3_FILENAME}")))
    return resolved


def _install_sam3() -> None:
    if _valid_file(SAM3_CHECKPOINT_PATH, MIN_SAM3_BYTES):
        print(f"present SAM3 checkpoint: {SAM3_CHECKPOINT_PATH}")
        return

    source = next(
        (
            candidate.resolve()
            for candidate in _cached_sam3_candidates()
            if _valid_file(candidate, MIN_SAM3_BYTES)
        ),
        None,
    )
    if source is None:
        from huggingface_hub import hf_hub_download

        print(f"downloading {SAM3_REPO_ID}/{SAM3_FILENAME}")
        source = Path(
            hf_hub_download(repo_id=SAM3_REPO_ID, filename=SAM3_FILENAME)
        ).resolve()

    print(f"installing SAM3 checkpoint: {source} -> {SAM3_CHECKPOINT_PATH}")
    _atomic_copy(source, SAM3_CHECKPOINT_PATH)
    if not _valid_file(SAM3_CHECKPOINT_PATH, MIN_SAM3_BYTES):
        raise RuntimeError(f"invalid SAM3 checkpoint: {SAM3_CHECKPOINT_PATH}")


def _install_contact_graspnet() -> None:
    if not CONTACT_GRASPNET_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Contact-GraspNet config is missing after submodule checkout: "
            f"{CONTACT_GRASPNET_CONFIG_PATH}"
        )
    if _valid_file(
        CONTACT_GRASPNET_CHECKPOINT_PATH,
        MIN_CONTACT_GRASPNET_BYTES,
    ):
        print(
            "present Contact-GraspNet checkpoint: "
            f"{CONTACT_GRASPNET_CHECKPOINT_PATH}"
        )
        return

    candidates = [
        ROOT.parent
        / "capx-aspire"
        / "capx"
        / "third_party"
        / "contact_graspnet_pytorch"
        / "checkpoints"
        / "contact_graspnet"
        / "checkpoints"
        / "model.pt",
    ]
    source = next(
        (
            candidate.resolve()
            for candidate in candidates
            if _valid_file(candidate, MIN_CONTACT_GRASPNET_BYTES)
        ),
        None,
    )
    if source is None:
        raise FileNotFoundError(
            "Contact-GraspNet model.pt is missing from its submodule and no "
            f"local source was found: {CONTACT_GRASPNET_CHECKPOINT_PATH}"
        )
    print(
        "installing Contact-GraspNet checkpoint: "
        f"{source} -> {CONTACT_GRASPNET_CHECKPOINT_PATH}"
    )
    _atomic_copy(source, CONTACT_GRASPNET_CHECKPOINT_PATH)


def main() -> int:
    _install_sam3()
    _install_contact_graspnet()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
