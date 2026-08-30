"""Repository-local checkpoint paths shared by tool launchers and installers."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAM3_CHECKPOINT_PATH = (
    PROJECT_ROOT / "third_party" / "sam3" / "checkpoints" / "sam3.pt"
)

CONTACT_GRASPNET_ROOT = PROJECT_ROOT / "third_party" / "contact_graspnet_pytorch"
CONTACT_GRASPNET_CONFIG_PATH = (
    CONTACT_GRASPNET_ROOT / "checkpoints" / "contact_graspnet" / "config.yaml"
)
CONTACT_GRASPNET_CHECKPOINT_DIR = (
    CONTACT_GRASPNET_ROOT
    / "checkpoints"
    / "contact_graspnet"
    / "checkpoints"
)
CONTACT_GRASPNET_CHECKPOINT_PATH = CONTACT_GRASPNET_CHECKPOINT_DIR / "model.pt"
