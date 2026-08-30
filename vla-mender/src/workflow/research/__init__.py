"""Prompt-driven, multi-task repair research infrastructure."""

from .campaign import RepairCampaign, TaskSession
from .config import RepairConfig, RepairConfigError, load_repair_config, resolve_repair_inputs
from .state import (
    ExplorationReviewRequired,
    FailureModeSession,
    PromotionResult,
    RepairCandidate,
    RepairRound,
    SoftBudgetReviewRequired,
)

__all__ = [
    "RepairCampaign",
    "RepairConfig",
    "RepairConfigError",
    "FailureModeSession",
    "ExplorationReviewRequired",
    "PromotionResult",
    "RepairCandidate",
    "RepairRound",
    "SoftBudgetReviewRequired",
    "TaskSession",
    "load_repair_config",
    "resolve_repair_inputs",
]
