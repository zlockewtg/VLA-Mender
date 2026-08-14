"""Pinned OpenPI runtime backend.

OpenPI is intentionally imported lazily.  Prompt rendering and diagnosis can
therefore run in a lightweight process, while rollout workers use the exact
vendored checkout and checkpoint contract selected by the experiment.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .parameters import ExperimentSettings


class OpenPIBackendError(RuntimeError):
    pass


def default_openpi_root() -> Path:
    module = Path(__file__).resolve()
    for parent in module.parents:
        candidate = parent / "third_party" / "openpi"
        if candidate.is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise OpenPIBackendError("cannot locate vendored third_party/openpi checkout")


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _pinned_commit(root: Path) -> str | None:
    pin = root.parent / "openpi.commit"
    if pin.is_file():
        value = pin.read_text(encoding="utf-8").strip()
        return value or None
    return None


def openpi_runtime_preflight(
    settings: ExperimentSettings, *, require_checkpoint: bool = True
) -> dict[str, Any]:
    """Validate source pin, environment selection and checkpoint prerequisites."""

    backend = settings.backend
    root = (
        Path(backend.openpi_source) if backend.openpi_source else default_openpi_root()
    )
    root = root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        raise OpenPIBackendError(f"OpenPI source checkout is invalid: {root}")
    actual = _git_commit(root)
    expected = backend.openpi_commit or _pinned_commit(root)
    if expected and actual and actual != expected:
        raise OpenPIBackendError(
            f"OpenPI checkout is not pinned: expected {expected}, found {actual} ({root})"
        )
    env_value = backend.openpi_environment or os.environ.get("VLA_MENDER_OPENPI_ENV")
    environment = (
        Path(env_value).expanduser().resolve()
        if env_value
        else Path(sys.prefix).resolve()
    )
    if backend.openpi_environment and not environment.exists():
        raise OpenPIBackendError(
            f"configured OpenPI environment does not exist: {environment}"
        )
    current_environment = Path(sys.prefix).resolve()
    if backend.openpi_environment and current_environment != environment:
        raise OpenPIBackendError(
            "the experiment selects an OpenPI environment but is running under a different "
            f"interpreter: configured={environment}, current={current_environment}. "
            f"Launch with {environment / 'bin/python'} (and install the LIBERO runtime there)."
        )
    checkpoint = settings.task.checkpoint.resolve()
    weight_path = checkpoint / "model.safetensors"
    if backend.openpi_norm_stats:
        configured_stats = Path(backend.openpi_norm_stats).expanduser().resolve()
        norm_stats = (
            configured_stats.parent
            if configured_stats.name == "norm_stats.json"
            else configured_stats
        )
    else:
        norm_candidates = (
            checkpoint / "assets" / "physical-intelligence" / "libero",
            checkpoint / "physical-intelligence" / "libero",
        )
        norm_stats = next(
            (
                candidate
                for candidate in norm_candidates
                if (candidate / "norm_stats.json").is_file()
            ),
            norm_candidates[0],
        )
    if require_checkpoint:
        if not weight_path.is_file():
            raise OpenPIBackendError(
                f"OpenPI checkpoint is missing model.safetensors: {weight_path}"
            )
        if not (norm_stats / "norm_stats.json").is_file():
            raise OpenPIBackendError(
                "OpenPI checkpoint is missing LIBERO normalization statistics: "
                f"{norm_stats}. Set backend.openpi_norm_stats or add the file to the checkpoint."
            )
    return {
        "name": "openpi",
        "source": str(root),
        "commit": actual,
        "expected_commit": expected,
        "environment": str(environment),
        "python": sys.executable,
        "checkpoint": str(checkpoint),
        "weights": str(weight_path),
        "norm_stats": str(norm_stats / "norm_stats.json"),
    }


class OpenPIBackend:
    def __init__(
        self,
        settings: ExperimentSettings,
        device: str,
        *,
        compile_mode: str | None = None,
    ):
        self.settings = settings
        self.device = device
        self.compile_mode = compile_mode
        self._policy: Any | None = None
        self.runtime = openpi_runtime_preflight(settings)

    @property
    def policy(self) -> Any:
        if self._policy is None:
            self._policy = self._load_policy()
        return self._policy

    def infer(self, observation: dict[str, Any]) -> Any:
        return self.policy.infer(observation)

    def _load_policy(self) -> Any:
        root = Path(self.runtime["source"])
        for source in (root / "src", root / "packages" / "openpi-client" / "src"):
            if source.is_dir() and str(source) not in sys.path:
                sys.path.insert(0, str(source))
        import safetensors.torch
        from openpi import transforms
        from openpi.models_pytorch import pi0_pytorch
        from openpi.policies import policy as policy_module
        from openpi.shared import normalize
        from openpi.training import config as training_config

        checkpoint = self.settings.task.checkpoint
        train_cfg = training_config.get_config(self.settings.task.policy_config)
        model_cfg = train_cfg.model
        if self.compile_mode is not None:
            supported = getattr(model_cfg, "__dataclass_fields__", {})
            if "pytorch_compile_mode" not in supported:
                if self.compile_mode != "none":
                    raise OpenPIBackendError(
                        f"OpenPI model config {type(model_cfg).__name__} does not "
                        f"support compile mode {self.compile_mode!r}"
                    )
            else:
                model_cfg = dataclasses.replace(
                    model_cfg,
                    pytorch_compile_mode=(
                        None if self.compile_mode == "none" else self.compile_mode
                    ),
                )
                train_cfg = dataclasses.replace(train_cfg, model=model_cfg)
        model = pi0_pytorch.PI0Pytorch(config=model_cfg)
        missing, unexpected = safetensors.torch.load_model(
            model, str(checkpoint / "model.safetensors"), strict=False
        )
        missing = set(missing) - {
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        }
        if missing:
            raise OpenPIBackendError(
                f"checkpoint tensor mismatch: missing={sorted(missing)}"
            )
        if unexpected:
            logging.getLogger(__name__).warning(
                "ignoring checkpoint tensors not used by the policy: %s",
                sorted(unexpected),
            )
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        norm_stats = Path(self.runtime["norm_stats"]).parent
        data_cfg = train_cfg.data.create(train_cfg.assets_dirs, model_cfg)
        stats = normalize.load(norm_stats)
        return policy_module.Policy(
            model,
            transforms=[
                transforms.InjectDefaultPrompt(None),
                *data_cfg.data_transforms.inputs,
                transforms.Normalize(stats, use_quantiles=data_cfg.use_quantile_norm),
                *data_cfg.model_transforms.inputs,
            ],
            output_transforms=[
                *data_cfg.model_transforms.outputs,
                transforms.Unnormalize(stats, use_quantiles=data_cfg.use_quantile_norm),
                *data_cfg.data_transforms.outputs,
            ],
            sample_kwargs={"num_steps": self.settings.rollout.inference_steps},
            metadata=train_cfg.policy_metadata,
            is_pytorch=True,
            pytorch_device=self.device,
        )
