#!/usr/bin/env python3
"""Build a compact, traceable index of experiment results under a data root.

The index deliberately does not copy per-frame or per-episode payloads.  Every
source result JSON remains addressable through ``result_artifacts[].path`` and
is protected by a SHA-256 digest.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULT_TOKEN = re.compile(r"(^|[_-])(result|results|summary|report|metrics?)($|[_-])", re.I)
CHECKPOINT_KEY = re.compile(r"(^|_)(checkpoint|ckpt)(_|$)", re.I)
METRIC_KEY = re.compile(
    r"(success|loss|rate|mean|median|std|score|accuracy|return|count|total|steps|frames|episodes|error)",
    re.I,
)
SKIP_METRIC_KEY = re.compile(r"(path|sha|digest|seed|index|id|port|time_ms|timestamp)", re.I)
OUTPUT_NAME = "experiment_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/public/tgy/data"))
    parser.add_argument(
        "--output", type=Path, default=Path("/mnt/public/tgy/data") / OUTPUT_NAME
    )
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=Path("/mnt/public/tgy/capx-aspire/aspire/vla_mender/openpi"),
        help="OpenPI tree containing src/openpi/training/config.py and optional wandb runs",
    )
    return parser.parse_args()


def json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool, int, float))


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return repr(value)


def selected_fields(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        if hasattr(value, name):
            result[name] = jsonable(getattr(value, name))
    return result


def config_snapshot(config: Any, source: str) -> dict[str, Any]:
    data = config.data
    base = getattr(data, "base_config", None)
    return {
        "source": source,
        "config_name": config.name,
        "dataset": {
            "repo_id": jsonable(getattr(data, "repo_id", None)),
            "episodes": jsonable(getattr(data, "episodes", None)),
            "asset_id": jsonable(getattr(getattr(data, "assets", None), "asset_id", None)),
            "trainable_index_manifest": jsonable(
                getattr(base, "trainable_index_manifest", None)
            ),
            "base_fm_score_manifest": jsonable(getattr(base, "base_fm_score_manifest", None)),
            "balance_sampling_by": jsonable(getattr(base, "balance_sampling_by", None)),
            "balance_sampling_weights": jsonable(
                getattr(base, "balance_sampling_weights", None)
            ),
            "mask_zero_arm_action_loss": jsonable(
                getattr(base, "mask_zero_arm_action_loss", None)
            ),
            "zero_arm_action_loss_dataset_sources": jsonable(
                getattr(base, "zero_arm_action_loss_dataset_sources", None)
            ),
        },
        "parameters": {
            "model": selected_fields(
                config.model,
                (
                    "action_dim",
                    "action_horizon",
                    "max_token_len",
                    "dtype",
                    "paligemma_variant",
                    "action_expert_variant",
                    "pi05",
                ),
            ),
            "initialization_checkpoint": jsonable(
                getattr(config, "pytorch_weight_path", None)
            ),
            "training_precision": jsonable(
                getattr(config, "pytorch_training_precision", None)
            ),
            "lr_schedule": jsonable(config.lr_schedule),
            "optimizer": jsonable(config.optimizer),
            "ema_decay": jsonable(getattr(config, "ema_decay", None)),
            "batch_size": jsonable(getattr(config, "batch_size", None)),
            "micro_batch_size": jsonable(
                getattr(config, "pytorch_micro_batch_size", None)
            ),
            "num_workers": jsonable(getattr(config, "num_workers", None)),
            "num_train_steps": jsonable(getattr(config, "num_train_steps", None)),
            "seed": jsonable(getattr(config, "seed", None)),
            "save_interval": jsonable(getattr(config, "save_interval", None)),
            "keep_period": jsonable(getattr(config, "keep_period", None)),
            "fsdp_devices": jsonable(getattr(config, "fsdp_devices", None)),
        },
    }


def load_registry_configs(openpi_root: Path) -> tuple[dict[str, dict[str, Any]], str | None]:
    source_dir = openpi_root / "src"
    config_file = source_dir / "openpi/training/config.py"
    if not config_file.is_file():
        return {}, f"missing {config_file}"
    sys.path.insert(0, str(source_dir))
    try:
        from openpi.training import config as training_config  # type: ignore

        configs = {
            item.name: config_snapshot(item, str(config_file))
            for item in training_config._CONFIGS  # noqa: SLF001 - registry is the canonical source
        }
        return configs, None
    except Exception as exc:  # keep result indexing useful without OpenPI deps
        return {}, f"{type(exc).__name__}: {exc}"


def unwrap_wandb(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"value"}:
        return unwrap_wandb(value["value"])
    if isinstance(value, dict):
        return {str(key): unwrap_wandb(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unwrap_wandb(item) for item in value]
    return value


def load_wandb_configs(openpi_root: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], str | None]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        return {}, f"PyYAML unavailable: {exc}"

    found: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted((openpi_root / "wandb").glob("run-*/files/config.yaml")):
        try:
            raw = yaml.safe_load(path.read_text())
            value = unwrap_wandb(raw)
            metadata = value.get("_wandb", {})
            args: list[str] = []
            for environment in metadata.get("e", {}).values():
                if isinstance(environment, dict) and environment.get("args"):
                    args = [str(item) for item in environment["args"]]
                    break
            config_name = args[0] if args else str(value.get("config_name", ""))
            exp_name = str(value.get("exp_name", ""))
            if not config_name or not exp_name:
                continue
            data = value.get("data", {})
            base = data.get("base_config", {}) if isinstance(data, dict) else {}
            snapshot = {
                "source": str(path),
                "config_name": config_name,
                "experiment_name": exp_name,
                "dataset": {
                    "repo_id": data.get("repo_id"),
                    "episodes": data.get("episodes"),
                    "asset_id": (data.get("assets") or {}).get("asset_id"),
                    "trainable_index_manifest": base.get("trainable_index_manifest"),
                    "base_fm_score_manifest": base.get("base_fm_score_manifest"),
                    "balance_sampling_by": base.get("balance_sampling_by"),
                    "balance_sampling_weights": base.get("balance_sampling_weights"),
                    "mask_zero_arm_action_loss": base.get("mask_zero_arm_action_loss"),
                    "zero_arm_action_loss_dataset_sources": base.get(
                        "zero_arm_action_loss_dataset_sources"
                    ),
                },
                "parameters": {
                    "model": value.get("model"),
                    "initialization_checkpoint": value.get("pytorch_weight_path"),
                    "training_precision": value.get("pytorch_training_precision"),
                    "lr_schedule": value.get("lr_schedule"),
                    "optimizer": value.get("optimizer"),
                    "ema_decay": value.get("ema_decay"),
                    "batch_size": value.get("batch_size"),
                    "micro_batch_size": value.get("pytorch_micro_batch_size"),
                    "num_workers": value.get("num_workers"),
                    "num_train_steps": value.get("num_train_steps"),
                    "seed": value.get("seed"),
                    "save_interval": value.get("save_interval"),
                    "keep_period": value.get("keep_period"),
                    "fsdp_devices": value.get("fsdp_devices"),
                },
            }
            found[(config_name, exp_name)] = snapshot
        except Exception:
            continue
    return found, None


def load_contract_configs(
    data_root: Path, registry: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Recover removed base configs from immutable data-ablation contracts.

    Some contracts prove that a new registered config differs from an older
    config only in its data paths.  In that case the registered successor is a
    safe donor for model/optimizer parameters, while OLD_REPO in the pinned
    program source supplies the older training dataset.
    """
    recovered: dict[str, dict[str, Any]] = {}
    for path in data_root.rglob("training_config_contract.json"):
        try:
            contract = json.loads(path.read_text())
            base_name = contract.get("base_config_name")
            donor_name = contract.get("new_config_name")
            program = contract.get("program_source", "")
            match = re.search(r'^OLD_REPO\s*=\s*["\']([^"\']+)["\']', program, re.M)
            if not base_name or not match:
                continue
            old_repo = match.group(1)
            non_data = contract.get("non_data_parameters", {})
            if donor_name in registry:
                snapshot = copy.deepcopy(registry[donor_name])
            else:
                snapshot = {
                    "dataset": {},
                    "parameters": {
                        "model": non_data.get("model"),
                        "initialization_checkpoint": non_data.get("pytorch_weight_path"),
                        "training_precision": non_data.get("precision"),
                        "lr_schedule": non_data.get("lr_schedule"),
                        "optimizer": non_data.get("optimizer"),
                        "ema_decay": non_data.get("ema_decay"),
                        "batch_size": non_data.get("batch_size"),
                        "micro_batch_size": non_data.get("pytorch_micro_batch_size"),
                        "num_workers": non_data.get("num_workers"),
                        "num_train_steps": non_data.get("num_train_steps"),
                        "seed": non_data.get("seed"),
                        "save_interval": non_data.get("save_interval"),
                        "keep_period": non_data.get("keep_period"),
                        "fsdp_devices": non_data.get("fsdp_devices"),
                    },
                }
            snapshot.update(
                {
                    "source": str(path),
                    "config_name": base_name,
                    "recovery_method": (
                        "data-ablation contract states successor parameters are identical; "
                        "OLD_REPO supplies the removed base dataset"
                    ),
                }
            )
            snapshot["dataset"]["repo_id"] = old_repo
            snapshot["dataset"]["trainable_index_manifest"] = (
                old_repo + "/meta/trainable_index_manifest.json"
            )
            snapshot["dataset"]["balance_sampling_by"] = non_data.get(
                "balance_sampling_by"
            )
            recovered[base_name] = snapshot
        except Exception:
            continue
    return recovered


def is_result_json(path: Path, output: Path) -> bool:
    return path != output and path.suffix.lower() == ".json" and bool(RESULT_TOKEN.search(path.stem))


def checkpoint_strings(value: Any, inherited_config: str | None = None) -> list[tuple[str, str | None]]:
    found: list[tuple[str, str | None]] = []
    if not isinstance(value, dict):
        return found
    local_config = value.get("config_name") if isinstance(value.get("config_name"), str) else inherited_config
    for key, item in value.items():
        if CHECKPOINT_KEY.search(str(key)) and isinstance(item, str) and "/" in item:
            found.append((item, local_config))
        elif isinstance(item, dict):
            found.extend(checkpoint_strings(item, local_config))
        elif isinstance(item, list) and len(item) <= 100:
            for child in item:
                if isinstance(child, dict):
                    found.extend(checkpoint_strings(child, local_config))
    return found


def compact_metrics(value: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, item in value.items():
        if json_scalar(item) and METRIC_KEY.search(key) and not SKIP_METRIC_KEY.search(key):
            metrics[key] = item

    for key in ("overall", "suite_summary", "tasks", "protocol"):
        item = value.get(key)
        if isinstance(item, dict):
            metrics[key] = item

    episodes = value.get("episodes")
    if isinstance(episodes, list):
        episode_dicts = [item for item in episodes if isinstance(item, dict)]
        successes = sum(item.get("success") is True for item in episode_dicts)
        metrics["episode_list_summary"] = {
            "episodes": len(episode_dicts),
            "successes": successes,
            "success_rate": successes / len(episode_dicts) if episode_dicts else None,
        }

    checkpoint_reports: dict[str, Any] = {}
    checkpoints = value.get("checkpoints")
    checkpoint_items = checkpoints.items() if isinstance(checkpoints, dict) else ()
    for checkpoint_name, checkpoint in checkpoint_items:
        if not isinstance(checkpoint, dict):
            continue
        datasets: dict[str, Any] = {}
        for dataset_name, dataset in (checkpoint.get("datasets") or {}).items():
            if not isinstance(dataset, dict):
                continue
            aggregates = {
                name: dataset[name]
                for name in ("training_weighted_mean", "trajectory_equal_mean")
                if isinstance(dataset.get(name), dict)
            }
            distribution = dataset.get("sample_distribution")
            if isinstance(distribution, dict):
                aggregates["sample_distribution_means"] = {
                    name: stats.get("mean")
                    for name, stats in distribution.items()
                    if isinstance(stats, dict) and "mean" in stats
                }
            if aggregates:
                datasets[str(dataset_name)] = aggregates
        if datasets:
            checkpoint_reports[str(checkpoint_name)] = datasets
    if checkpoint_reports:
        metrics["checkpoint_dataset_metrics"] = checkpoint_reports
    return metrics


def checkpoint_layout(path: str) -> tuple[str | None, str | None, int | None]:
    parts = Path(path).parts
    step = int(parts[-1]) if parts and parts[-1].isdigit() else None
    positions = [index for index, part in enumerate(parts) if part in {"checkpoints", "ckpts"}]
    if not positions:
        return None, None, step
    index = positions[-1] + 1
    if index < len(parts) and parts[index] == "openpi-finetunes":
        index += 1
    config_name = parts[index] if index < len(parts) else None
    exp_name = parts[index + 1] if index + 1 < len(parts) - (1 if step is not None else 0) else None
    return config_name, exp_name, step


def path_hints(path: str) -> dict[str, Any]:
    text = path.lower()
    hints: dict[str, Any] = {}
    patterns = {
        "batch_size": r"(?:^|_)bs(\d+)(?:_|$)",
        "action_horizon": r"(?:^|_)(?:horizon|ah)(\d+)(?:_|$)",
        "warmup_steps": r"(?:^|_)warmup(\d+)(?:_|$)",
        "control_frequency_hz": r"(?:^|_)(\d+)hz(?:_|$)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            hints[key] = int(match.group(1))
    lr_match = re.search(r"(?:^|_)lr(\d+(?:p\d+)?)e(\d+)(?:_|$)", text)
    if lr_match:
        hints["peak_learning_rate"] = float(lr_match.group(1).replace("p", ".")) * 10 ** -int(
            lr_match.group(2)
        )
    goal_match = re.search(r"(?:^|_)goal(\d+)(k?)(?:_|$)", text)
    if goal_match:
        hints["num_train_steps"] = int(goal_match.group(1)) * (1000 if goal_match.group(2) else 1)
    return hints


def experiment_root(data_root: Path, artifact: Path) -> Path:
    relative = artifact.relative_to(data_root)
    if relative.parts[0] == "libero_eval" and len(relative.parts) >= 3:
        return data_root.joinpath(*relative.parts[:2])
    return data_root / relative.parts[0]


def directory_size(path: Path) -> tuple[int, int, int]:
    """Return apparent byte size, regular-file count, and stat error count."""
    size_bytes = 0
    file_count = 0
    stat_errors = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                size_bytes += child.stat().st_size
                file_count += 1
        except OSError:
            stat_errors += 1
    return size_bytes, file_count, stat_errors


def human_size(size_bytes: int) -> str:
    size = float(size_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    openpi_root = args.openpi_root.resolve()

    registry, registry_error = load_registry_configs(openpi_root)
    wandb, wandb_error = load_wandb_configs(openpi_root)
    contract_configs = load_contract_configs(data_root, registry)
    registry.update(contract_configs)
    artifact_records: list[dict[str, Any]] = []
    artifact_payloads: list[tuple[Path, dict[str, Any], list[tuple[str, str | None]]]] = []
    parse_errors: list[dict[str, str]] = []
    checkpoint_configs: dict[str, set[str]] = defaultdict(set)

    for path in sorted(data_root.rglob("*.json")):
        resolved = path.resolve()
        if not is_result_json(resolved, output):
            continue
        raw = path.read_bytes()
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise TypeError("top-level JSON value is not an object")
        except Exception as exc:
            parse_errors.append({"path": str(resolved), "error": f"{type(exc).__name__}: {exc}"})
            continue
        pairs = sorted(set(checkpoint_strings(payload)))
        for checkpoint, config_name in pairs:
            if config_name:
                checkpoint_configs[checkpoint].add(config_name)
        artifact_payloads.append((resolved, payload, pairs))
        artifact_records.append(
            {
                "path": str(resolved),
                "relative_path": str(resolved.relative_to(data_root)),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "top_level_keys": sorted(payload),
                "metrics": compact_metrics(payload),
                "checkpoints": sorted({checkpoint for checkpoint, _ in pairs}),
                "config_names": sorted({config for _, config in pairs if config}),
            }
        )

    all_checkpoints = sorted(
        {checkpoint for _, _, pairs in artifact_payloads for checkpoint, _ in pairs}
    )
    checkpoint_records: dict[str, dict[str, Any]] = {}
    checkpoint_ids: dict[str, str] = {}
    for checkpoint in all_checkpoints:
        path_config, exp_name, step = checkpoint_layout(checkpoint)
        observed = sorted(checkpoint_configs.get(checkpoint, set()))
        candidates = []
        if path_config:
            candidates.append(path_config)
        candidates.extend(name for name in observed if name not in candidates)

        snapshot = None
        match_method = None
        if path_config and exp_name and (path_config, exp_name) in wandb:
            snapshot = wandb[(path_config, exp_name)]
            match_method = "checkpoint_path_to_wandb_config"
        else:
            for name in candidates:
                if name in registry:
                    snapshot = registry[name]
                    if str(snapshot.get("source", "")).endswith("training_config_contract.json"):
                        match_method = "immutable_training_contract"
                    else:
                        match_method = (
                            "checkpoint_path_to_current_registry"
                            if name == path_config
                            else "result_config_name_to_current_registry"
                        )
                    break

        identifier = "ckpt-" + hashlib.sha1(checkpoint.encode()).hexdigest()[:12]
        checkpoint_ids[checkpoint] = identifier
        checkpoint_records[identifier] = {
            "path": checkpoint,
            "path_exists_at_index_time": Path(checkpoint).exists(),
            "checkpoint_step": step,
            "path_config_name": path_config,
            "path_experiment_name": exp_name,
            "observed_result_config_names": observed,
            "training_metadata_status": "resolved" if snapshot else "unresolved",
            "training_metadata_match_method": match_method,
            "training": snapshot,
            "path_derived_parameter_hints": path_hints(checkpoint) if not snapshot else {},
        }

    by_path = {record["path"]: record for record in artifact_records}
    experiments: dict[str, dict[str, Any]] = {}
    for path, _, pairs in artifact_payloads:
        root = experiment_root(data_root, path)
        relative_root = str(root.relative_to(data_root))
        record = experiments.setdefault(
            relative_root,
            {
                "experiment_id": "exp-" + hashlib.sha1(relative_root.encode()).hexdigest()[:12],
                "name": root.name,
                "path": str(root),
                "result_artifacts": [],
                "checkpoint_refs": [],
            },
        )
        record["result_artifacts"].append(by_path[str(path)])
        record["checkpoint_refs"].extend(checkpoint_ids[checkpoint] for checkpoint, _ in pairs)

    for record in experiments.values():
        record["result_artifacts"].sort(key=lambda item: item["relative_path"])
        record["checkpoint_refs"] = sorted(set(record["checkpoint_refs"]))
        size_bytes, file_count, stat_errors = directory_size(Path(record["path"]))
        record["size_bytes"] = size_bytes
        record["size_human"] = human_size(size_bytes)
        record["file_count"] = file_count
        record["size_stat_errors"] = stat_errors

    ordered_experiments = sorted(
        experiments.values(), key=lambda item: (-item["size_bytes"], item["path"])
    )
    for rank, record in enumerate(ordered_experiments, start=1):
        record["size_rank"] = rank

    result = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_root": str(data_root),
        "scope": {
            "description": (
                "Every JSON below data_root whose filename contains result, summary, report, or metric; "
                "per-episode arrays are summarized and the immutable source path/hash is retained."
            ),
            "experiment_grouping": (
                "First directory below data_root, except libero_eval which is grouped by its second-level directory."
            ),
            "experiment_sort": (
                "Descending total apparent file size; all state/GPU/worker shards below an experiment root "
                "are counted together and are not separate experiments."
            ),
            "excluded": [
                "dataset-only metadata without a result/summary/report/metric filename",
                "the generated experiment_results.json itself",
            ],
        },
        "counts": {
            "experiments": len(experiments),
            "result_artifacts": len(artifact_records),
            "checkpoints": len(checkpoint_records),
            "resolved_checkpoint_training_metadata": sum(
                item["training_metadata_status"] == "resolved"
                for item in checkpoint_records.values()
            ),
            "unresolved_checkpoint_training_metadata": sum(
                item["training_metadata_status"] == "unresolved"
                for item in checkpoint_records.values()
            ),
            "parse_errors": len(parse_errors),
        },
        "training_metadata_sources": {
            "priority": [
                "wandb_training_snapshot",
                "immutable_training_contract",
                "current_openpi_registry",
                "checkpoint_path_hints",
            ],
            "openpi_root": str(openpi_root),
            "registry_config_count": len(registry),
            "wandb_training_snapshot_count": len(wandb),
            "recovered_contract_config_count": len(contract_configs),
            "registry_load_error": registry_error,
            "wandb_load_error": wandb_error,
        },
        "checkpoints": checkpoint_records,
        "experiments": ordered_experiments,
        "parse_errors": parse_errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), **result["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
