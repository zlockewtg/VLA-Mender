"""YAML-driven orchestration for repaired-dataset preparation and construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .builder import build_dataset
from .config import load_config
from .research_manifest import materialize_research_manifest


class DatasetRunConfigError(ValueError):
    pass


def _resolve(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise DatasetRunConfigError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _unknown(section: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    extra = set(value) - allowed
    if extra:
        raise DatasetRunConfigError(f"unknown {section} keys: {sorted(extra)}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to change existing run artifact: {path}")
        return
    path.write_text(content, encoding="utf-8")


def load_run_settings(path: str | Path) -> dict[str, Any]:
    settings_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise DatasetRunConfigError("dataset run settings must be a YAML mapping")
    data = dict(raw)
    _unknown("top-level", data, {"schema_version", "run", "source", "build", "training"})
    if int(data.get("schema_version", 0)) != 1:
        raise DatasetRunConfigError("dataset run schema_version must be 1")
    for key in ("run", "source", "build"):
        if not isinstance(data.get(key), Mapping):
            raise DatasetRunConfigError(f"{key} must be a mapping")
    run = dict(data["run"])
    source = dict(data["source"])
    build = dict(data["build"])
    _unknown("run", run, {"output_dir"})
    _unknown(
        "source",
        source,
        {
            "adapter",
            "pre_repair_run",
            "repair_run",
            "selection_manifest",
            "task_index",
        },
    )
    if source.get("adapter") != "research_quality_selection":
        raise DatasetRunConfigError(
            "source.adapter must be research_quality_selection for standalone repair outputs"
        )
    required_source = {"pre_repair_run", "repair_run", "selection_manifest"}
    missing = required_source - set(source)
    if missing:
        raise DatasetRunConfigError(f"missing source keys: {sorted(missing)}")
    forbidden = {"episodes_manifest", "task_catalog"} & set(build)
    if forbidden:
        raise DatasetRunConfigError(
            f"build keys are generated from source and must be omitted: {sorted(forbidden)}"
        )
    required_build = {"output", "reference_dataset", "cameras", "action"}
    missing = required_build - set(build)
    if missing:
        raise DatasetRunConfigError(f"missing build keys: {sorted(missing)}")
    base = settings_path.parent
    return {
        "settings_path": settings_path,
        "run_root": _resolve(base, run.get("output_dir"), "run.output_dir"),
        "source": {
            **source,
            "pre_repair_run": _resolve(base, source["pre_repair_run"], "source.pre_repair_run"),
            "repair_run": _resolve(base, source["repair_run"], "source.repair_run"),
            "selection_manifest": _resolve(
                base, source["selection_manifest"], "source.selection_manifest"
            ),
            "task_index": int(source.get("task_index", 0)),
        },
        "build": build,
        "training": dict(data.get("training") or {}),
    }


def prepare_dataset_run(settings_path: str | Path) -> dict[str, Any]:
    settings = load_run_settings(settings_path)
    root: Path = settings["run_root"]
    root.mkdir(parents=True, exist_ok=True)
    source = settings["source"]
    episodes_path = root / "episodes_manifest.json"
    if episodes_path.exists():
        episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    else:
        episodes = materialize_research_manifest(
            pre_repair_run=source["pre_repair_run"],
            repair_run=source["repair_run"],
            selection_manifest=source["selection_manifest"],
            output=episodes_path,
            task_index=source["task_index"],
        )
    if not isinstance(episodes, dict) or not episodes.get("episodes"):
        raise DatasetRunConfigError(f"generated episode manifest is invalid: {episodes_path}")
    expected = {
        "pre_repair_run": str(source["pre_repair_run"]),
        "repair_run": str(source["repair_run"]),
        "selection_manifest": str(source["selection_manifest"]),
        "task_index": int(source["task_index"]),
    }
    mismatches = {
        key: (episodes.get(key), value)
        for key, value in expected.items()
        if episodes.get(key) != value
    }
    if mismatches:
        raise DatasetRunConfigError(f"existing episode manifest differs from settings: {mismatches}")

    task_catalog = root / "tasks.jsonl"
    task_row = {
        "task_index": int(source["task_index"]),
        "task": str(episodes["task"]),
    }
    _write_immutable(task_catalog, json.dumps(task_row, ensure_ascii=False) + "\n")

    build = dict(settings["build"])
    base = settings["settings_path"].parent
    for key in ("output", "reference_dataset"):
        build[key] = str(_resolve(base, build[key], f"build.{key}"))
    if build.get("provenance_files"):
        build["provenance_files"] = [
            str(_resolve(base, item, "build.provenance_files"))
            for item in build["provenance_files"]
        ]
    build["episodes_manifest"] = str(episodes_path)
    build["task_catalog"] = str(task_catalog)
    resolved_config = root / "dataset.resolved.yaml"
    _write_immutable(
        resolved_config,
        yaml.safe_dump(build, sort_keys=False, allow_unicode=True),
    )
    config = load_config(resolved_config)

    training_path: Path | None = None
    if settings["training"]:
        training = dict(settings["training"])
        training["dataset"] = str(config.output)
        training["trainable_index_manifest"] = str(
            config.output / "meta/trainable_index_manifest.json"
        )
        training_path = root / "training.resolved.yaml"
        _write_immutable(
            training_path,
            yaml.safe_dump(training, sort_keys=False, allow_unicode=True),
        )

    manifest = {
        "schema_version": 1,
        "stage": "dataset_prepared",
        "settings": str(settings["settings_path"]),
        "settings_sha256": _sha256(settings["settings_path"]),
        "run_root": str(root),
        "episodes_manifest": str(episodes_path),
        "episodes_manifest_sha256": _sha256(episodes_path),
        "episode_count": len(episodes["episodes"]),
        "dataset_config": str(resolved_config),
        "dataset_config_sha256": _sha256(resolved_config),
        "dataset_output": str(config.output),
        "training_config": str(training_path) if training_path else None,
    }
    manifest_path = root / "run_manifest.json"
    _write_immutable(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return manifest


def build_dataset_run(settings_path: str | Path) -> dict[str, Any]:
    manifest = prepare_dataset_run(settings_path)
    config = load_config(manifest["dataset_config"])
    report = build_dataset(config)
    root = Path(manifest["run_root"])
    report_path = root / "build_report.json"
    _write_immutable(
        report_path,
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    return report
