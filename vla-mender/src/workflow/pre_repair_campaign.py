"""Render an ordered, fail-fast bundle of independent pre-repair tasks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .failure_diagnosis import write_task_prompt
from .parameters import ExperimentPlan, load_settings, write_resolved_settings


def _campaign_fingerprint(plan: ExperimentPlan) -> str:
    payload = [
        {"order": index, "key": task.key, "settings_fingerprint": task.settings.fingerprint()}
        for index, task in enumerate(plan.tasks)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _render_campaign_prompt(manifest: dict[str, Any]) -> str:
    template_path = (
        Path(__file__).resolve().parents[1]
        / "run"
        / "pre_repair"
        / "campaign_prompt.md"
    )
    template = template_path.read_text(encoding="utf-8")
    rows = []
    for task in manifest["tasks"]:
        rows.append(
            f"{task['order'] + 1}. `{task['key']}` — "
            f"`{task['suite']}:{task['task_id']}`\n"
            f"   - Task prompt: `{task['prompt']}`\n"
            f"   - Resolved settings: `{task['resolved_settings']}`\n"
            f"   - Run root: `{task['run_root']}`\n"
            f"   - Settings fingerprint: `{task['settings_fingerprint']}`"
        )
    values = {
        "{{CAMPAIGN_ROOT}}": manifest["campaign_root"],
        "{{CAMPAIGN_MANIFEST}}": manifest["manifest_path"],
        "{{CAMPAIGN_FINGERPRINT}}": manifest["campaign_fingerprint"],
        "{{TASK_COUNT}}": str(manifest["task_count"]),
        "{{TASK_LIST}}": "\n\n".join(rows),
    }
    for placeholder, value in values.items():
        template = template.replace(placeholder, value)
    unresolved = sorted(set(re.findall(r"\{\{[^{}]+\}\}", template)))
    if unresolved:
        raise ValueError(f"unresolved campaign prompt placeholders: {unresolved}")
    return template


def write_campaign_bundle(
    plan: ExperimentPlan,
    *,
    settings_path: str | Path,
    campaign_root: str | Path,
    prompt_path: str | Path,
) -> dict[str, Any]:
    """Write isolated task contracts plus one sequential coordinator prompt."""

    if not plan.is_campaign:
        raise ValueError("write_campaign_bundle requires a top-level tasks list")
    source = Path(settings_path).expanduser().resolve()
    root = Path(campaign_root).expanduser().resolve()
    destination = Path(prompt_path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "campaign_manifest.json"
    campaign_fingerprint = _campaign_fingerprint(plan)
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(existing_manifest, dict)
            or existing_manifest.get("artifact_type") != "vla_mender.pre_repair_campaign"
            or existing_manifest.get("campaign_fingerprint") != campaign_fingerprint
        ):
            raise FileExistsError(
                "campaign root already contains a different campaign contract; "
                f"select a new output directory: {root}"
            )
    tasks: list[dict[str, Any]] = []
    for index, task in enumerate(plan.tasks):
        task_root = root / "tasks" / f"{index:03d}_{task.key}"
        resolved_path = task_root / "experiment.resolved.yaml"
        prompt = task_root / "failure_diagnosis" / "prompt.md"
        if task_root.exists() and not resolved_path.is_file() and any(task_root.iterdir()):
            raise FileExistsError(
                "task run root already contains artifacts without a resolved contract: "
                f"{task_root}"
            )
        if resolved_path.is_file():
            existing_settings = load_settings(resolved_path)
            if existing_settings.fingerprint() != task.settings.fingerprint():
                raise FileExistsError(
                    "task run root already contains a different resolved contract; "
                    f"select a new campaign output: {task_root}"
                )
        else:
            write_resolved_settings(task.settings, resolved_path)
        write_task_prompt(
            task.settings,
            prompt.parent,
            filename=prompt.name,
            run_root=task_root,
            settings_path=resolved_path,
        )
        tasks.append(
            {
                "order": index,
                "key": task.key,
                "suite": task.settings.task.suite,
                "task_id": task.settings.task.task_id,
                "settings_fingerprint": task.settings.fingerprint(),
                "run_root": str(task_root),
                "resolved_settings": str(resolved_path),
                "prompt": str(prompt),
                "completion_manifest": str(task_root / "repair_handoff" / "manifest.json"),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "vla_mender.pre_repair_campaign",
        "execution": "sequential_fail_fast",
        "source_settings": str(source),
        "campaign_root": str(root),
        "manifest_path": str(manifest_path),
        "campaign_prompt": str(destination),
        "campaign_fingerprint": campaign_fingerprint,
        "task_count": len(tasks),
        "tasks": tasks,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_render_campaign_prompt(manifest), encoding="utf-8")
    return manifest
