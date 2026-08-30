"""Generate one IDE coordinator prompt from a multi-task repair YAML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from workflow.research.config import load_repair_config, resolve_repair_inputs
from workflow.research.util import atomic_write_json, atomic_write_text, sha256_file


def _task_sections(
    resolved: dict[str, Any], *, gpu_groups: list[list[int]]
) -> str:
    sections: list[str] = []
    jobs = list(resolved["jobs"])
    for task_index, task in enumerate(resolved["tasks"]):
        task_jobs = [item for item in jobs if item["task_key"] == task["task_key"]]
        modes: dict[str, list[dict[str, Any]]] = {}
        for job in task_jobs:
            modes.setdefault(str(job["failure_mode_id"]), []).append(job)
        mode_lines = []
        for mode_id, values in sorted(modes.items()):
            example = values[0]
            if int(resolved["schema_version"]) >= 2:
                mode_lines.append(
                    f"  - `{mode_id}`: {len(values)} open seeds; "
                    f"phase `{example['failure_phase']}`; "
                    f"diagnosis `{example['failure_mode']}`"
                )
            else:
                partitions: dict[str, int] = {}
                for value in values:
                    partition = str(value["initial_partition"])
                    partitions[partition] = partitions.get(partition, 0) + 1
                mode_lines.append(
                    f"  - `{mode_id}`: {len(values)} resets; initial partitions "
                    f"`{partitions}`; phase `{example['failure_phase']}`; "
                    f"diagnosis `{example['failure_mode']}`"
                )
        handoff = task.get("handoff_manifest", task.get("diagnosis", "legacy artifacts"))
        sections.extend(
            [
                f"### Task `{task['task_key']}`",
                "",
                f"- LIBERO: `{task['suite']}:{task['task_id']}` — {task['description']}",
                f"- Pre-repair run: `{task['run_root']}`",
                f"- Repair handoff: `{handoff}`",
                f"- Assigned GPU group: `{json.dumps(gpu_groups[task_index])}`",
                f"- Successful trajectory manifest: `{task['successful_episodes']}`",
                f"- Prepared repair resets: {task['job_count']}",
                *mode_lines,
                "",
            ]
        )
    return "\n".join(sections).rstrip()


def _render(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    if "{{" in rendered or "}}" in rendered:
        unresolved = sorted(
            token.split("}}", 1)[0] + "}}"
            for token in rendered.split("{{")[1:]
            if "}}" in token
        )
        raise ValueError(f"unresolved repair prompt placeholders: {unresolved}")
    return rendered


def generate(settings_path: str | Path) -> dict[str, Any]:
    config = load_repair_config(settings_path)
    resolved = resolve_repair_inputs(config)
    output = config.campaign.output_dir
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "repair_resolved.yaml"
    prompt_path = output / "prompt_generated.md"
    manifest_path = output / "prompt_manifest.json"

    resolved_document = config.to_dict()
    resolved_document["resolved_tasks"] = resolved["tasks"]
    resolved_document["jobs"] = resolved["jobs"]
    atomic_write_text(
        resolved_path,
        yaml.safe_dump(resolved_document, sort_keys=False, allow_unicode=True),
    )
    # New campaigns keep the complete frozen inventory in repair_resolved.yaml.
    # Remove the legacy generated split inventory when refreshing an old output.
    (output / "repair_jobs_resolved.json").unlink(missing_ok=True)

    template_name = "prompt_v2.md" if config.schema_version >= 2 else "prompt.md"
    template_path = Path(__file__).with_name(template_name)
    template = template_path.read_text(encoding="utf-8")
    replacements = {
        "CAMPAIGN_NAME": config.campaign.name,
        "OUTPUT_DIR": str(output.resolve()),
        "RESOLVED_SETTINGS": str(resolved_path.resolve()),
        "PROJECT_ROOT": str(config.project.root),
        "SOURCE_ROOT": str(config.project.source_root),
        "KNOWLEDGE_ROOT": str(config.project.knowledge_root),
        "PYTHON": str(config.environment.python),
        "WORKING_DIRECTORY": str(config.environment.working_directory),
        "LIBERO_ROOT": str(config.environment.libero_root),
        "PARALLEL_TASKS": str(config.campaign.parallel_tasks),
        "GPUS": json.dumps(list(config.resources.gpus)),
        "GPUS_PER_TASK": str(config.resources.gpus_per_task),
        "WORKERS_PER_GPU": str(config.resources.workers_per_gpu),
        "SERVICE_PROFILE": config.resources.services.profile,
        "EXTRA_ENV": json.dumps(config.environment.env, ensure_ascii=False, sort_keys=True),
        "SOFT_TASK_HOURS": str(config.repair.soft_task_hours),
        "SMOKE_MIN_SEEDS": str(config.repair.smoke_min_seeds),
        "SMOKE_MAX_SEEDS": str(config.repair.smoke_max_seeds),
        "NO_GAIN_REVIEW_CANDIDATES": str(
            config.repair.consecutive_no_gain_candidates
        ),
        "PER_SEED_REVIEW_ATTEMPTS": str(config.repair.per_seed_policy_attempts),
        "TASK_SECTIONS": _task_sections(
            resolved,
            gpu_groups=[
                list(config.resources.gpus)[
                    (index % config.campaign.parallel_tasks)
                    * config.resources.gpus_per_task :
                    ((index % config.campaign.parallel_tasks) + 1)
                    * config.resources.gpus_per_task
                ]
                for index in range(len(resolved["tasks"]))
            ],
        ),
    }
    atomic_write_text(prompt_path, _render(template, replacements).rstrip() + "\n")
    manifest = {
        "schema_version": config.schema_version,
        "stage": "repair_prompt",
        "source_settings": str(config.source_path),
        "resolved_settings": str(resolved_path),
        "prompt": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "task_count": len(resolved["tasks"]),
        "job_count": len(resolved["jobs"]),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = generate(args.settings)
    print(manifest["prompt"])


if __name__ == "__main__":
    main()
