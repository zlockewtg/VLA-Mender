"""Small orchestration layer for the generic pre-repair stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .failure_diagnosis import (
    build_agent_prompt,
    materialize_reset_bank,
    write_task_prompt,
)
from .parameters import load_settings, write_resolved_settings
from .rollout import run_rollout


def prepare_prompt(settings_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Render the complete pre-repair prompt from YAML only."""

    settings_source = Path(settings_path).expanduser().resolve()
    settings = load_settings(settings_source)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    resolved_path = root / "experiment.resolved.yaml"
    write_resolved_settings(settings, resolved_path)
    prompt_path = write_task_prompt(
        settings,
        root / "failure_diagnosis",
        run_root=root,
        settings_path=resolved_path,
    )
    manifest = {
        "schema_version": 1,
        "stage": "prompt",
        "settings_fingerprint": settings.fingerprint(),
        "settings": str(settings_source),
        "resolved_settings": str(resolved_path),
        "run_root": str(root),
        "prompt": str(prompt_path),
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "requested_settings": settings.as_dict(),
    }
    (root / "prompt_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def prepare_experiment(
    settings_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Run initial-state generation and baseline rollout only."""
    settings = load_settings(settings_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_resolved_settings(settings, output / "experiment.resolved.yaml")
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "settings_fingerprint": settings.fingerprint(),
                "stage": "rollout",
                "requested_settings": settings.as_dict(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_rollout(settings, output / "rollout")


def prepare_diagnosis(
    settings_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    settings = load_settings(settings_path)
    root = Path(output_dir).resolve()
    return build_agent_prompt(root / "rollout", root / "failure_diagnosis", settings)


def materialize(
    settings_path: str | Path, output_dir: str | Path, diagnosis_path: str | Path
) -> dict[str, Any]:
    settings = load_settings(settings_path)
    root = Path(output_dir).resolve()
    return materialize_reset_bank(
        settings, root / "rollout", diagnosis_path, root / "failure_diagnosis"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generic LIBERO pre-repair pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prompt", "rollout", "diagnose"):
        command = sub.add_parser(name)
        command.add_argument("--settings", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    reset = sub.add_parser("materialize")
    reset.add_argument("--settings", type=Path, required=True)
    reset.add_argument("--output", type=Path, required=True)
    reset.add_argument("--diagnosis", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prompt":
        prepare_prompt(args.settings, args.output)
    elif args.command == "rollout":
        prepare_experiment(args.settings, args.output)
    elif args.command == "diagnose":
        prepare_diagnosis(args.settings, args.output)
    else:
        materialize(args.settings, args.output, args.diagnosis)


if __name__ == "__main__":
    main()
