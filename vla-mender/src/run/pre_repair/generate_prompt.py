"""Render a task-specific pre-repair prompt from a YAML contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow.failure_diagnosis import write_task_prompt
from workflow.parameters import load_settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Rendered prompt path (default: prompt.generated.md beside the YAML)",
    )
    args = parser.parse_args(argv)
    settings_source = args.settings.expanduser().resolve()
    settings = load_settings(settings_source)
    destination = (
        args.output.expanduser().resolve()
        if args.output is not None
        else settings_source.with_name("prompt.generated.md")
    )
    run_root = settings_source.parent
    prompt_path = write_task_prompt(
        settings,
        destination.parent,
        filename=destination.name,
        run_root=run_root,
        settings_path=settings_source,
    )
    print(prompt_path)


if __name__ == "__main__":
    main()
