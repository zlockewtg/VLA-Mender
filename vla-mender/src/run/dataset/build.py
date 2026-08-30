"""Prepare or build a repaired LeRobot dataset from one YAML settings file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from workflow.dataset.run import build_dataset_run, load_run_settings, prepare_dataset_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Validate the high-level YAML without writing run artifacts.",
    )
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="Resolve source selection and builder/training configs without building parquet.",
    )
    args = parser.parse_args(argv)
    if args.validate_config_only:
        settings = load_run_settings(args.settings)
        result = {
            "valid": True,
            "run_root": str(settings["run_root"]),
            "dataset_output": str(settings["build"]["output"]),
        }
    elif args.prepare_only:
        result = prepare_dataset_run(args.settings)
    else:
        result = build_dataset_run(args.settings)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
