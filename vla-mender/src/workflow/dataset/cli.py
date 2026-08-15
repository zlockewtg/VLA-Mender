"""Command-line entry point for generic repaired-dataset construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .builder import build_dataset
from .config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Parse and validate YAML without reading datasets or writing output.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.validate_config_only:
        print(json.dumps({"valid": True, "output": str(config.output)}, indent=2))
        return 0
    print(json.dumps(build_dataset(config), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
