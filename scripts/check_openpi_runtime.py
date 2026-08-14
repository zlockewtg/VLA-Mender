#!/usr/bin/env python3
"""Dependency-free OpenPI runtime and checkpoint preflight."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path


def run_python(python: Path, code: str) -> tuple[bool, str]:
    try:
        value=subprocess.run([str(python), "-c", code], text=True, capture_output=True, check=False)
    except OSError as exc:
        return False, str(exc)
    return value.returncode == 0, (value.stdout.strip() or value.stderr.strip())


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1]/"third_party/openpi")
    parser.add_argument("--environment", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--norm-stats", type=Path, default=None)
    parser.add_argument("--settings", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args=parser.parse_args()
    source=args.source.resolve()
    env=(args.environment or (Path(os.environ["VLA_MENDER_OPENPI_ENV"]) if os.environ.get("VLA_MENDER_OPENPI_ENV") else Path(sys.prefix))).resolve()
    python=env/"bin/python"
    pin=(source.parent/"openpi.commit").read_text().strip() if (source.parent/"openpi.commit").is_file() else None
    actual=subprocess.run(["git","-C",str(source),"rev-parse","HEAD"],text=True,capture_output=True,check=False).stdout.strip() or None
    imports={}
    for name in ("openpi", "torch", "jax", "flax", "orbax.checkpoint", "lerobot", "transformers"):
        ok, detail=run_python(python, f"import {name}; print(getattr({name.split('.')[0]}, '__version__', 'ok'))")
        imports[name]={"ok":ok,"detail":detail}
    source_import_code = f"import sys; sys.path.insert(0, {str(source / 'src')!r}); import openpi; print(openpi.__file__)"
    openpi_ok, openpi_module=run_python(python, source_import_code)
    openpi_source_ok=openpi_ok and str(source / "src") in openpi_module
    # Torch only needs to import successfully here. The shared evaluation
    # runtime is intentionally allowed to provide a compatible Torch build
    # (for example 2.6.0+cu124) instead of being rejected by an extra
    # VLA-Mender-side exact-version check.
    required_versions={"jax":"0.5.3", "flax":"0.10.2",
                       "orbax.checkpoint":"0.11.13", "transformers":"4.53.2"}
    distribution_names={"orbax.checkpoint":"orbax-checkpoint"}
    for name, required in required_versions.items():
        dist=distribution_names.get(name, name)
        version_ok, version_detail=run_python(
            python, f"import importlib.metadata as m; print(m.version({dist!r}))"
        )
        installed_version=str(version_detail).split("+", 1)[0]
        imports[name]["required_version"]=required
        imports[name]["installed_version"]=version_detail
        imports[name]["version_ok"]=imports[name]["ok"] and version_ok and installed_version == required
    checkpoint=args.checkpoint.resolve() if args.checkpoint else None
    if args.norm_stats:
        norm=args.norm_stats.resolve()
    elif checkpoint:
        norm_candidates=(checkpoint/"assets/physical-intelligence/libero/norm_stats.json",
                        checkpoint/"physical-intelligence/libero/norm_stats.json")
        norm=next((candidate for candidate in norm_candidates if candidate.is_file()), norm_candidates[0])
    else:
        norm=None
    report={"schema_version":1,"source":str(source),"expected_commit":pin,"actual_commit":actual,
            "environment":str(env),"python":str(python),"openpi_module":openpi_module,
            "openpi_module_matches_source":openpi_source_ok,"imports":imports,
            "checkpoint":str(checkpoint) if checkpoint else None,
            "weights_present":bool(checkpoint and (checkpoint/"model.safetensors").is_file()),
            "norm_stats":str(norm) if norm else None,"norm_stats_present":bool(norm and norm.is_file())}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True,exist_ok=True); args.json_out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    checkpoint_ok = checkpoint is None or report["weights_present"] and report["norm_stats_present"]
    imports_ok = all(item["ok"] and item.get("version_ok", True) for item in imports.values())
    return 0 if source.joinpath("pyproject.toml").is_file() and actual == pin and openpi_source_ok and imports_ok and checkpoint_ok else 1

if __name__ == "__main__": raise SystemExit(main())
