#!/usr/bin/env python3
"""Mirror successful programs from NVIDIA's public ASPIRE Task Gallery."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SOURCE_PAGE = "https://research.nvidia.com/labs/gear/aspire/"
SOURCE_GALLERY_DATA = SOURCE_PAGE + "data/gallery_libero90.json?v=65"
SOURCE_GALLERY_JS = SOURCE_PAGE + "js/gallery.js?v=92"
ROOT = Path(__file__).resolve().parent
USER_AGENT = "capx-aspire-task-gallery-sync/1"
BUILTIN_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "Exception",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "max",
    "min",
    "print",
    "range",
    "reversed",
    "round",
    "RuntimeError",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "ValueError",
    "zip",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fetch(url: str, *, attempts: int = 6) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - preserve the final network error
            error = exc
            if attempt + 1 < attempts:
                time.sleep(min(4.0, 0.4 * (2**attempt)))
    raise RuntimeError(f"failed to fetch {url} after {attempts} attempts") from error


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _absolute_url(path: str) -> str:
    return urllib.parse.urljoin(SOURCE_PAGE, path)


def _relative_program_path(source_path: str) -> Path:
    prefix = "assets/code/"
    if not source_path.startswith(prefix) or not source_path.endswith(".py"):
        raise ValueError(f"unexpected Task Gallery program path: {source_path!r}")
    relative = Path(source_path[len(prefix) :])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Task Gallery program escapes code root: {source_path!r}")
    return Path("programs") / relative


def _libero_entries(payload: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for group, tasks in payload.items():
        if not isinstance(group, str) or not isinstance(tasks, list):
            raise ValueError("unexpected LIBERO Task Gallery schema")
        for task in tasks:
            if not isinstance(task, dict):
                raise ValueError("unexpected LIBERO Task Gallery task record")
            source_path = str(dict(task.get("after", {})).get("c", ""))
            if not source_path:
                continue
            entries.append(
                {
                    "label": str(task["label"]),
                    "domain": "LIBERO",
                    "group": group,
                    "source_path": source_path,
                    "source_index": SOURCE_GALLERY_DATA,
                }
            )
    return entries


def _array_body(source: str, name: str) -> str:
    match = re.search(rf"const\s+{re.escape(name)}\s*=\s*\[(.*?)\n\s*\];", source, re.S)
    if not match:
        raise ValueError(f"could not find {name} in upstream gallery.js")
    return match.group(1)


def _supplemental_entries(source: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    specs = (
        ("ROBOSUITE_TASKS", "Robosuite", "Robosuite"),
        ("MORE_DEMOS", "", "More demos"),
        ("REAL_TASKS", "Real", "Real"),
    )
    for array_name, default_domain, group in specs:
        body = _array_body(source, array_name)
        starts = list(re.finditer(r"\{\s*label:\s*\"([^\"]+)\"", body))
        for index, start in enumerate(starts):
            record = body[start.start() : starts[index + 1].start() if index + 1 < len(starts) else None]
            after = re.search(r"after:\s*\{(.*?)\}", record, re.S)
            if not after:
                continue
            explicit = re.search(r'\bc:\s*"([^"]+\.py)"', after.group(1))
            helper = re.search(r'\bc:\s*C\("([^"]+)"\)', after.group(1))
            if explicit:
                source_path = explicit.group(1)
            elif helper:
                source_path = f"assets/code/gallery/{helper.group(1)}.py"
            else:
                continue
            domain = default_domain
            if not domain:
                domain_match = re.search(r'domain:\s*"([^"]+)"', record)
                domain = domain_match.group(1) if domain_match else "Other"
            entries.append(
                {
                    "label": start.group(1),
                    "domain": domain,
                    "group": group,
                    "source_path": source_path,
                    "source_index": SOURCE_GALLERY_JS,
                }
            )
    return entries


def _external_call_names(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    local = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in local and name not in BUILTIN_CALLS:
                names.add(name)
    return sorted(names)


def _build(
    gallery_payload: dict[str, Any], gallery_js: str
) -> tuple[dict[str, Any], dict[Path, str]]:
    records = _libero_entries(gallery_payload) + _supplemental_entries(gallery_js)
    source_paths = [record["source_path"] for record in records]
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("Task Gallery contains duplicate successful-program paths")

    def fetch_program(record: dict[str, str]) -> tuple[dict[str, str], bytes]:
        return record, _fetch(_absolute_url(record["source_path"]))

    downloaded: list[tuple[dict[str, str], bytes]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for item in pool.map(fetch_program, records):
            downloaded.append(item)

    files: dict[Path, str] = {}
    strategies: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record, payload in downloaded:
        code = payload.decode("utf-8")
        relative_path = _relative_program_path(record["source_path"])
        program_stem = Path(record["source_path"]).stem
        strategy_id = f"{_slug(record['domain'])}-{_slug(program_stem)}"
        if strategy_id in seen_ids:
            raise ValueError(f"duplicate Task Gallery strategy id: {strategy_id}")
        seen_ids.add(strategy_id)
        keywords = sorted(set(re.findall(r"[a-z0-9]+", record["label"].casefold())))
        digest = _sha256_bytes(payload)
        strategies.append(
            {
                "id": strategy_id,
                "label": record["label"],
                "domain": record["domain"],
                "group": record["group"],
                "program_role": "successful_final",
                "source_index": record["source_index"],
                "source_url": _absolute_url(record["source_path"]),
                "relative_path": relative_path.as_posix(),
                "sha256": digest,
                "bytes": len(payload),
                "keywords": keywords,
                "api_names": _external_call_names(code),
            }
        )
        files[ROOT / relative_path] = code

    strategies.sort(key=lambda item: (str(item["domain"]), str(item["label"]), str(item["id"])))
    workflow_path = ROOT / "WORKFLOW.md"
    if not workflow_path.is_file():
        raise FileNotFoundError(f"strategy workflow is missing: {workflow_path}")
    manifest = {
        "schema_version": 1,
        "source_page": SOURCE_PAGE,
        "source_gallery_data": SOURCE_GALLERY_DATA,
        "source_gallery_js": SOURCE_GALLERY_JS,
        "source_gallery_data_sha256": _sha256_bytes(
            json.dumps(gallery_payload, sort_keys=True, ensure_ascii=False).encode()
        ),
        "source_gallery_js_sha256": _sha256_bytes(gallery_js.encode()),
        "workflow_relative_path": "WORKFLOW.md",
        "workflow_sha256": _sha256_bytes(workflow_path.read_bytes()),
        "strategy_count": len(strategies),
        "domain_counts": {
            domain: sum(item["domain"] == domain for item in strategies)
            for domain in sorted({str(item["domain"]) for item in strategies})
        },
        "strategies": strategies,
    }
    files[ROOT / "manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    return manifest, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gallery-data", default=SOURCE_GALLERY_DATA)
    parser.add_argument("--source-gallery-js", default=SOURCE_GALLERY_JS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the local snapshot with the current website without writing files",
    )
    args = parser.parse_args()

    gallery_payload = json.loads(_fetch(args.source_gallery_data))
    if not isinstance(gallery_payload, dict):
        raise ValueError("NVIDIA LIBERO gallery data must be a JSON object")
    gallery_js = _fetch(args.source_gallery_js).decode("utf-8")
    manifest, files = _build(gallery_payload, gallery_js)
    mismatches = [
        path for path, text in files.items() if not path.is_file() or path.read_text() != text
    ]
    if args.check:
        if mismatches:
            print("Out-of-date files:")
            for path in mismatches:
                print(path.relative_to(ROOT))
            return 1
        print(
            f"Verified {manifest['strategy_count']} Task Gallery strategies "
            f"against {args.source_gallery_data}"
        )
        return 0

    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(
        f"Imported {manifest['strategy_count']} successful Task Gallery programs: "
        f"{manifest['domain_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
