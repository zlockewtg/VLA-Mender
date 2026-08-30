"""Minimal capability boundary for observation-conditioned repair programs."""

from __future__ import annotations

import ast
import builtins
import math
from types import MappingProxyType
from typing import Any

import numpy as np


ALLOWED_IMPORT_ROOTS = {"collections", "math", "numpy", "statistics"}
DENIED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
DENIED_ATTRIBUTES = {
    "DataSource",
    "ctypeslib",
    "dump",
    "dumps",
    "fromfile",
    "load",
    "loads",
    "memmap",
    "popen",
    "save",
    "savetxt",
    "savez",
    "savez_compressed",
    "system",
    "tofile",
}


def validate_program(source: str) -> list[str]:
    """Reject only filesystem/process/network/reflection escape surfaces."""

    violations: list[str] = []
    try:
        tree = ast.parse(source, filename="<repair_program>")
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    violations.append(
                        f"import is outside the repair capability boundary: {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                violations.append(
                    f"import is outside the repair capability boundary: {node.module}"
                )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in DENIED_CALLS
        ):
            violations.append(f"call is unavailable in repair programs: {node.func.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            violations.append(f"dunder reflection is unavailable: {node.attr}")
        elif isinstance(node, ast.Attribute) and node.attr in DENIED_ATTRIBUTES:
            violations.append(f"attribute is outside the repair capability boundary: {node.attr}")
    return sorted(set(violations))


def _controlled_import(
    name: str,
    globals_value: dict[str, Any] | None = None,
    locals_value: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    del globals_value, locals_value
    if level != 0 or name.split(".", 1)[0] not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"repair program import denied: {name}")
    return builtins.__import__(name, {}, {}, fromlist, level)


def policy_builtins() -> MappingProxyType[str, Any]:
    values = dict(vars(builtins))
    for name in DENIED_CALLS:
        values.pop(name, None)
    values["__import__"] = _controlled_import
    return MappingProxyType(values)


def execute_program(
    source: str,
    *,
    functions: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    violations = validate_program(source)
    if violations:
        raise ValueError(f"repair program rejected: {violations}")
    namespace: dict[str, Any] = {
        "__builtins__": policy_builtins(),
        "__name__": "__repair_program__",
        "np": np,
        "math": math,
        "INPUTS": observation,
        **functions,
    }
    exec(compile(source, "<repair_program>", "exec"), namespace, namespace)
    return namespace
