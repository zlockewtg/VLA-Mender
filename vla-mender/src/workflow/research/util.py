"""Small filesystem primitives shared by the repair research runtime."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


LOOPBACK_NO_PROXY_HOSTS = ("127.0.0.1", "localhost", "::1")
HARDLINK_UNSUPPORTED_ERRNOS = frozenset(
    {
        errno.EPERM,
        errno.EXDEV,
        errno.EOPNOTSUPP,
        errno.ENOSYS,
    }
)


def add_loopback_no_proxy(environment: dict[str, str]) -> None:
    """Ensure subprocess HTTP clients connect to local tool services directly."""
    for key in ("NO_PROXY", "no_proxy"):
        existing = [
            value.strip()
            for value in environment.get(key, "").split(",")
            if value.strip()
        ]
        environment[key] = ",".join(
            dict.fromkeys([*LOOPBACK_NO_PROXY_HOSTS, *existing])
        )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_text(path: str | Path, value: str, *, mode: int | None = None) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name + "_", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(path: str | Path, value: Any, *, mode: int | None = None) -> Path:
    return atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        mode=mode,
    )


def hardlink_unsupported(error: OSError) -> bool:
    """Return whether a hardlink failed because the filesystem cannot provide it."""

    return error.errno in HARDLINK_UNSUPPORTED_ERRNOS


def supports_cross_directory_hardlinks(
    source_directory: str | Path,
    destination_directory: str | Path,
) -> bool:
    """Probe hardlink support between two directories without retaining artifacts."""

    source_root = Path(source_directory)
    destination_root = Path(destination_directory)
    source_root.mkdir(parents=True, exist_ok=True)
    destination_root.mkdir(parents=True, exist_ok=True)
    source_fd, source_name = tempfile.mkstemp(prefix=".hardlink_probe_", dir=source_root)
    os.close(source_fd)
    source = Path(source_name)
    destination = destination_root / source.name
    try:
        try:
            os.link(source, destination)
        except OSError as exc:
            if hardlink_unsupported(exc):
                return False
            raise
        return True
    finally:
        destination.unlink(missing_ok=True)
        source.unlink(missing_ok=True)


def atomic_hardlink(source: str | Path, destination: str | Path) -> Path:
    """Atomically expose immutable bytes, copying when hardlinks are unavailable."""

    source_path = Path(source).resolve()
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination_path.name + "_",
        suffix=".link",
        dir=destination_path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        try:
            os.link(source_path, temporary)
        except OSError as exc:
            if not hardlink_unsupported(exc):
                raise
            with source_path.open("rb") as source_stream, temporary.open("xb") as output:
                shutil.copyfileobj(source_stream, output)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, source_path.stat().st_mode & 0o777)
        temporary.replace(destination_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination_path


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return normalized or "item"


def readable_slug(value: str, *, fallback: str = "item", max_length: int = 96) -> str:
    """Return a lower-case, human-readable filesystem component."""

    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    normalized = normalized[:max_length].rstrip("_")
    return normalized or fallback


def append_jsonl(path: str | Path, value: Any) -> Path:
    """Append one durable JSON object while serializing concurrent writers."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with locked_file(destination.with_suffix(destination.suffix + ".lock")):
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return destination


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def resolve_beneath(root: Path, value: str | Path) -> Path:
    path = resolve_path(root, value)
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes trusted root {root}: {path}") from exc
    return path


@contextlib.contextmanager
def locked_file(path: str | Path, *, blocking: bool = True) -> Iterator[Any]:
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+", encoding="utf-8")
    flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(stream.fileno(), flags)
        yield stream
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()
