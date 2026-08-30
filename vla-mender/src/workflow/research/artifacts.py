"""Legacy content-addressed artifacts and v2 readable repair attempts."""

from __future__ import annotations

import ast
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import (
    atomic_write_json,
    atomic_write_text,
    locked_file,
    read_json,
    safe_component,
    sha256_file,
    sha256_text,
    hardlink_unsupported,
    supports_cross_directory_hardlinks,
    utc_now,
)


@dataclass(frozen=True)
class ProgramRecord:
    id: str
    sha256: str
    path: Path
    task_key: str
    mode_ids: tuple[str, ...]
    derived_from: tuple[str, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sha256": self.sha256,
            "path": str(self.path),
            "task_key": self.task_key,
            "mode_ids": list(self.mode_ids),
            "derived_from": list(self.derived_from),
            "created_at": self.created_at,
        }


class ProgramStore:
    """Freeze full policy programs once and reuse them by content hash."""

    def __init__(self, campaign_root: str | Path):
        self.root = Path(campaign_root).resolve() / "program_store"
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        source: str,
        *,
        task_key: str,
        mode_ids: list[str] | tuple[str, ...],
        derived_from: list[str] | tuple[str, ...] = (),
    ) -> ProgramRecord:
        if not source.strip():
            raise ValueError("repair program cannot be empty")
        canonical_source = source.rstrip() + "\n"
        ast.parse(canonical_source, filename="<repair_program>")
        digest = sha256_text(canonical_source)
        program_id = f"program_{digest}"
        directory = self.root / digest
        path = directory / "repair_program.py"
        metadata_path = directory / "program.json"
        directory.mkdir(parents=True, exist_ok=True)
        with locked_file(directory / "program.lock"):
            if not path.exists():
                atomic_write_text(path, canonical_source)
            elif sha256_file(path) != digest:
                raise RuntimeError(f"content-addressed repair program changed: {path}")
            created_at = utc_now()
            if metadata_path.exists():
                metadata = read_json(metadata_path)
                task_keys = list(metadata.get("task_keys", []))
                all_modes = list(metadata.get("mode_ids", []))
                all_derived = list(metadata.get("derived_from", []))
                created_at = str(metadata.get("created_at", created_at))
            else:
                task_keys, all_modes, all_derived = [], [], []
            for value, target in (
                (task_key, task_keys),
                *[(str(value), all_modes) for value in mode_ids],
                *[(str(value), all_derived) for value in derived_from],
            ):
                if value and value not in target:
                    target.append(value)
            atomic_write_json(
                metadata_path,
                {
                    "schema_version": 1,
                    "id": program_id,
                    "sha256": digest,
                    "path": str(path),
                    "task_keys": sorted(task_keys),
                    "mode_ids": sorted(all_modes),
                    "derived_from": all_derived,
                    "created_at": created_at,
                    "updated_at": utc_now(),
                },
            )
        return ProgramRecord(
            id=program_id,
            sha256=digest,
            path=path,
            task_key=task_key,
            mode_ids=tuple(str(value) for value in mode_ids),
            derived_from=tuple(str(value) for value in derived_from),
            created_at=created_at,
        )

    def get(self, program_id: str) -> ProgramRecord:
        digest = program_id.removeprefix("program_")
        metadata_path = self.root / digest / "program.json"
        metadata = read_json(metadata_path)
        path = Path(str(metadata["path"]))
        if sha256_file(path) != str(metadata["sha256"]):
            raise RuntimeError(f"content-addressed repair program changed: {path}")
        task_keys = list(metadata.get("task_keys", []))
        return ProgramRecord(
            id=str(metadata["id"]),
            sha256=str(metadata["sha256"]),
            path=path,
            task_key=str(task_keys[0] if task_keys else ""),
            mode_ids=tuple(str(value) for value in metadata.get("mode_ids", [])),
            derived_from=tuple(str(value) for value in metadata.get("derived_from", [])),
            created_at=str(metadata["created_at"]),
        )


class TaskProgramStore:
    """Task-bound view used by an IDE subagent."""

    def __init__(self, store: ProgramStore, task_key: str):
        self._store = store
        self.task_key = task_key

    def save(
        self,
        source: str,
        mode_ids: list[str] | tuple[str, ...],
        derived_from: list[str] | tuple[str, ...] = (),
    ) -> ProgramRecord:
        return self._store.save(
            source,
            task_key=self.task_key,
            mode_ids=mode_ids,
            derived_from=derived_from,
        )

    def get(self, program_id: str) -> ProgramRecord:
        return self._store.get(program_id)


class AttemptStore:
    """Commit a job/program result atomically and reuse exact matching results."""

    def __init__(self, campaign_root: str | Path):
        self.root = Path(campaign_root).resolve() / "attempts"
        self.root.mkdir(parents=True, exist_ok=True)

    def result_path(self, task_key: str, job_id: str, program_sha256: str) -> Path:
        return (
            self.root
            / safe_component(task_key)
            / safe_component(job_id)
            / program_sha256
            / "result.json"
        )

    def committed(self, task_key: str, job_id: str, program_sha256: str) -> dict[str, Any] | None:
        path = self.result_path(task_key, job_id, program_sha256)
        return read_json(path) if path.is_file() else None

    def begin(self, task_key: str, job_id: str, program_sha256: str) -> Path:
        parent = self.result_path(task_key, job_id, program_sha256).parent
        parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="attempt_", dir=parent))

    def commit(
        self,
        temporary: str | Path,
        *,
        task_key: str,
        job_id: str,
        program_sha256: str,
        result: dict[str, Any],
        canonical: bool = True,
    ) -> Path:
        temp = Path(temporary)
        destination = self.result_path(task_key, job_id, program_sha256).parent
        result_path = destination / "result.json"
        with locked_file(destination / "attempt.lock"):
            if canonical and result_path.exists():
                shutil.rmtree(temp, ignore_errors=True)
                return result_path
            atomic_write_json(temp / "result.json", result)
            if not canonical:
                reruns = destination / "reruns"
                reruns.mkdir(parents=True, exist_ok=True)
                rerun = reruns / f"attempt_{uuid.uuid4().hex}"
                temp.replace(rerun)
                return rerun / "result.json"
            # Commit result.json last so a resumable result never points at
            # partially moved evidence.
            for item in temp.iterdir():
                if item.name == "result.json":
                    continue
                target = destination / item.name
                if target.exists():
                    continue
                item.replace(target)
            atomic_write_json(result_path, result)
            shutil.rmtree(temp, ignore_errors=True)
        return result_path


class ReadableAttemptStore:
    """V2 attempt store rooted in a readable candidate directory."""

    EVIDENCE_FILES = ("wide.mp4", "wrist.mp4", "trajectory.json")
    DEDUP_FILES = (
        *EVIDENCE_FILES,
        "terminal_observation.json",
        "stdout.log",
        "stderr.log",
        "worker_stdout.log",
        "worker_stderr.log",
    )

    def __init__(
        self,
        campaign_root: str | Path,
        *,
        evidence_dedupe: str = "auto",
    ):
        self.campaign_root = Path(campaign_root).resolve()
        if evidence_dedupe not in {"auto", "hardlink", "off"}:
            raise ValueError("evidence_dedupe must be auto, hardlink, or off")
        self.evidence_dedupe = evidence_dedupe
        self.cache_index_path = self.campaign_root / "runtime" / "cache_index.json"
        self.cache_lock_path = self.campaign_root / "runtime" / "cache_index.lock"
        self.evidence_index_path = self.campaign_root / "runtime" / "evidence_index.json"
        self.evidence_lock_path = self.campaign_root / "runtime" / "evidence_index.lock"
        self.evidence_blob_root = self.campaign_root / "runtime" / "evidence_blobs"
        self.evidence_blob_root.mkdir(parents=True, exist_ok=True)
        self.evidence_dedupe_enabled = evidence_dedupe == "hardlink" or (
            evidence_dedupe == "auto"
            and supports_cross_directory_hardlinks(
                self.campaign_root / "runtime",
                self.evidence_blob_root,
            )
        )
        with locked_file(self.cache_lock_path):
            if not self.cache_index_path.exists():
                atomic_write_json(
                    self.cache_index_path,
                    {"schema_version": 2, "entries": {}, "updated_at": utc_now()},
                )
        with locked_file(self.evidence_lock_path):
            if not self.evidence_index_path.exists():
                atomic_write_json(
                    self.evidence_index_path,
                    {
                        "schema_version": 2,
                        "next_blob": 1,
                        "entries": {},
                        "updated_at": utc_now(),
                    },
                )

    def _deduplicate_files(self, root: Path) -> None:
        """Hardlink identical durable bytes while keeping every readable path."""

        if not self.evidence_dedupe_enabled:
            return
        for name in self.DEDUP_FILES:
            path = root / name
            if not path.is_file() or path.stat().st_size == 0:
                continue
            digest = sha256_file(path)
            with locked_file(self.evidence_lock_path):
                if not self.evidence_dedupe_enabled:
                    return
                index = read_json(self.evidence_index_path)
                entries = index.setdefault("entries", {})
                blob_value = entries.get(digest)
                blob = Path(str(blob_value)) if blob_value else None
                if blob is not None and not blob.is_file():
                    entries.pop(digest, None)
                    blob = None
                if blob is None:
                    blob_id = int(index.get("next_blob", 1))
                    index["next_blob"] = blob_id + 1
                    blob = self.evidence_blob_root / f"blob_{blob_id:06d}"
                    try:
                        os.link(path, blob)
                    except OSError as exc:
                        if self.evidence_dedupe == "auto" and hardlink_unsupported(exc):
                            self.evidence_dedupe_enabled = False
                            return
                        raise
                    entries[digest] = str(blob)
                else:
                    replacement = path.with_name(f".{path.name}.deduplicated")
                    replacement.unlink(missing_ok=True)
                    try:
                        os.link(blob, replacement)
                    except OSError as exc:
                        if self.evidence_dedupe == "auto" and hardlink_unsupported(exc):
                            self.evidence_dedupe_enabled = False
                            return
                        raise
                    path.unlink()
                    replacement.replace(path)
                index["updated_at"] = utc_now()
                atomic_write_json(self.evidence_index_path, index)

    def _unlink_deduplicated(self, path: Path) -> None:
        if not path.is_file():
            return
        digest = sha256_file(path)
        with locked_file(self.evidence_lock_path):
            index = read_json(self.evidence_index_path)
            blob_value = index.get("entries", {}).get(digest)
            path.unlink(missing_ok=True)
            if blob_value:
                blob = Path(str(blob_value))
                if not blob.is_file() or blob.stat().st_nlink <= 1:
                    blob.unlink(missing_ok=True)
                    index.get("entries", {}).pop(digest, None)
                    index["updated_at"] = utc_now()
                    atomic_write_json(self.evidence_index_path, index)

    @staticmethod
    def _seed_slug(job: dict[str, Any]) -> str:
        value = str(job.get("seed_slug", ""))
        if value:
            return safe_component(value)
        return f"episode_{int(job['episode_index']):06d}_frame_{int(job['reset_frame_index']):06d}"

    def result_path(self, program: ProgramRecord, job: dict[str, Any]) -> Path:
        return program.path.parent / "evidence" / self._seed_slug(job) / "result.json"

    def committed(self, program: ProgramRecord, job: dict[str, Any]) -> dict[str, Any] | None:
        path = self.result_path(program, job)
        if path.is_file():
            return read_json(path)
        key = self._cache_key(program, job)
        with locked_file(self.cache_lock_path):
            index = read_json(self.cache_index_path)
            cached_path_value = index.get("entries", {}).get(key)
        if not cached_path_value:
            return None
        cached_path = Path(str(cached_path_value))
        if not cached_path.is_file():
            return None
        result = read_json(cached_path)
        return {**result, "cache_source_path": str(cached_path.parent)}

    @staticmethod
    def _cache_key(program: ProgramRecord, job: dict[str, Any]) -> str:
        reset_identity = job.get("reset_hash", job["job_id"])
        action_noise = job.get("action_noise")
        if action_noise is None:
            return f"{program.sha256}:{reset_identity}"
        noise_identity = sha256_text(
            json.dumps(
                action_noise,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )
        return f"{program.sha256}:{reset_identity}:action-noise-{noise_identity}"

    def cached_result_path(self, program: ProgramRecord, job: dict[str, Any]) -> Path:
        direct = self.result_path(program, job)
        if direct.is_file():
            return direct
        with locked_file(self.cache_lock_path):
            index = read_json(self.cache_index_path)
            cached = index.get("entries", {}).get(self._cache_key(program, job))
        return Path(str(cached)) if cached else direct

    def execution_lock(self, program: ProgramRecord, job: dict[str, Any]) -> Path:
        return self.result_path(program, job).parent / "execution.lock"

    def begin(self, program: ProgramRecord, job: dict[str, Any]) -> Path:
        parent = self.result_path(program, job).parent
        parent.mkdir(parents=True, exist_ok=True)
        counter_path = parent / "pending_counter.json"
        with locked_file(parent / "pending_counter.lock"):
            counter = read_json(counter_path).get("next", 1) if counter_path.is_file() else 1
            atomic_write_json(counter_path, {"next": int(counter) + 1})
            temporary = parent / f"pending_{int(counter):04d}"
            temporary.mkdir()
        return temporary

    def commit(
        self,
        temporary: str | Path,
        *,
        program: ProgramRecord,
        job: dict[str, Any],
        result: dict[str, Any],
        canonical: bool = True,
    ) -> Path:
        temp = Path(temporary)
        destination = self.result_path(program, job).parent
        result_path = destination / "result.json"
        with locked_file(destination / "attempt.lock"):
            if canonical and result_path.exists():
                shutil.rmtree(temp, ignore_errors=True)
                return result_path
            atomic_write_json(temp / "result.json", result)
            self._deduplicate_files(temp)
            if not canonical:
                reruns = destination / "reruns"
                reruns.mkdir(parents=True, exist_ok=True)
                existing = [
                    int(path.name.removeprefix("rerun_"))
                    for path in reruns.glob("rerun_[0-9][0-9][0-9][0-9]")
                    if path.name.removeprefix("rerun_").isdigit()
                ]
                rerun = reruns / f"rerun_{(max(existing, default=0) + 1):04d}"
                temp.replace(rerun)
                return rerun / "result.json"
            for item in temp.iterdir():
                if item.name == "result.json":
                    continue
                target = destination / item.name
                if target.exists():
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                    continue
                item.replace(target)
            atomic_write_json(result_path, result)
            shutil.rmtree(temp, ignore_errors=True)
            with locked_file(self.cache_lock_path):
                index = read_json(self.cache_index_path)
                index.setdefault("entries", {})[self._cache_key(program, job)] = str(result_path)
                index["updated_at"] = utc_now()
                atomic_write_json(self.cache_index_path, index)
        return result_path

    def commit_diagnostic(
        self,
        temporary: str | Path,
        *,
        program: ProgramRecord,
        job: dict[str, Any],
        result: dict[str, Any],
    ) -> Path:
        """Keep reset/infra evidence without claiming a canonical policy result."""

        temp = Path(temporary)
        destination = self.result_path(program, job).parent
        outcome = safe_component(str(result.get("outcome", "diagnostic")))
        with locked_file(destination / "attempt.lock"):
            diagnostics = destination / "diagnostics"
            diagnostics.mkdir(parents=True, exist_ok=True)
            existing = [
                int(path.name.split("_", 2)[1])
                for path in diagnostics.glob("diagnostic_[0-9][0-9][0-9][0-9]_*")
                if len(path.name.split("_", 2)) >= 3
                and path.name.split("_", 2)[1].isdigit()
            ]
            diagnostic = diagnostics / (
                f"diagnostic_{(max(existing, default=0) + 1):04d}_{outcome}"
            )
            atomic_write_json(temp / "result.json", result)
            self._deduplicate_files(temp)
            temp.replace(diagnostic)
        return diagnostic / "result.json"

    def compact_success(self, attempt_dir: str | Path) -> None:
        """Drop bulky rollout evidence while retaining result, terminal state, and logs."""

        root = Path(attempt_dir)
        for name in self.EVIDENCE_FILES:
            self._unlink_deduplicated(root / name)
