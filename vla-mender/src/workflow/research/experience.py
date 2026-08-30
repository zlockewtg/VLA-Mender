"""Concurrent, experiment-local library of successful programs and skills."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import ProgramStore
from .util import (
    atomic_hardlink,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    locked_file,
    read_json,
    safe_component,
    sha256_text,
    utc_now,
)


class ExperienceLibrary:
    def __init__(
        self,
        campaign_root: str | Path,
        program_store: ProgramStore,
        *,
        schema_version: int = 1,
    ):
        self.root = Path(campaign_root).resolve() / "experience"
        self.programs_root = self.root / "programs"
        self.skills_root = self.root / "skills"
        self.policies_root = self.root / "task_policies"
        for path in (self.root, self.programs_root, self.skills_root, self.policies_root):
            path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.lock_path = self.root / "index.lock"
        self.program_store = program_store
        self.schema_version = int(schema_version)
        with locked_file(self.lock_path):
            if not self.index_path.exists():
                atomic_write_json(
                    self.index_path,
                    {
                        "schema_version": self.schema_version,
                        "version": 0,
                        "items": [],
                        "updated_at": utc_now(),
                    },
                )

    def _read(self) -> dict[str, Any]:
        return read_json(self.index_path)

    def snapshot(self) -> dict[str, Any]:
        with locked_file(self.lock_path):
            return self._read()

    def search(
        self,
        query: str = "",
        *,
        kinds: list[str] | tuple[str, ...] | None = None,
        task_key: str | None = None,
        mode_id: str | None = None,
    ) -> list[dict[str, Any]]:
        snapshot = self.snapshot()
        tokens = [token.lower() for token in query.split() if token.strip()]
        allowed = set(kinds or ())
        matches: list[dict[str, Any]] = []
        for item in snapshot.get("items", []):
            if allowed and str(item.get("kind")) not in allowed:
                continue
            item_tasks = set(str(value) for value in item.get("task_keys", []))
            if item.get("task_key"):
                item_tasks.add(str(item["task_key"]))
            if task_key is not None and task_key not in item_tasks:
                continue
            if mode_id is not None and mode_id not in item.get("mode_ids", []):
                continue
            text = " ".join(
                str(item.get(key, ""))
                for key in ("id", "kind", "name", "task_key", "mode_ids", "description")
            ).lower()
            if all(token in text for token in tokens):
                matches.append(dict(item))
        return list(reversed(matches))

    def _publish(self, item: dict[str, Any]) -> dict[str, Any]:
        with locked_file(self.lock_path):
            index = self._read()
            items = list(index.get("items", []))
            existing = next((value for value in items if value.get("id") == item["id"]), None)
            if existing is None:
                items.append(item)
                index["version"] = int(index.get("version", 0)) + 1
            else:
                for field in ("mode_ids", "derived_from", "task_keys"):
                    combined = [str(value) for value in existing.get(field, [])]
                    for value in item.get(field, []):
                        if str(value) not in combined:
                            combined.append(str(value))
                    item[field] = combined
                item["created_at"] = existing.get("created_at", item.get("created_at"))
                existing.update(item)
            index["items"] = items
            index["updated_at"] = utc_now()
            atomic_write_json(self.index_path, index)
            return {**item, "library_version": int(index["version"])}

    def publish_program(
        self,
        task_key: str,
        mode_id: str,
        program_id: str,
        results: Any,
        *,
        description: str = "",
    ) -> dict[str, Any]:
        if self.schema_version >= 2:
            raise RuntimeError("v2 campaigns publish only the single current FM experience")
        program = self.program_store.get(program_id)
        alias = (
            self.programs_root
            / (
                f"{safe_component(task_key)}__{safe_component(mode_id)}__"
                f"repair__{program.sha256[:12]}.py"
            )
        )
        if not alias.exists():
            atomic_write_text(alias, program.path.read_text(encoding="utf-8"))
        item = {
            "id": (
                f"fm_program_{program.sha256}_{safe_component(task_key)}_"
                f"{safe_component(mode_id)}"
            ),
            "kind": "fm_program",
            "name": alias.stem,
            "task_key": task_key,
            "task_keys": [task_key],
            "mode_ids": [mode_id],
            "program_id": program.id,
            "program_sha256": program.sha256,
            "path": str(alias),
            "description": description,
            "results": results,
            "derived_from": list(program.derived_from),
            "created_at": utc_now(),
        }
        return self._publish(item)

    def publish_skill(
        self,
        name: str,
        source: str,
        *,
        derived_from: str,
        task_key: str = "",
        mode_ids: list[str] | tuple[str, ...] = (),
        description: str = "",
    ) -> dict[str, Any]:
        if self.schema_version >= 2:
            raise RuntimeError("v2 skills must be bundled with the current FM experience")
        compile(source, "<experience_skill>", "exec")
        digest = sha256_text(source)
        path = self.skills_root / f"{safe_component(name)}__{digest[:12]}.py"
        if not path.exists():
            atomic_write_text(path, source.rstrip() + "\n")
        item = {
            "id": f"skill_{digest}_{safe_component(name)}",
            "kind": "skill",
            "name": name,
            "task_key": task_key,
            "task_keys": [task_key] if task_key else [],
            "mode_ids": list(mode_ids),
            "path": str(path),
            "sha256": digest,
            "description": description,
            "derived_from": [derived_from],
            "created_at": utc_now(),
        }
        return self._publish(item)

    def publish_task_policy(
        self,
        task_key: str,
        *,
        programs_by_mode: dict[str, str],
        skill_ids: list[str] | tuple[str, ...] = (),
        phase_analysis: str | None = None,
        description: str = "",
    ) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "task_key": task_key,
            "task_keys": [task_key],
            "programs_by_mode": dict(sorted(programs_by_mode.items())),
            "skill_ids": list(skill_ids),
            "phase_analysis": phase_analysis,
            "created_at": utc_now(),
        }
        digest = sha256_text(canonical_json(payload))
        path = self.policies_root / f"{safe_component(task_key)}__task_policy__{digest[:12]}.json"
        atomic_write_json(path, payload)
        item = {
            "id": f"task_policy_{digest}",
            "kind": "task_policy",
            "name": path.stem,
            "task_key": task_key,
            "mode_ids": sorted(programs_by_mode),
            "path": str(path),
            "description": description,
            "derived_from": [*programs_by_mode.values(), *skill_ids],
            "created_at": utc_now(),
        }
        return self._publish(item)

    def source(self, item_id: str) -> str:
        matches = [item for item in self.snapshot().get("items", []) if item.get("id") == item_id]
        if len(matches) != 1:
            raise KeyError(item_id)
        return Path(str(matches[0]["path"])).read_text(encoding="utf-8")

    def get(self, item_id: str) -> dict[str, Any]:
        matches = [item for item in self.snapshot().get("items", []) if item.get("id") == item_id]
        if len(matches) != 1:
            raise KeyError(item_id)
        return dict(matches[0])

    def promote_fm_experience(
        self,
        *,
        task_key: str,
        task_slug: str,
        mode_id: str,
        program_path: str | Path,
        manifest: dict[str, Any],
        skills: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Atomically replace the one searchable experience slot for a task/FM."""

        if self.schema_version < 2:
            raise RuntimeError("single-slot FM experience is available only in v2 campaigns")
        slot_id = f"fm_experience_{safe_component(task_key)}_{safe_component(mode_id)}"
        slot = self.root / safe_component(task_slug) / safe_component(mode_id)
        program_destination = slot / "program.py"
        manifest_destination = slot / "experience.json"
        skills_root = slot / "skills"
        source_program = Path(program_path).resolve()
        skill_sources = dict(skills or {})
        for name, skill_source in skill_sources.items():
            compile(skill_source, f"<experience_skill:{name}>", "exec")
        with locked_file(self.lock_path):
            slot.mkdir(parents=True, exist_ok=True)
            skills_root.mkdir(parents=True, exist_ok=True)
            atomic_hardlink(source_program, program_destination)
            retained_skill_names: set[str] = set()
            for name, skill_source in sorted(skill_sources.items()):
                filename = safe_component(name) + ".py"
                retained_skill_names.add(filename)
                atomic_write_text(skills_root / filename, skill_source.rstrip() + "\n")
            for old_skill in skills_root.glob("*.py"):
                if old_skill.name not in retained_skill_names:
                    old_skill.unlink()
            payload = {
                **manifest,
                "schema_version": 2,
                "id": slot_id,
                "kind": "fm_experience",
                "task_key": task_key,
                "task_slug": task_slug,
                "mode_ids": [mode_id],
                "path": str(program_destination),
                "skills": sorted(retained_skill_names),
                "updated_at": utc_now(),
            }
            atomic_write_json(manifest_destination, payload)
            index = self._read()
            items = [item for item in index.get("items", []) if item.get("id") != slot_id]
            items.append(payload)
            index.update(
                {
                    "schema_version": 2,
                    "version": int(index.get("version", 0)) + 1,
                    "items": items,
                    "updated_at": utc_now(),
                }
            )
            atomic_write_json(self.index_path, index)
            return {**payload, "library_version": int(index["version"])}

    def update_fm_metadata(
        self,
        *,
        task_key: str,
        task_slug: str,
        mode_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Update coverage/status of the existing FM slot without changing its program."""

        if self.schema_version < 2:
            raise RuntimeError("single-slot FM experience is available only in v2 campaigns")
        slot_id = f"fm_experience_{safe_component(task_key)}_{safe_component(mode_id)}"
        manifest_path = (
            self.root / safe_component(task_slug) / safe_component(mode_id) / "experience.json"
        )
        with locked_file(self.lock_path):
            if not manifest_path.is_file():
                raise KeyError(slot_id)
            payload = read_json(manifest_path)
            payload.update(updates)
            payload["updated_at"] = utc_now()
            atomic_write_json(manifest_path, payload)
            index = self._read()
            items = [item for item in index.get("items", []) if item.get("id") != slot_id]
            items.append(payload)
            index["items"] = items
            index["version"] = int(index.get("version", 0)) + 1
            index["updated_at"] = utc_now()
            atomic_write_json(self.index_path, index)
            return {**payload, "library_version": int(index["version"])}
