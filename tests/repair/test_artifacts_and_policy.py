from __future__ import annotations

import errno
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from workflow.research import artifacts as artifact_module
from workflow.research import util
from workflow.research.artifacts import ProgramStore, ReadableAttemptStore
from workflow.research.experience import ExperienceLibrary
from workflow.research.policy import validate_program


def test_concurrent_experience_publication_keeps_every_item(tmp_path: Path) -> None:
    store = ProgramStore(tmp_path)
    library = ExperienceLibrary(tmp_path, store)

    def publish(index: int) -> None:
        library.publish_skill(
            f"skill_{index}",
            f"def skill_{index}():\n    return {index}\n",
            derived_from="seed",
            task_key=f"task_{index % 2}",
            mode_ids=[f"FM-{index:02d}"],
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(publish, range(12)))
    snapshot = library.snapshot()
    assert snapshot["version"] == 12
    assert len(snapshot["items"]) == 12


def test_program_boundary_keeps_math_and_rejects_private_io() -> None:
    assert validate_program("import numpy as np\nx = np.ones(3)") == []
    violations = validate_program("from pathlib import Path\nx = open('/tmp/private')")
    assert any("pathlib" in value for value in violations)
    assert any("open" in value for value in violations)
    assert any("load" in value for value in validate_program("secret = np.load('/tmp/private')"))


def test_atomic_hardlink_falls_back_to_atomic_copy(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source" / "program.py"
    destination = tmp_path / "published" / "program.py"
    source.parent.mkdir()
    source.write_text("RESULT = True\n", encoding="utf-8")

    def reject_hardlink(*_args, **_kwargs) -> None:
        raise OSError(errno.EPERM, "cross-directory hardlinks are unavailable")

    monkeypatch.setattr(util.os, "link", reject_hardlink)
    util.atomic_hardlink(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_ino != source.stat().st_ino


def test_readable_attempt_store_auto_disables_unsupported_dedupe(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        artifact_module,
        "supports_cross_directory_hardlinks",
        lambda *_args, **_kwargs: False,
    )
    store = ReadableAttemptStore(tmp_path, evidence_dedupe="auto")
    program = ProgramStore(tmp_path).save(
        "RESULT = True",
        task_key="task",
        mode_ids=["FM-01"],
    )
    job = {
        "job_id": "task:e000000-f000001",
        "episode_index": 0,
        "reset_frame_index": 1,
        "reset_hash": "reset-0",
        "seed_slug": "episode_000000_frame_000001",
    }
    pending = store.begin(program, job)
    (pending / "wide.mp4").write_bytes(b"video")
    (pending / "trajectory.json").write_text("{}\n", encoding="utf-8")

    result_path = store.commit(
        pending,
        program=program,
        job=job,
        result={"outcome": "success", "success": True},
    )

    assert store.evidence_dedupe_enabled is False
    assert result_path.is_file()
    assert (result_path.parent / "wide.mp4").read_bytes() == b"video"
    assert not list(store.evidence_blob_root.iterdir())
    assert store.committed(program, job) == {"outcome": "success", "success": True}


def test_attempt_cache_distinguishes_action_noise_seed(tmp_path: Path) -> None:
    store = ReadableAttemptStore(tmp_path, evidence_dedupe="off")
    program = ProgramStore(tmp_path).save(
        "RESULT = True",
        task_key="task",
        mode_ids=["FM-01"],
    )
    common = {
        "job_id": "task:retry",
        "reset_hash": "same-simulator-state",
    }
    first = {
        **common,
        "action_noise": {
            "type": "ornstein_uhlenbeck",
            "seed": 100,
            "standard_deviation": [0.03] * 3 + [0.02] * 3,
        },
    }
    retry = {
        **common,
        "action_noise": {
            "type": "ornstein_uhlenbeck",
            "seed": 101,
            "standard_deviation": [0.03] * 3 + [0.02] * 3,
        },
    }

    assert store._cache_key(program, first) != store._cache_key(program, retry)
    assert store._cache_key(program, first) == store._cache_key(
        program,
        {**first, "action_noise": dict(reversed(first["action_noise"].items()))},
    )
