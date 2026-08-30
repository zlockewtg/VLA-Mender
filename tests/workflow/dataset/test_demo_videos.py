from __future__ import annotations

from workflow.dataset.demo_videos import _evenly_spaced


def test_default_demo_selection_covers_first_middle_and_last() -> None:
    entries = [{"episode_index": index} for index in range(40)]
    assert [row["episode_index"] for row in _evenly_spaced(entries, 3)] == [0, 20, 39]


def test_demo_selection_uses_all_short_datasets() -> None:
    entries = [{"episode_index": index} for index in range(2)]
    assert _evenly_spaced(entries, 3) == entries
