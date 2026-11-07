"""Course lock: dataset split hashes two known base ids."""

from __future__ import annotations

from imageguard.evaluation.datasets import split_of


def test_split_of_known_ids_is_locked() -> None:
    assert split_of("course-d1") == "dev"
    assert split_of("img-0001") == "test"
