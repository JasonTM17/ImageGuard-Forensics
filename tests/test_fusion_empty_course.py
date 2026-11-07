"""Course lock: an empty method list stays inconclusive with one warning."""

from __future__ import annotations

from imageguard.core.enums import Verdict
from imageguard.forensics.fusion import combine


def test_empty_fusion_is_inconclusive_with_one_warning() -> None:
    overall, signals, warnings = combine([])
    assert overall == Verdict.INCONCLUSIVE
    assert signals == []
    assert warnings == [
        "không method nào chạy được — toàn NOT_APPLICABLE/ERROR",
    ]