"""Course lock: two clean detectors fuse to no indication."""

from __future__ import annotations

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.forensics.fusion import combine


def test_two_clean_detectors_fuse_to_no_indication() -> None:
    results = [
        AlgorithmResult(
            method=MethodName.COPY_MOVE_KEYPOINT,
            verdict=Verdict.NO_INDICATION,
            score=None,
            score_definition="",
        ),
        AlgorithmResult(
            method=MethodName.COPY_MOVE_BLOCK,
            verdict=Verdict.NO_INDICATION,
            score=None,
            score_definition="",
        ),
    ]
    overall, signals, warnings = combine(results)
    assert overall == Verdict.NO_INDICATION
    assert signals == []
    assert warnings == []
