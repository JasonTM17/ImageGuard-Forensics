"""Course lock: only NOT_APPLICABLE inputs fuse to inconclusive with a warning."""

from __future__ import annotations

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.forensics.fusion import combine


def test_lone_not_applicable_is_inconclusive_with_warning() -> None:
    result = AlgorithmResult(
        method=MethodName.ELA,
        verdict=Verdict.NOT_APPLICABLE,
        score=None,
        score_definition="",
        evidence={},
    )
    overall, signals, warnings = combine([result])
    assert overall == Verdict.INCONCLUSIVE
    assert signals == []
    assert warnings
    assert any(
        "NOT_APPLICABLE" in item or "không method" in item for item in warnings
    )
