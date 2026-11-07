"""Course lock: a lone ERROR result fuses to inconclusive with a warning."""

from __future__ import annotations

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.forensics.fusion import combine


def test_lone_error_result_is_inconclusive_with_warning() -> None:
    result = AlgorithmResult(
        method=MethodName.COPY_MOVE_KEYPOINT,
        verdict=Verdict.ERROR,
        score=None,
        score_definition="",
        evidence={},
    )
    overall, signals, warnings = combine([result])
    assert overall == Verdict.INCONCLUSIVE
    assert signals == []
    assert warnings
    assert any("ERROR" in item for item in warnings)
