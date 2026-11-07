"""Course lock: one available aux inconclusive does not corroborate."""

from __future__ import annotations

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.forensics.fusion import combine


def test_lone_aux_inconclusive_has_no_corroboration() -> None:
    result = AlgorithmResult(
        method=MethodName.ELA,
        verdict=Verdict.INCONCLUSIVE,
        score=None,
        score_definition="",
        evidence={},
    )
    overall, signals, warnings = combine([result])
    assert overall is Verdict.INCONCLUSIVE
    assert signals == []
    assert warnings == []
