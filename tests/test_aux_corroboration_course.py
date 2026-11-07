"""Course lock: two available aux cues corroborate and unavailable cues do not."""

from __future__ import annotations

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.forensics.fusion import combine


def _result(method: MethodName, evidence: dict | None = None) -> AlgorithmResult:
    return AlgorithmResult(
        method=method,
        verdict=Verdict.INCONCLUSIVE,
        evidence=evidence or {},
    )


def test_two_available_aux_cues_corroborate() -> None:
    overall, signals, warnings = combine(
        [
            _result(MethodName.ELA),
            _result(MethodName.FREQUENCY_ANALYSIS),
        ]
    )
    assert overall == Verdict.INCONCLUSIVE
    assert signals == ["aux_corroboration: ela, frequency_analysis"]
    assert warnings == []


def test_unavailable_aux_cues_do_not_corroborate() -> None:
    overall, signals, warnings = combine(
        [
            _result(MethodName.ELA, {"cue_unavailable": True}),
            _result(MethodName.FREQUENCY_ANALYSIS, {"cue_unavailable": True}),
        ]
    )
    assert overall == Verdict.INCONCLUSIVE
    assert signals == []
    assert warnings == []
