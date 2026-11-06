"""Fusion truth table — mọi nhánh của bảng luật có test."""

from __future__ import annotations

import pytest

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.forensics.fusion import combine


def _r(method: MethodName, verdict: Verdict, score: float | None = None,
       evidence: dict | None = None):
    return AlgorithmResult(
        method=method, verdict=verdict, score=score,
        score_definition="test def" if score is not None else "",
        evidence=evidence or {},
    )


KP, BL, EL, EX = (MethodName.COPY_MOVE_KEYPOINT, MethodName.COPY_MOVE_BLOCK,
                  MethodName.ELA, MethodName.EXIF_AUDIT)


@pytest.mark.parametrize("verdicts,expected", [
    ([Verdict.SUSPICIOUS_INDICATION, Verdict.NO_INDICATION], Verdict.SUSPICIOUS_INDICATION),
    ([Verdict.NO_INDICATION, Verdict.NO_INDICATION], Verdict.NO_INDICATION),
    ([Verdict.NO_INDICATION, Verdict.INCONCLUSIVE], Verdict.INCONCLUSIVE),
    ([Verdict.INCONCLUSIVE, Verdict.INCONCLUSIVE], Verdict.INCONCLUSIVE),
    ([Verdict.NOT_APPLICABLE, Verdict.NO_INDICATION], Verdict.NO_INDICATION),
    ([Verdict.NOT_APPLICABLE, Verdict.NOT_APPLICABLE], Verdict.INCONCLUSIVE),
    ([Verdict.ERROR, Verdict.NO_INDICATION], Verdict.NO_INDICATION),
    ([Verdict.ERROR], Verdict.INCONCLUSIVE),
    ([Verdict.SUSPICIOUS_INDICATION, Verdict.ERROR], Verdict.SUSPICIOUS_INDICATION),
])
def test_combine_table(verdicts, expected) -> None:
    methods = [KP, BL, EL, EX]
    results = [_r(methods[i % 4], v, score=1.0 if v == Verdict.SUSPICIOUS_INDICATION else None)
               for i, v in enumerate(verdicts)]
    v, signals, _ = combine(results)
    assert v == expected
    if expected == Verdict.SUSPICIOUS_INDICATION:
        assert signals  # phải liệt kê method nào kết luận


def test_empty_results_inconclusive() -> None:
    v, _, w = combine([])
    assert v == Verdict.INCONCLUSIVE and w


def test_aux_can_never_convict() -> None:
    """ELA/EXIF chỉ trả tối đa INCONCLUSIVE → một mình không thể SUSPICIOUS."""
    results = [_r(EL, Verdict.INCONCLUSIVE), _r(EX, Verdict.INCONCLUSIVE)]
    v, _, _ = combine(results)
    assert v == Verdict.INCONCLUSIVE


def test_aux_corroboration_skips_unavailable_cue() -> None:
    results = [
        _r(MethodName.SPLICING_CUES, Verdict.INCONCLUSIVE,
           evidence={"cue_unavailable": True}),
        _r(MethodName.NOISE_INCONSISTENCY, Verdict.INCONCLUSIVE),
    ]
    _, signals, _ = combine(results)
    assert signals == []


def test_aux_corroboration_counts_available_cues() -> None:
    results = [
        _r(MethodName.SPLICING_CUES, Verdict.INCONCLUSIVE),
        _r(MethodName.NOISE_INCONSISTENCY, Verdict.INCONCLUSIVE),
    ]
    _, signals, _ = combine(results)
    assert signals == [
        "aux_corroboration: splicing_cues, noise_inconsistency",
    ]


def test_score_without_definition_fails() -> None:
    with pytest.raises(ValueError):
        _r(KP, Verdict.SUSPICIOUS_INDICATION)  # score None nhưng gọi có score
        AlgorithmResult(method=KP, verdict=Verdict.SUSPICIOUS_INDICATION,
                        score=5.0, score_definition="")
