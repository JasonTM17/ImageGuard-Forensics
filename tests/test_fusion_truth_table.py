"""Exhaustive verdict truth table for ``fusion.combine``.

Locks documented fusion rules against every pair of :class:`Verdict` values for
a primary + auxiliary pair and for two primary detectors. Spot checks in
``tests/test_fusion.py`` and course locks are not duplicated here.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Sequence
from typing import Final

import pytest

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.forensics.fusion import AUX_METHODS, combine

_ALL_VERDICTS: Final[tuple[Verdict, ...]] = (
    Verdict.SUSPICIOUS_INDICATION,
    Verdict.NO_INDICATION,
    Verdict.INCONCLUSIVE,
    Verdict.NOT_APPLICABLE,
    Verdict.ERROR,
)

_PRIMARY_AUX_PAIRS: Final[tuple[tuple[MethodName, MethodName], ...]] = (
    (MethodName.COPY_MOVE_KEYPOINT, MethodName.ELA),
    (MethodName.COPY_MOVE_BLOCK, MethodName.NOISE_INCONSISTENCY),
)

_PRIMARY_PAIR: Final[tuple[MethodName, MethodName]] = (
    MethodName.COPY_MOVE_KEYPOINT,
    MethodName.COPY_MOVE_BLOCK,
)

_AUX_CONTRACT_VERDICTS: Final[tuple[Verdict, ...]] = tuple(
    v for v in _ALL_VERDICTS if v is not Verdict.SUSPICIOUS_INDICATION
)


def _result(method: MethodName, verdict: Verdict) -> AlgorithmResult:
    """Minimal stand-in :class:`AlgorithmResult` for fusion-only tests."""
    score: float | None = None
    score_definition = ""
    if verdict is Verdict.SUSPICIOUS_INDICATION:
        score = 1.0
        score_definition = "truth-table stub score"
    return AlgorithmResult(
        method=method,
        verdict=verdict,
        score=score,
        score_definition=score_definition,
        evidence={},
    )


def _reference_combine(
    results: Sequence[AlgorithmResult],
) -> tuple[Verdict, list[str], list[str]]:
    """Mirror documented rules in ``imageguard.forensics.fusion.combine``."""
    signals: list[str] = []
    warnings: list[str] = []
    active = [r for r in results if r.verdict is not Verdict.NOT_APPLICABLE]
    usable: list[AlgorithmResult] = []
    for r in active:
        if r.verdict is Verdict.ERROR:
            warnings.append(f"{r.method.value}: method trả ERROR — loại khỏi fusion")
        else:
            usable.append(r)

    suspicious = [
        r.method.value
        for r in usable
        if r.verdict is Verdict.SUSPICIOUS_INDICATION
    ]
    if suspicious:
        signals.extend(suspicious)
        return Verdict.SUSPICIOUS_INDICATION, signals, warnings

    if usable and all(r.verdict is Verdict.NO_INDICATION for r in usable):
        return Verdict.NO_INDICATION, signals, warnings

    aux_hit = [
        r.method.value
        for r in usable
        if r.method in AUX_METHODS
        and r.verdict is Verdict.INCONCLUSIVE
        and not r.evidence.get("cue_unavailable")
    ]
    if len(aux_hit) >= 2:
        signals.append("aux_corroboration: " + ", ".join(aux_hit))
    if not usable:
        warnings.append("không method nào chạy được — toàn NOT_APPLICABLE/ERROR")
    return Verdict.INCONCLUSIVE, signals, warnings


def _pair_id(
    methods: tuple[MethodName, MethodName], verdicts: tuple[Verdict, Verdict]
) -> str:
    a, b = methods
    v0, v1 = verdicts
    return f"{a.value}+{b.value}:{v0.value},{v1.value}"


def _build_cases(
    methods: tuple[MethodName, MethodName],
) -> list[tuple[tuple[MethodName, MethodName], tuple[Verdict, Verdict]]]:
    return [
        (methods, combo)
        for combo in itertools.product(_ALL_VERDICTS, repeat=2)
    ]


_TRUTH_TABLE_CASES = (
    _build_cases(_PRIMARY_AUX_PAIRS[0])
    + _build_cases(_PRIMARY_AUX_PAIRS[1])
    + _build_cases(_PRIMARY_PAIR)
)


@pytest.mark.parametrize(
    ("methods", "verdicts"),
    _TRUTH_TABLE_CASES,
    ids=[_pair_id(m, v) for m, v in _TRUTH_TABLE_CASES],
)
def test_combine_truth_table_pair(
    methods: tuple[MethodName, MethodName],
    verdicts: tuple[Verdict, Verdict],
) -> None:
    """Every verdict pair matches the documented fusion decision table."""
    m0, m1 = methods
    v0, v1 = verdicts
    results = [_result(m0, v0), _result(m1, v1)]
    snapshot = copy.deepcopy(results)

    expected_v, expected_signals, expected_warnings = _reference_combine(results)
    overall, signals, warnings = combine(results)

    assert overall is expected_v
    assert signals == expected_signals
    assert warnings == expected_warnings
    assert isinstance(overall, Verdict)

    primaries = {MethodName.COPY_MOVE_KEYPOINT, MethodName.COPY_MOVE_BLOCK}
    usable_verdicts = [
        r.verdict
        for r in results
        if r.verdict is not Verdict.NOT_APPLICABLE and r.verdict is not Verdict.ERROR
    ]
    if any(
        r.verdict is Verdict.SUSPICIOUS_INDICATION and r.method in primaries
        for r in results
        if r.verdict is not Verdict.NOT_APPLICABLE and r.verdict is not Verdict.ERROR
    ):
        assert overall is Verdict.SUSPICIOUS_INDICATION
        assert signals

    if usable_verdicts and all(v is Verdict.NO_INDICATION for v in usable_verdicts):
        assert overall is Verdict.NO_INDICATION

    for r in results:
        if r.verdict is Verdict.ERROR:
            assert any(r.method.value in w and "ERROR" in w for w in warnings)

    assert results == snapshot


@pytest.mark.parametrize(
    "verdict",
    _AUX_CONTRACT_VERDICTS,
    ids=[v.value for v in _AUX_CONTRACT_VERDICTS],
)
def test_auxiliary_verdicts_never_suspicious_in_contract(verdict: Verdict) -> None:
    """Auxiliary methods must not emit SUSPICIOUS per method contract."""
    for aux in (MethodName.ELA, MethodName.NOISE_INCONSISTENCY):
        results = [_result(aux, verdict)]
        overall, _signals, _warnings = combine(results)
        assert overall is not Verdict.SUSPICIOUS_INDICATION


def test_empty_input_documented_inconclusive() -> None:
    overall, signals, warnings = combine([])
    assert overall is Verdict.INCONCLUSIVE
    assert signals == []
    assert warnings == ["không method nào chạy được — toàn NOT_APPLICABLE/ERROR"]


def test_all_not_applicable_documented_inconclusive() -> None:
    results = [
        _result(MethodName.COPY_MOVE_KEYPOINT, Verdict.NOT_APPLICABLE),
        _result(MethodName.ELA, Verdict.NOT_APPLICABLE),
    ]
    before = copy.deepcopy(results)
    overall, signals, warnings = combine(results)
    assert overall is Verdict.INCONCLUSIVE
    assert signals == []
    assert warnings == ["không method nào chạy được — toàn NOT_APPLICABLE/ERROR"]
    assert results == before


def test_combine_does_not_mutate_input_list() -> None:
    payload = [
        _result(MethodName.COPY_MOVE_BLOCK, Verdict.NO_INDICATION),
        _result(MethodName.ELA, Verdict.INCONCLUSIVE),
    ]
    original = copy.deepcopy(payload)
    combine(payload)
    assert payload == original
