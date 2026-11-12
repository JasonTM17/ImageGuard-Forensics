"""Tests for scripts/fusion_offline_summary.py (synthetic reports only)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from fusion_offline_summary import (
    METHOD_BLOCK,
    METHOD_KEYPOINT,
    aggregate_samples,
    load_report,
    row_to_result,
    summarize,
    wilson_interval,
)

from imageguard.core.enums import Verdict
from imageguard.forensics.fusion import combine


def _wilson_ref(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def test_wilson_known_values():
    # Textbook-style check: 81/263
    low, high = wilson_interval(81, 263)
    ref_lo, ref_hi = _wilson_ref(81, 263)
    assert low == pytest.approx(ref_lo, rel=1e-9)
    assert high == pytest.approx(ref_hi, rel=1e-9)


def test_wilson_edge_cases():
    lo, hi = wilson_interval(0, 0)
    assert math.isnan(lo) and math.isnan(hi)
    lo, hi = wilson_interval(0, 10)
    assert 0.0 <= lo <= hi <= 1.0
    with pytest.raises(ValueError):
        wilson_interval(-1, 5)
    with pytest.raises(ValueError):
        wilson_interval(6, 5)


def _row(sample_id: str, label: str, method: str, status: str) -> dict:
    return {
        "sample_id": sample_id,
        "label": label,
        "method": method,
        "status": status,
        "score": 1.0,
        "score_definition": "test",
    }


def test_offline_fused_matches_combine():
    statuses = [
        Verdict.SUSPICIOUS_INDICATION,
        Verdict.NO_INDICATION,
        Verdict.INCONCLUSIVE,
        Verdict.ERROR,
        Verdict.NOT_APPLICABLE,
    ]
    rows: list[dict] = []
    for sv in statuses:
        for bv in statuses:
            sid = f"pair_{sv.value}_{bv.value}"
            rows.append(_row(sid, "authentic", METHOD_KEYPOINT, sv.value))
            rows.append(_row(sid, "authentic", METHOD_BLOCK, bv.value))
            kp = row_to_result(rows[-2])
            bl = row_to_result(rows[-1])
            expected, _, _ = combine([kp, bl])
            samples = aggregate_samples(rows)
            got = next(s for s in samples if s.sample_id == sid)
            assert got.fused == expected


def test_synthetic_report_summary(tmp_path: Path):
    report = {
        "rows": [
            _row("Au_1", "authentic", METHOD_KEYPOINT, Verdict.INCONCLUSIVE.value),
            _row("Au_1", "authentic", METHOD_BLOCK, Verdict.NO_INDICATION.value),
            _row("Tp_1", "forged", METHOD_KEYPOINT, Verdict.SUSPICIOUS_INDICATION.value),
            _row("Tp_1", "forged", METHOD_BLOCK, Verdict.NO_INDICATION.value),
        ]
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    data = load_report(path)
    summary = summarize(aggregate_samples(data["rows"]))
    assert summary["n_samples"] == 2
    fused = summary["by_view"]["fused_keypoint_block"]
    assert fused["originals"]["verdicts"][Verdict.INCONCLUSIVE.value]["count"] == 1
    assert fused["edits"]["verdicts"][Verdict.SUSPICIOUS_INDICATION.value]["count"] == 1


def test_load_report_validation(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="rows"):
        load_report(bad)
    with pytest.raises(ValueError):
        load_report(tmp_path / "missing.json")
