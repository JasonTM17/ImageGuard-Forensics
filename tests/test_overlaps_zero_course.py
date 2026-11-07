"""Course lock: zero overlap needs both bounds and includes a zero endpoint."""

from __future__ import annotations

from imageguard.evaluation.enhance_bootstrap import overlaps_zero


def test_overlaps_zero_bounds_are_locked() -> None:
    assert overlaps_zero({"low": -0.2, "high": 0.1}) is True
    assert overlaps_zero({"low": 0.0, "high": 1.0}) is True
    assert overlaps_zero({"low": 0.1, "high": None}) is None
