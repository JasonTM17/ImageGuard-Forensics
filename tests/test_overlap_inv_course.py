"""Course lock: an inverted interval does not overlap zero."""

from __future__ import annotations

from imageguard.evaluation.enhance_bootstrap import overlaps_zero


def test_inverted_interval_misses_zero() -> None:
    assert overlaps_zero({"low": 0.4, "high": 0.2}) is False
