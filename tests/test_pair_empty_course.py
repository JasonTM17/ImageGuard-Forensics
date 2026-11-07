"""Course lock: an empty paired bootstrap reports no pairs."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_bootstrap import bootstrap_paired_diff


def test_empty_paired_bootstrap_has_no_pairs() -> None:
    result = bootstrap_paired_diff([], [1.0], np.random.default_rng(0), n_boot=4)
    assert result == {
        "n": 0,
        "mean": None,
        "low": None,
        "high": None,
        "n_boot": 4,
        "method": "percentile",
        "n_pairs": 0,
    }
