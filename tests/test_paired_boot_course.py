"""Course lock: a constant paired difference bootstraps to that difference."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_bootstrap import bootstrap_paired_diff


def test_constant_paired_difference() -> None:
    result = bootstrap_paired_diff(
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0],
        np.random.default_rng(0),
        n_boot=32,
    )
    assert result == {
        "n": 4,
        "mean": 1.0,
        "low": 1.0,
        "high": 1.0,
        "n_boot": 32,
        "method": "percentile",
        "n_pairs": 4,
    }
