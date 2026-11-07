"""Course lock: a seeded percentile bootstrap of four values."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_bootstrap import bootstrap_mean


def test_seeded_bootstrap_mean_is_locked() -> None:
    result = bootstrap_mean(
        [0.0, 1.0, 2.0, 3.0],
        np.random.default_rng(0),
        n_boot=32,
    )
    assert result == {
        "n": 4,
        "mean": 1.5,
        "low": 0.25,
        "high": 2.5,
        "n_boot": 32,
        "method": "percentile",
    }
