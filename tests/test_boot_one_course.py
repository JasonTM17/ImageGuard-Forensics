"""Course lock: one finite sample has a mean and no percentile bounds."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_bootstrap import bootstrap_mean


def test_one_sample_bootstrap_has_no_bounds() -> None:
    result = bootstrap_mean([2.5], np.random.default_rng(0), n_boot=4)
    assert result == {
        "n": 1,
        "mean": 2.5,
        "low": None,
        "high": None,
        "n_boot": 4,
        "method": "percentile",
    }
