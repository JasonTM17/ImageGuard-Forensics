"""Course lock: an empty bootstrap reports no samples."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_bootstrap import bootstrap_mean


def test_empty_bootstrap_has_no_samples() -> None:
    result = bootstrap_mean([], np.random.default_rng(0), n_boot=4)
    assert result == {
        "n": 0,
        "mean": None,
        "low": None,
        "high": None,
        "n_boot": 4,
        "method": "percentile",
    }
