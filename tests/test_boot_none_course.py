"""Course lock: a missing sample is dropped before the bootstrap mean."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_bootstrap import bootstrap_mean


def test_missing_sample_is_dropped() -> None:
    result = bootstrap_mean([1.0, None], np.random.default_rng(0), n_boot=4)
    assert result == {
        "n": 1,
        "mean": 1.0,
        "low": None,
        "high": None,
        "n_boot": 4,
        "method": "percentile",
    }
