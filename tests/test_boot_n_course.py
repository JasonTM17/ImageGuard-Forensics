"""Course lock: bootstrap rejects a zero count and a boolean count."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_bootstrap import bootstrap_mean


def test_bootstrap_rejects_bad_resample_counts() -> None:
    values = [1.0]
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError) as zero:
        bootstrap_mean(values, rng, n_boot=0)
    assert str(zero.value) == "n_boot must be at least 1"
    with pytest.raises(TypeError) as flag:
        bootstrap_mean(values, rng, n_boot=True)
    assert str(flag.value) == "n_boot must be an integer"
