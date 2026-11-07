"""Course lock: two identical rank samples have a Mann-Whitney value of one."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.modelbench import mann_whitney_u


def test_identical_ranks_have_mann_whitney_one() -> None:
    sample = np.arange(6)
    assert mann_whitney_u(sample, sample.copy()) == 1.0
