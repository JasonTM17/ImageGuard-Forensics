"""Course lock: Mann-Whitney is undefined below four samples and fixed when separated."""

from __future__ import annotations

import math

import numpy as np

from imageguard.evaluation.modelbench import mann_whitney_u


def test_mann_whitney_small_is_nan_and_separated_is_locked() -> None:
    small = mann_whitney_u(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0, 7.0]))
    assert math.isnan(small)
    separated = mann_whitney_u(np.arange(8.0), np.arange(8.0) + 10)
    assert separated == 0.0007775304469403844
