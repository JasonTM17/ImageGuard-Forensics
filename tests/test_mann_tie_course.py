"""Course lock: identical samples make the Mann-Whitney variance zero."""

from __future__ import annotations

import math

import numpy as np

from imageguard.evaluation.modelbench import mann_whitney_u


def test_tied_samples_have_no_mann_whitney_pvalue() -> None:
    tied = np.ones(4, dtype=float)
    assert math.isnan(mann_whitney_u(tied, tied.copy()))
