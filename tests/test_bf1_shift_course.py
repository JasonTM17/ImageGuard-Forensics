"""Course lock: a one-pixel shift at zero tolerance scores one half."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.modelbench import boundary_f1


def test_zero_tolerance_shift_scores_one_half() -> None:
    ground_truth = np.zeros((6, 6), np.uint8)
    ground_truth[1:3, 1:3] = 255
    prediction = np.zeros((6, 6), np.uint8)
    prediction[1:3, 2:4] = 255
    assert boundary_f1(prediction, ground_truth, tol=0) == 0.5
