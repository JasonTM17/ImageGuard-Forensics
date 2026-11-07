"""Course lock: boundary F1 of a two-pixel shift, an empty prediction, and an empty ground truth."""

from __future__ import annotations

import math

import numpy as np

from imageguard.evaluation.modelbench import boundary_f1


def test_boundary_f1_shift_and_empty_masks() -> None:
    gt = np.zeros((16, 16), np.uint8)
    gt[4:8, 4:8] = 255
    pred = np.zeros((16, 16), np.uint8)
    pred[6:10, 6:10] = 255
    assert boundary_f1(pred, gt, tol=2) == 0.875
    assert boundary_f1(np.zeros((16, 16), np.uint8), gt) == 0.0
    assert math.isnan(boundary_f1(pred, np.zeros((16, 16), np.uint8)))