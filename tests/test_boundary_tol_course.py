"""Course lock: a one-pixel boundary tolerance on a shifted 2x2 block."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.metrics import boundary_tolerance


def test_boundary_tolerance_one_pixel_is_locked() -> None:
    gt = np.zeros((8, 8), np.uint8)
    gt[3:5, 3:5] = 255
    pred = np.zeros((8, 8), np.uint8)
    pred[4:6, 4:6] = 255
    assert boundary_tolerance(pred, gt, k=1) == {
        "skipped": None,
        "precision": 1.0,
        "recall": 0.25,
        "f1": 0.4,
        "iou": 0.25,
        "tp": 4,
        "fp": 0,
        "fn": 12,
        "rule": None,
    }
