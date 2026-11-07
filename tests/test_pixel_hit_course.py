"""Course lock: an exact pixel overlap scores one on every rate."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.metrics import pixel_metrics


def test_exact_pixel_overlap_is_perfect() -> None:
    gt = np.zeros((4, 4), np.uint8)
    gt[0:2, 0:2] = 255
    assert pixel_metrics(gt.copy(), gt) == {
        "skipped": None,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "iou": 1.0,
        "tp": 4,
        "fp": 0,
        "fn": 0,
        "rule": None,
    }
