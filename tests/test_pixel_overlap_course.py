"""Course lock: a half-shifted mask has IoU one quarter."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.metrics import pixel_metrics


def test_half_shifted_mask_locks_pixel_counts() -> None:
    gt = np.zeros((10, 10), np.uint8)
    gt[0:5, :] = 255
    pred = np.zeros((10, 10), np.uint8)
    pred[3:8, :] = 255
    assert pixel_metrics(pred, gt) == {
        "skipped": None,
        "precision": 0.4,
        "recall": 0.4,
        "f1": 0.4000000000000001,
        "iou": 0.25,
        "tp": 20,
        "fp": 30,
        "fn": 30,
        "rule": None,
    }
