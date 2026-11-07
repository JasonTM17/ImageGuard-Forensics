"""Course lock: a missing prediction against a non-empty ground truth is rule R2."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.metrics import pixel_metrics


def test_missing_prediction_is_rule_r2() -> None:
    gt = np.zeros((6, 6), np.uint8)
    gt[0:2, 0:2] = 255
    assert pixel_metrics(None, gt) == {
        "skipped": None,
        "precision": None,
        "recall": 0.0,
        "f1": 0.0,
        "iou": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 4,
        "rule": "R2",
    }
