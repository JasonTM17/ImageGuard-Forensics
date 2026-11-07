"""Course lock: aggregating one rule-R2 image keeps precision unset and the other rates at zero."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.metrics import aggregate_pixel, pixel_metrics


def test_one_rule_r2_image_aggregates_to_zero_recall() -> None:
    gt = np.zeros((4, 4), np.uint8)
    gt[0, 0] = 255
    assert aggregate_pixel([pixel_metrics(None, gt)]) == {
        "n_images_scored": 1,
        "macro": {"precision": None, "recall": 0.0, "f1": 0.0, "iou": 0.0},
        "micro": {"precision": None, "recall": 0.0, "f1": 0.0, "iou": 0.0},
    }
