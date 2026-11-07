"""Course lock: one perfect pixel image aggregates to one on every rate."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.metrics import aggregate_pixel, pixel_metrics


def test_one_perfect_pixel_image_aggregates_to_one() -> None:
    gt = np.zeros((4, 4), np.uint8)
    gt[0:2, 0:2] = 255
    assert aggregate_pixel([pixel_metrics(gt.copy(), gt)]) == {
        "n_images_scored": 1,
        "macro": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0},
        "micro": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0},
    }
