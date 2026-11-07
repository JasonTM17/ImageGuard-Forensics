"""Course lock: IoU is undefined on a shape mismatch and one for identical pixels."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_metrics import iou


def test_iou_shape_mismatch_and_identical_pixel() -> None:
    assert iou(np.zeros((2, 2), np.uint8), np.zeros((2, 3), np.uint8)) is None
    left = np.array([[255, 0], [0, 0]], np.uint8)
    assert iou(left, left.copy()) == 1.0
