"""Course lock: partial overlap IoU is one half and two empty masks are undefined."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_metrics import iou


def test_iou_half_overlap_and_empty_union() -> None:
    left = np.zeros((4, 4), np.uint8)
    right = np.zeros((4, 4), np.uint8)
    left[0, 0] = 255
    right[0, 0] = 255
    right[1, 1] = 255
    assert iou(left, right) == 0.5
    assert iou(np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8)) is None
