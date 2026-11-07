"""Course lock: the benchmark IoU of two empty masks is zero."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.modelbench import iou


def test_benchmark_iou_of_empty_masks_is_zero() -> None:
    blank = np.zeros((4, 4), np.uint8)
    assert iou(blank, blank.copy()) == 0.0
