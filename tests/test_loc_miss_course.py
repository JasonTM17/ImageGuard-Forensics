"""Course lock: a missing mask scores as a miss, and a shape mismatch is not scored."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.modelbench import localization_scores


def test_missing_mask_and_shape_mismatch() -> None:
    gt = np.zeros((8, 8), np.uint8)
    gt[2:6, 2:6] = 255
    missing = localization_scores(None, gt)
    assert missing == {
        "mask_present": False,
        "mask_status": "missing",
        "iou": 0.0,
        "bf1": 0.0,
    }
    mismatch = localization_scores(np.zeros((8, 7), np.uint8), gt)
    assert mismatch == {
        "mask_present": True,
        "mask_status": "shape_mismatch",
        "iou": None,
        "bf1": None,
    }
