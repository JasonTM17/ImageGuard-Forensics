"""Course lock: a mask that matches its ground truth scores one."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.modelbench import localization_scores


def test_matching_mask_scores_one() -> None:
    gt = np.zeros((4, 4), np.uint8)
    gt[0:2, 0:2] = 255
    assert localization_scores(gt.copy(), gt) == {
        "mask_present": True,
        "mask_status": "valid",
        "iou": 1.0,
        "bf1": 1.0,
    }
