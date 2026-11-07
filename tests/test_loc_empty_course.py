"""Course lock: a missing mask on an empty ground truth is not scored."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.modelbench import localization_scores


def test_missing_mask_on_empty_ground_truth_is_unscored() -> None:
    assert localization_scores(None, np.zeros((4, 4), np.uint8)) == {
        "mask_present": False,
        "mask_status": "missing",
        "iou": None,
        "bf1": None,
    }
