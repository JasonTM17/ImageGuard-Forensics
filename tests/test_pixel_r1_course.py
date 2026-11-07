"""Course lock: an empty ground-truth mask is skipped by rule R1."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.metrics import pixel_metrics


def test_empty_ground_truth_is_rule_r1() -> None:
    pred = np.zeros((4, 4), np.uint8)
    pred[0, 0] = 255
    assert pixel_metrics(pred, np.zeros((4, 4), np.uint8)) == {
        "skipped": "R1: gt empty",
    }
