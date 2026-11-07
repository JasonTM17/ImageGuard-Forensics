"""Course lock: a short prediction leaves the extra ground-truth label uncounted."""

from __future__ import annotations

from imageguard.evaluation.metrics import image_metrics


def test_short_prediction_keeps_the_ground_truth_length() -> None:
    assert image_metrics([True], [True, False]) == {
        "tp": 1,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "n": 2,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "specificity": None,
        "accuracy": 0.5,
    }
