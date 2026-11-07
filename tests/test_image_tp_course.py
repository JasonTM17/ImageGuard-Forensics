"""Course lock: one true-positive image scores one, with specificity unset."""

from __future__ import annotations

from imageguard.evaluation.metrics import image_metrics


def test_one_true_positive_image() -> None:
    assert image_metrics([True], [True]) == {
        "tp": 1,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "n": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "specificity": None,
        "accuracy": 1.0,
    }
