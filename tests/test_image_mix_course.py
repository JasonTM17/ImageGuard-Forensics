"""Course lock: one true positive and one true negative are perfect."""

from __future__ import annotations

from imageguard.evaluation.metrics import image_metrics


def test_one_true_positive_and_one_true_negative() -> None:
    assert image_metrics([True, False], [True, False]) == {
        "tp": 1,
        "fp": 0,
        "tn": 1,
        "fn": 0,
        "n": 2,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "specificity": 1.0,
        "accuracy": 1.0,
    }
