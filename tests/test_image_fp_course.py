"""Course lock: one false-positive image has precision and accuracy of zero."""

from __future__ import annotations

from imageguard.evaluation.metrics import image_metrics


def test_one_false_positive_image() -> None:
    assert image_metrics([True], [False]) == {
        "tp": 0,
        "fp": 1,
        "tn": 0,
        "fn": 0,
        "n": 1,
        "precision": 0.0,
        "recall": None,
        "f1": 0.0,
        "specificity": 0.0,
        "accuracy": 0.0,
    }
