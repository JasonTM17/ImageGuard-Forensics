"""Course lock: one false-negative image has recall and accuracy of zero."""

from __future__ import annotations

from imageguard.evaluation.metrics import image_metrics


def test_one_false_negative_image() -> None:
    assert image_metrics([False], [True]) == {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 1,
        "n": 1,
        "precision": None,
        "recall": 0.0,
        "f1": 0.0,
        "specificity": None,
        "accuracy": 0.0,
    }
