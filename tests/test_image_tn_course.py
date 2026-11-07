"""Course lock: two true negatives leave precision and recall unset."""

from __future__ import annotations

from imageguard.evaluation.metrics import image_metrics


def test_two_true_negatives() -> None:
    assert image_metrics([False, False], [False, False]) == {
        "tp": 0,
        "fp": 0,
        "tn": 2,
        "fn": 0,
        "n": 2,
        "precision": None,
        "recall": None,
        "f1": None,
        "specificity": 1.0,
        "accuracy": 1.0,
    }
