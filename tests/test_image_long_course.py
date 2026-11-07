"""Course lock: a longer prediction counts only the ground-truth length."""

from __future__ import annotations

from imageguard.evaluation.metrics import image_metrics


def test_long_prediction_keeps_the_ground_truth_length() -> None:
    assert image_metrics([True, False], [True]) == {
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
