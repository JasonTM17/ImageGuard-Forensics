"""Course lock: an empty prediction list leaves every rate undefined."""

from __future__ import annotations

from imageguard.evaluation.metrics import image_metrics


def test_empty_image_metrics_are_undefined() -> None:
    assert image_metrics([], []) == {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "n": 0,
        "precision": None,
        "recall": None,
        "f1": None,
        "specificity": None,
        "accuracy": None,
    }
