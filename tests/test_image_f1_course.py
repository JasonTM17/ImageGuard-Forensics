"""Course lock: one miss and no false alarm fixes image F1."""

from __future__ import annotations

from imageguard.evaluation.metrics import image_metrics


def test_one_hit_one_miss_locks_f1() -> None:
    assert image_metrics([True, False, False], [True, True, False]) == {
        "tp": 1,
        "fp": 0,
        "tn": 1,
        "fn": 1,
        "n": 3,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 0.6666666666666666,
        "specificity": 1.0,
        "accuracy": 0.6666666666666666,
    }
