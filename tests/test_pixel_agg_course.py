"""Course lock: aggregating no scored images leaves every rate undefined."""

from __future__ import annotations

from imageguard.evaluation.metrics import aggregate_pixel


def test_empty_pixel_aggregate_is_undefined() -> None:
    assert aggregate_pixel([]) == {
        "n_images_scored": 0,
        "macro": {"precision": None, "recall": None, "f1": None, "iou": None},
        "micro": {"precision": None, "recall": None, "f1": None, "iou": None},
    }
