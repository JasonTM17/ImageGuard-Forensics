"""Metrics edge cases — toàn bộ tổ hợp R1–R5 + zero-division."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.metrics import (
    aggregate_pixel,
    image_metrics,
    pixel_metrics,
)


def _m(fill: np.ndarray | int, shape=(10, 10)) -> np.ndarray:
    a = np.zeros(shape, np.uint8)
    if isinstance(fill, np.ndarray):
        return fill
    a[:fill] = 255
    return a


def test_image_metrics_perfect() -> None:
    m = image_metrics([True, False], [True, False])
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_image_metrics_r5_one_class() -> None:
    """Split chỉ có authentic → precision/recall N/A=None, không bịa số."""
    m = image_metrics([False, False], [False, False])
    assert m["precision"] is None and m["recall"] is None
    assert m["specificity"] == 1.0 and m["accuracy"] == 1.0


def test_image_metrics_all_wrong() -> None:
    m = image_metrics([True, True], [False, False])
    assert m["precision"] == 0.0 and m["f1"] == 0.0


def test_pixel_r1_gt_empty_skips() -> None:
    r = pixel_metrics(_m(30), _m(0))
    assert r["skipped"].startswith("R1")


def test_pixel_r2_pred_empty_f1_zero() -> None:
    gt = np.zeros((10, 10), np.uint8); gt[2:6, 2:6] = 255
    r = pixel_metrics(None, gt)
    assert r["rule"] == "R2" and r["f1"] == 0.0 and r["iou"] == 0.0
    assert r["precision"] is None and r["recall"] == 0.0


def test_pixel_normal_case() -> None:
    gt = np.zeros((10, 10), np.uint8); gt[0:5, :] = 255
    pred = np.zeros((10, 10), np.uint8); pred[3:8, :] = 255
    r = pixel_metrics(pred, gt)
    assert r["skipped"] is None
    assert r["precision"] is not None and 0 < r["iou"] < 1
    assert r["tp"] == 2 * 10


def test_pixel_dim_mismatch_rejected() -> None:
    import pytest

    gt = np.zeros((10, 10), np.uint8); gt[0:5] = 255
    pred = np.zeros((8, 8), np.uint8)
    with pytest.raises(ValueError):
        pixel_metrics(pred, gt)


def test_aggregate_macro_micro() -> None:
    gt1 = np.zeros((10, 10), np.uint8); gt1[0:5] = 255
    p1 = np.zeros((10, 10), np.uint8); p1[0:5] = 255
    gt2 = np.zeros((10, 10), np.uint8); gt2[0:5] = 255
    agg = aggregate_pixel([
        pixel_metrics(p1, gt1),           # perfect
        pixel_metrics(None, gt2),          # R2 miss
        pixel_metrics(_m(30), _m(0)),      # R1 skip — authentic
    ])
    assert agg["n_images_scored"] == 2
    assert agg["macro"]["f1"] == 0.5  # mean(1.0, 0.0)
    assert agg["micro"]["f1"] is not None


def test_aggregate_empty_split() -> None:
    agg = aggregate_pixel([])
    assert agg["n_images_scored"] == 0 and agg["macro"]["f1"] is None
