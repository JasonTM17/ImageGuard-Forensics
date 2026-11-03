"""Tests cho modelbench metrics — boundary-F1, MWU, aggregate."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.evaluation.modelbench import (
    aggregate,
    boundary_f1,
    iou,
    mann_whitney_u,
    noise_sigma_map,
)


def _mask(h, w, rect):
    m = np.zeros((h, w), np.uint8)
    x, y, rw, rh = rect
    cv2.rectangle(m, (x, y), (x + rw, y + rh), 255, -1)
    return m


def test_iou_perfect_and_zero():
    gt = _mask(100, 100, (10, 10, 30, 30))
    assert iou(gt, gt) == 1.0
    far = _mask(100, 100, (60, 60, 30, 30))
    assert iou(far, gt) == 0.0


def test_boundary_f1_tolerates_shift():
    """Mask lệch 4px: IoU thuần thấp nhưng bF1@8 phải cao — đúng mục đích
    của dung sai mép."""
    gt = _mask(100, 100, (20, 20, 40, 40))
    shifted = _mask(100, 100, (24, 24, 40, 40))
    assert iou(shifted, gt) < 0.8
    assert boundary_f1(shifted, gt, tol=8) > 0.85


def test_boundary_f1_empty_pred_is_zero():
    gt = _mask(100, 100, (20, 20, 40, 40))
    assert boundary_f1(np.zeros_like(gt), gt) == 0.0


def test_mwu_separated_vs_same():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 1, 50)
    b = rng.normal(14, 1, 50)
    assert mann_whitney_u(a, b) < 0.001
    a2 = rng.normal(10, 1, 50)
    b2 = rng.normal(10, 1, 50)
    assert mann_whitney_u(a2, b2) > 0.05


def test_noise_sigma_map_catches_smooth_patch():
    rng = np.random.default_rng(1)
    img = rng.normal(128, 20, (128, 128)).astype(np.float32)
    img[40:88, 40:88] = cv2.GaussianBlur(img[40:88, 40:88], (11, 11), 5)
    sig = noise_sigma_map(img)
    assert sig.shape == (8, 8)
    # block (2..5,2..5) mịn → σ thấp hơn rõ ngoài
    inner = sig[2:6, 2:6].mean()
    outer = np.delete(sig, slice(2, 6), axis=0)[2:6].mean()
    assert inner < outer * 0.6


def test_aggregate_counts():
    samples = [{
        "kind": "erase",
        "methods": {
            "keypoint": {"verdict": "inconclusive", "score": 4,
                          "iou": 0.1, "bf1": 0.3},
            "noise": {"verdict": "no_indication", "score": 0.01,
                       "iou": 0.0, "bf1": 0.0},
        },
        "noise_sigma": {"in_median": 1, "out_median": 2, "mwu_p": 0.5},
    }] * 3
    agg = aggregate(samples)
    cell = agg["erase"]["keypoint"]
    assert cell["n"] == 3 and cell["inconclusive"] == 3
    assert cell["mean_iou"] == pytest.approx(0.1)
    assert cell["detection_rate"] == 0.0
