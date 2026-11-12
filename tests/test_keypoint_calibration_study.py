"""Unit tests for scripts/keypoint_calibration_study.py (synthetic image only)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from keypoint_calibration_study import (
    analyze_image,
    median_inlier_pair_distance_px,
    rank_auc,
    spatial_spread_fraction,
    wilson_interval,
)

from imageguard.core.config import load_thresholds
from imageguard.core.models import ImageDocument


def test_rank_auc_perfect_separation():
    labels = np.array([1, 1, 0, 0], dtype=np.int8)
    scores = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    assert rank_auc(labels, scores) == pytest.approx(1.0)


def test_rank_auc_rejects_mismatch():
    with pytest.raises(ValueError):
        rank_auc(np.array([1, 0]), np.array([1.0]))


def test_wilson_validation():
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


def _textured_bgr(rng: np.random.Generator, size: int = 320) -> np.ndarray:
    noise = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 1.2)


def test_feature_helpers_finite_on_synthetic(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    bgr = _textured_bgr(rng)
    doc = ImageDocument(
        array=bgr,
        width=bgr.shape[1],
        height=bgr.shape[0],
        channels=3,
        image_format="png",
        sha256="a" * 64,
        file_bytes=0,
        file_name="synthetic.png",
    )
    cfg = load_thresholds()
    metrics = analyze_image(doc, cfg)
    assert metrics.n_keypoints >= 0
    assert metrics.n_verified >= 0
    assert np.isfinite(metrics.spatial_spread)
    assert np.isfinite(metrics.median_pair_distance_px)


def test_spatial_spread_validation():
    kps = []
    pairs = np.empty((0, 2), dtype=np.int32)
    inlier = np.empty(0, dtype=np.int32)
    with pytest.raises(ValueError):
        spatial_spread_fraction(kps, pairs, inlier, (0, 10))


def test_median_pair_distance_empty():
    assert median_inlier_pair_distance_px([], np.empty((0, 2), np.int32), np.empty(0, np.int32)) == 0.0
