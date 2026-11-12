"""ipkit.threshold — Otsu and adaptive mean vs OpenCV."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.ipkit.threshold import adaptive_mean_threshold, otsu_threshold


def test_otsu_matches_cv2() -> None:
    rng = np.random.default_rng(99)
    gray = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    t_ip = otsu_threshold(gray)
    t_cv = float(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    assert t_ip == int(t_cv)


def test_otsu_bimodal_toy() -> None:
    gray = np.zeros((10, 20), dtype=np.uint8)
    gray[:, :10] = 50
    gray[:, 10:] = 200
    t = otsu_threshold(gray)
    t_cv = float(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    assert t == int(t_cv)


def test_adaptive_mean_vs_cv2_interior() -> None:
    rng = np.random.default_rng(11)
    gray = rng.integers(0, 256, (48, 48), dtype=np.uint8)
    block, c = 11, 4.0
    ours = adaptive_mean_threshold(gray, block, c)
    cv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, c,
    )
    # Mean filter uses BORDER_REPLICATE; rare ties differ from OpenCV uchar path (≤3/2304 here).
    mismatches = int(np.count_nonzero(ours != cv))
    assert mismatches <= 3


def test_threshold_validation() -> None:
    with pytest.raises(ValueError, match="block"):
        adaptive_mean_threshold(np.zeros((4, 4), np.uint8), 4, 0.0)
    with pytest.raises(ValueError, match="uint8"):
        otsu_threshold(np.ones((2, 2), dtype=np.int16))
