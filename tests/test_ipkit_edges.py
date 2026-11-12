"""ipkit.edges — convolve and gradients vs OpenCV oracle."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.ipkit.edges import convolve2d, gradient_magnitude, laplacian, sobel


def test_convolve_identity_kernel() -> None:
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    k = np.array([[1.0]], dtype=np.float64)
    out = convolve2d(img, k, "replicate")
    np.testing.assert_allclose(out, img, atol=1e-3)


def test_sobel_matches_cv2_on_random_gray() -> None:
    rng = np.random.default_rng(20261112)
    gray = rng.integers(0, 256, (48, 48), dtype=np.uint8)
    gx, gy = sobel(gray, "replicate")
    cv_gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    cv_gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    np.testing.assert_allclose(gx, cv_gx, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(gy, cv_gy, atol=1e-3, rtol=1e-3)

    mag = gradient_magnitude(gray)
    cv_mag = cv2.magnitude(cv_gx, cv_gy)
    np.testing.assert_allclose(mag, cv_mag, atol=1e-3, rtol=1e-3)


def test_laplacian_small_known() -> None:
    img = np.array([[0, 0, 0], [0, 200, 0], [0, 0, 0]], dtype=np.uint8)
    lap = laplacian(img, "zero")
    assert float(lap[1, 1]) == pytest.approx(-800.0, abs=1e-3)


def test_edges_bad_padding() -> None:
    with pytest.raises(ValueError, match="padding"):
        sobel(np.zeros((3, 3), np.uint8), "mirror")  # type: ignore[arg-type]
