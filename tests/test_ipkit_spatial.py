"""ipkit.spatial — box/Gaussian/median/sharpen vs OpenCV (replicate border)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.ipkit.noise import salt_pepper
from imageguard.ipkit.spatial import (
    box_kernel,
    gaussian_filter,
    gaussian_kernel,
    highboost,
    laplacian_sharpen,
    max_filter,
    mean_filter,
    median_filter,
    midpoint_filter,
    min_filter,
    unsharp_mask,
)


def test_box_kernel_sum_and_validation() -> None:
    k = box_kernel(5)
    assert k.shape == (5, 5)
    assert np.isclose(k.sum(), 1.0)
    assert np.allclose(k, k.flat[0])
    with pytest.raises(ValueError, match="odd"):
        box_kernel(4)
    with pytest.raises(TypeError):
        box_kernel(True)  # type: ignore[arg-type]


def test_gaussian_kernel_sum_and_validation() -> None:
    g = gaussian_kernel(3, sigma=1.5)
    assert g.shape == (3, 3)
    assert np.isclose(g.sum(), 1.0)
    assert g[1, 1] == pytest.approx(float(g.max()))
    with pytest.raises(ValueError, match="sigma"):
        gaussian_kernel(3, sigma=0.0)


def test_mean_filter_vs_cv2_blur() -> None:
    rng = np.random.default_rng(11)
    gray = rng.integers(0, 256, (32, 40), dtype=np.uint8)
    size = 5
    ours = mean_filter(gray, size)
    cv = cv2.blur(gray, (size, size), borderType=cv2.BORDER_REPLICATE)
    # convolve2d dùng float64; cv2.blur uint8 path: sai lệch tối đa 1 mức xám.
    assert np.max(np.abs(ours.astype(np.int16) - cv.astype(np.int16))) <= 1


def test_gaussian_filter_vs_cv2() -> None:
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, (28, 36), dtype=np.uint8)
    size, sigma = 7, 2.0
    ours = gaussian_filter(gray, size, sigma)
    cv = cv2.GaussianBlur(
        gray, (size, size), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE
    )
    # OpenCV tối ưu kernel; lệch nhỏ trên ảnh ngẫu nhiên nhỏ.
    diff = np.abs(ours.astype(np.int16) - cv.astype(np.int16))
    assert int(np.percentile(diff, 99)) <= 2


def test_median_vs_cv2_and_salt_pepper() -> None:
    base = np.full((24, 24), 120, dtype=np.uint8)
    noisy = salt_pepper(base, amount=0.15, seed=42)
    size = 3
    ours = median_filter(noisy, size)
    cv = cv2.medianBlur(noisy, size)
    assert ours.shape == cv.shape
    # Median loại nhiễu 0/255: ít pixel cực trị hơn ảnh nhiễu.
    extreme_noisy = int(np.count_nonzero((noisy == 0) | (noisy == 255)))
    extreme_out = int(np.count_nonzero((ours == 0) | (ours == 255)))
    assert extreme_out < extreme_noisy
    # Oracle: trung vị 3×3 khớp cv2.medianBlur trên vùng không biên lẻ.
    interior = slice(2, -2)
    assert np.array_equal(ours[interior, interior], cv[interior, interior])


def test_order_filters_constant_and_1x1() -> None:
    im = np.full((4, 4), 77, dtype=np.uint8)
    for fn in (min_filter, max_filter, median_filter, midpoint_filter):
        out = fn(im, 3)
        assert np.all(out == 77)
    one = np.array([[200]], dtype=np.uint8)
    assert median_filter(one, 3).shape == (1, 1)
    assert int(median_filter(one, 3)[0, 0]) == 200


def test_min_max_midpoint_toy() -> None:
    gray = np.array(
        [[1, 9, 2], [8, 5, 3], [4, 6, 7]],
        dtype=np.uint8,
    )
    assert int(min_filter(gray, 3)[1, 1]) == 1
    assert int(max_filter(gray, 3)[1, 1]) == 9
    assert int(midpoint_filter(gray, 3)[1, 1]) == 5
    assert int(median_filter(gray, 3)[1, 1]) == 5


def test_laplacian_sharpen_step_edge() -> None:
    # Bước thẳng đứng: gradient dọc ở cột chuyển tiếp.
    gray = np.zeros((16, 16), dtype=np.uint8)
    gray[:, 8:] = 200
    before = int(np.abs(int(gray[7, 8]) - int(gray[7, 7])))
    sharp = laplacian_sharpen(gray, c=-1.0)
    after = int(np.abs(int(sharp[7, 8]) - int(sharp[7, 7])))
    assert after >= before
    const = np.full((8, 8), 50, dtype=np.uint8)
    assert np.all(laplacian_sharpen(const, c=-1.0) == 50)


def test_unsharp_highboost_constant() -> None:
    im = np.full((6, 6), 90, dtype=np.uint8)
    assert np.all(unsharp_mask(im, 3, 1.0) == im)
    assert np.all(highboost(im, 1.0, 3, 1.0) == im)
    assert np.all(highboost(im, 2.5, 5, 1.2) == im)


def test_rgb_per_channel() -> None:
    rgb = np.stack(
        [
            np.full((5, 5), 10, dtype=np.uint8),
            np.full((5, 5), 100, dtype=np.uint8),
            np.full((5, 5), 200, dtype=np.uint8),
        ],
        axis=2,
    )
    out = mean_filter(rgb, 3)
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8


def test_spatial_validation() -> None:
    with pytest.raises(ValueError, match="uint8"):
        mean_filter(np.ones((2, 2), dtype=np.int16), 3)
    with pytest.raises(ValueError, match="c must"):
        laplacian_sharpen(np.zeros((2, 2), np.uint8), c=2.0)
    with pytest.raises(ValueError, match="k must"):
        highboost(np.zeros((2, 2), np.uint8), -1.0, 3, 1.0)
