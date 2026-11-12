"""ipkit.features_hog — HOG descriptor (Ch.6)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.ipkit.features_hog import (
    hog_descriptor,
    hog_feature_length,
    hog_visualize,
)


def test_hog_length_64x128() -> None:
    assert hog_feature_length((128, 64), cell=8, block=2, bins=9) == 3780
    img = np.zeros((128, 64), dtype=np.uint8)
    vec = hog_descriptor(img, cell=8, block=2, bins=9)
    assert vec.shape == (3780,)
    assert vec.dtype == np.float64


def test_hog_constant_image_zeros() -> None:
    img = np.full((16, 16), 128, dtype=np.uint8)
    vec = hog_descriptor(img, cell=8, block=2, bins=9)
    np.testing.assert_allclose(vec, 0.0, atol=1e-12)


def test_hog_vertical_stripes_horizontal_gradient() -> None:
    h, w = 32, 32
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, ::2] = 255
    vec = hog_descriptor(img, cell=8, block=2, bins=9, norm="L2")
    block_dim = 2 * 2 * 9
    blocks = vec.reshape(-1, block_dim)
    # Bin 0 ≈ hướng 0° (đạo hàm theo x) — năng lượng cao hơn các bin khác
    energy_bin0 = 0.0
    energy_other = 0.0
    for blk in blocks:
        h0 = blk[0] + blk[9] + blk[18] + blk[27]
        rest = blk.sum() - h0
        energy_bin0 += h0
        energy_other += rest
    assert energy_bin0 > energy_other


def test_hog_cv2_length_oracle() -> None:
    """So sánh độ dài với cv2 khi có; OpenCV 5.x headless có thể thiếu HOGDescriptor."""
    img = np.zeros((128, 64), dtype=np.uint8)
    ours = hog_descriptor(img, cell=8, block=2, bins=9)
    hog_cls = getattr(cv2, "HOGDescriptor", None)
    if hog_cls is None:
        pytest.skip("cv2.HOGDescriptor unavailable in this OpenCV build")
    hog = hog_cls((64, 128), (16, 16), (8, 8), (8, 8), 9)
    cv_vec = hog.compute(img)
    assert cv_vec is not None
    assert ours.shape[0] == int(cv_vec.shape[0])


def test_hog_bad_shape_and_norm() -> None:
    with pytest.raises(ValueError, match="multiples"):
        hog_descriptor(np.zeros((10, 10), np.uint8), cell=8)
    with pytest.raises(ValueError, match="norm"):
        hog_descriptor(np.zeros((16, 16), np.uint8), norm="L1")  # type: ignore[arg-type]


def test_hog_visualize_shape() -> None:
    g = np.tile(np.linspace(0, 255, 16, dtype=np.uint8), (16, 1))
    v = hog_visualize(g, cell=8)
    assert v.shape == g.shape
    assert v.dtype == np.uint8


def test_hog_feature_length_too_small() -> None:
    with pytest.raises(ValueError, match="too small"):
        hog_feature_length((8, 8), cell=8, block=2)
