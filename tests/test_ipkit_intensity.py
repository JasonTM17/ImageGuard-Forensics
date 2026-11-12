"""ipkit.intensity — phép biến đổi điểm (negative, log, gamma, ...) so với hành vi
mong đợi; threshold_binary đối chiếu OpenCV."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.ipkit.intensity import (
    bit_plane_slice,
    contrast_stretch,
    gamma_transform,
    log_transform,
    negative,
    piecewise_linear,
    threshold_binary,
)

_RAMP = np.arange(256, dtype=np.uint8).reshape(1, 256)


def test_negative_known_values() -> None:
    assert negative(np.array([[0, 128, 255]], dtype=np.uint8)).tolist() == [
        [255, 127, 0]
    ]


def test_negative_twice_is_identity() -> None:
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    np.testing.assert_array_equal(negative(negative(img)), img)


def test_negative_per_channel_rgb() -> None:
    rgb = np.array([[[0, 10, 20], [250, 200, 128]]], dtype=np.uint8)
    out = negative(rgb)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[0, 0], [255, 245, 235])
    np.testing.assert_array_equal(out[0, 1], [5, 55, 127])


def test_log_transform_spans_0_255_and_monotone() -> None:
    out = log_transform(_RAMP)
    # Đầu vào tăng đơn điệu -> đầu ra cũng tăng đơn điệu (log là hàm đồng biến).
    assert np.all(np.diff(out.astype(np.int16), axis=1) >= 0)
    # c được chọn để mức lớn nhất về 255, mức 0 giữ 0.
    assert int(out.min()) == 0
    assert int(out.max()) == 255


def test_log_transform_explicit_c() -> None:
    # c=0 cho phép? không: yêu cầu dương. Kiểm tra c tùy chỉnh hợp lệ.
    out = log_transform(np.array([[0, 255]], dtype=np.uint8), c=1.0)
    assert out.dtype == np.uint8
    assert int(out[0, 0]) == 0


def test_gamma_transform_gamma_one_is_identity() -> None:
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    np.testing.assert_array_equal(gamma_transform(img, 1.0), img)


def test_gamma_transform_brightens_dark() -> None:
    # gamma < 1 mở rộng vùng tối: 64 -> 128 (thay vì giữ nguyên).
    out = gamma_transform(np.array([[0, 64, 255]], dtype=np.uint8), 0.5)
    assert out.tolist() == [[0, 128, 255]]


def test_contrast_stretch_min_max() -> None:
    out = contrast_stretch(np.array([[20, 200]], dtype=np.uint8))
    assert out.tolist() == [[0, 255]]


def test_contrast_stretch_constant_unchanged() -> None:
    c = np.full((3, 3), 77, dtype=np.uint8)
    np.testing.assert_array_equal(contrast_stretch(c), c)


def test_contrast_stretch_percentile_clip() -> None:
    rng = np.random.default_rng(23)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    out = contrast_stretch(img, low=25.0, high=75.0)
    assert out.dtype == np.uint8
    assert int(out.min()) == 0
    assert int(out.max()) == 255


def test_piecewise_linear_identity() -> None:
    rng = np.random.default_rng(31)
    img = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    np.testing.assert_array_equal(piecewise_linear(img, [(0, 0), (255, 255)]), img)


def test_piecewise_linear_known_mapping() -> None:
    # Hai điểm (0,0) và (128,255): đoạn giữa nội suy, sau 128 giữ hằng 255.
    out = piecewise_linear(
        np.array([[0, 64, 128, 255]], dtype=np.uint8), [(0, 0), (128, 255)]
    )
    assert out.tolist() == [[0, 128, 255, 255]]


def test_threshold_binary_matches_cv2() -> None:
    rng = np.random.default_rng(41)
    gray = rng.integers(0, 256, (48, 48), dtype=np.uint8)
    ours = threshold_binary(gray, 128)
    cv = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
    np.testing.assert_array_equal(ours, cv)


def test_threshold_binary_known() -> None:
    assert threshold_binary(np.array([[10, 200]], dtype=np.uint8), 128).tolist() == [
        [0, 255]
    ]


def test_bit_plane_slice_known() -> None:
    img = np.array([[1, 3, 128]], dtype=np.uint8)
    # bit 0 (LSB): 1 -> 1, 3 -> 1, 128 -> 0
    assert bit_plane_slice(img, 0).tolist() == [[255, 255, 0]]
    # bit 7 (MSB): 1 -> 0, 3 -> 0, 128 -> 1
    assert bit_plane_slice(img, 7).tolist() == [[0, 0, 255]]


def test_bit_plane_slice_all_planes_reconstruct() -> None:
    # Cộng dồn 8 mặt phẳng (mỗi mặt là 0/1) tái tạo lại mức xám gốc.
    rng = np.random.default_rng(53)
    img = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    recon = np.zeros_like(img, dtype=np.int32)
    for b in range(8):
        recon += (bit_plane_slice(img, b) // 255) << b
    np.testing.assert_array_equal(recon, img)


# --- Validation ---


def test_intensity_rejects_non_uint8() -> None:
    with pytest.raises(ValueError, match="uint8"):
        negative(np.ones((2, 2), dtype=np.int16))


def test_intensity_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="channels"):
        negative(np.zeros((2, 2, 4), dtype=np.uint8))


def test_gamma_validation() -> None:
    with pytest.raises(ValueError, match="gamma"):
        gamma_transform(np.zeros((2, 2), np.uint8), 0.0)
    with pytest.raises(ValueError, match="gamma"):
        gamma_transform(np.zeros((2, 2), np.uint8), -0.5)


def test_log_transform_bad_c() -> None:
    with pytest.raises(ValueError, match="c"):
        log_transform(np.zeros((2, 2), np.uint8), c=0.0)


def test_contrast_stretch_bad_percentiles() -> None:
    with pytest.raises(ValueError, match="low"):
        contrast_stretch(np.zeros((2, 2), np.uint8), low=50.0, high=50.0)
    with pytest.raises(ValueError, match="low"):
        contrast_stretch(np.zeros((2, 2), np.uint8), low=101.0, high=100.0)


def test_piecewise_linear_validation() -> None:
    img = np.zeros((2, 2), np.uint8)
    with pytest.raises(ValueError, match="strictly increasing"):
        piecewise_linear(img, [(0, 0), (100, 100), (50, 50)])
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        piecewise_linear(img, [(0, 0), (300, 255)])
    with pytest.raises(ValueError, match="rows"):
        piecewise_linear(img, [(0, 0)])


def test_threshold_binary_validation() -> None:
    with pytest.raises(TypeError, match="thresh"):
        threshold_binary(np.zeros((2, 2), np.uint8), 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="thresh"):
        threshold_binary(np.zeros((2, 2), np.uint8), 256)


def test_bit_plane_slice_validation() -> None:
    with pytest.raises(TypeError, match="bit"):
        bit_plane_slice(np.zeros((2, 2), np.uint8), 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bit"):
        bit_plane_slice(np.zeros((2, 2), np.uint8), 8)
