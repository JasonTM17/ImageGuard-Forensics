"""ipkit.histogram — histogram, cân bằng, đặc tả và xử lý cục bộ."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.ipkit.histogram import (
    cdf,
    equalize,
    histogram,
    local_equalize,
    match_histogram,
    normalized_histogram,
)


def test_histogram_sum_equals_pixel_count() -> None:
    rng = np.random.default_rng(60)
    gray = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    assert int(histogram(gray).sum()) == gray.size


def test_histogram_known_bins() -> None:
    h = histogram(np.array([[0, 0, 255, 255, 255]], dtype=np.uint8))
    assert int(h[0]) == 2
    assert int(h[255]) == 3
    assert int(h.sum()) == 5
    assert h.dtype == np.int64
    assert h.shape == (256,)


def test_normalized_histogram_sums_to_one() -> None:
    gray = np.array([[0, 0, 255]], dtype=np.uint8)
    p = normalized_histogram(gray)
    assert p.dtype == np.float64
    assert float(p.sum()) == pytest.approx(1.0)
    assert float(p[0]) == pytest.approx(2.0 / 3.0)
    assert float(p[255]) == pytest.approx(1.0 / 3.0)


def test_cdf_monotone_and_ends_at_one() -> None:
    rng = np.random.default_rng(70)
    gray = rng.integers(0, 256, (20, 20), dtype=np.uint8)
    c = cdf(gray)
    assert np.all(np.diff(c) >= 0.0)
    assert float(c[-1]) == pytest.approx(1.0)


def test_equalize_known_rounding() -> None:
    # Ảnh chỉ có 0 và 255: CDF tại 0 là 0.5 -> 255*0.5 = 127.5 -> làm tròn 128.
    # (Lưu ý: OpenCV cắt cụt về 127, ta làm tròn theo bài giảng.)
    out = equalize(np.array([[0, 255]], dtype=np.uint8))
    assert out.tolist() == [[128, 255]]


def test_equalize_matches_cv2_within_rounding() -> None:
    rng = np.random.default_rng(42)
    gray = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    ours = equalize(gray)
    cv = cv2.equalizeHist(gray)
    # Bài giảng "round to nearest"; OpenCV dùng saturate_cast (cắt cụt). Hai kết
    # quả chỉ lệch tối đa vài mức cường độ do khác cách lượng tử hóa.
    diff = np.abs(ours.astype(np.int16) - cv.astype(np.int16))
    assert int(diff.max()) <= 2


def test_equalize_constant_image_unchanged() -> None:
    c = np.full((5, 5), 99, dtype=np.uint8)
    np.testing.assert_array_equal(equalize(c), c)


def test_equalize_known_cdf_mapping() -> None:
    # 2 pixel mức 50, 1 pixel mức 200: cdf(50)=2/3, cdf(200)=3/3.
    # s(50) = round(255*2/3) = 170, s(200) = round(255) = 255.
    out = equalize(np.array([[50, 50, 200]], dtype=np.uint8))
    assert out.tolist() == [[170, 170, 255]]


def test_match_histogram_known_small_case() -> None:
    src = np.array([[0, 128, 255]], dtype=np.uint8)
    ref = np.array([[100, 100, 150]], dtype=np.uint8)
    out = match_histogram(src, ref)
    assert out.tolist() == [[0, 100, 150]]


def test_match_histogram_self_is_near_identity() -> None:
    rng = np.random.default_rng(42)
    gray = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    matched = match_histogram(gray, gray)
    # Đặc tả về chính nó: ánh xạ đơn điệu và rất gần đồng nhất. Trên ảnh dày đặc
    # mức xám, sai lệch chỉ vài mức do các bin trống làm "bậc thang" CDF bằng nhau.
    diff = np.abs(matched.astype(np.int16) - gray.astype(np.int16))
    assert int(diff.max()) <= 2


def test_match_histogram_monotone_transform() -> None:
    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
    out = match_histogram(ramp, ramp)
    # Ảnh vào tăng đơn điệu -> ảnh ra cũng tăng đơn điệu (không đảo thứ tự mức xám).
    assert np.all(np.diff(out.astype(np.int16), axis=1) >= 0)


def test_match_histogram_reference_image_equals_histogram_array() -> None:
    rng = np.random.default_rng(8)
    src = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    ref = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    via_image = match_histogram(src, ref)
    via_hist = match_histogram(src, histogram(ref))
    np.testing.assert_array_equal(via_image, via_hist)


def test_match_histogram_shape_and_dtype() -> None:
    src = np.arange(9, dtype=np.uint8).reshape(3, 3)
    ref = np.zeros((4, 4), dtype=np.uint8)
    out = match_histogram(src, ref)
    assert out.shape == (3, 3)
    assert out.dtype == np.uint8


def test_local_equalize_constant_unchanged() -> None:
    c = np.full((6, 6), 40, dtype=np.uint8)
    np.testing.assert_array_equal(local_equalize(c, 2), c)


def test_local_equalize_single_tile_equals_global() -> None:
    rng = np.random.default_rng(14)
    gray = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    # Ô lớn hơn cả ảnh -> chỉ có một ô -> kết quả trùng cân bằng toàn cục.
    np.testing.assert_array_equal(local_equalize(gray, 100), equalize(gray))


def test_local_equalize_shape_and_dtype() -> None:
    rng = np.random.default_rng(22)
    gray = rng.integers(0, 256, (10, 10), dtype=np.uint8)
    out = local_equalize(gray, 3)
    assert out.shape == (10, 10)
    assert out.dtype == np.uint8


def test_local_equalize_clip_limit_smoke() -> None:
    rng = np.random.default_rng(30)
    gray = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    out = local_equalize(gray, 4, clip_limit=0.05)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8


# --- Validation ---


def test_histogram_rejects_non_gray_or_non_uint8() -> None:
    with pytest.raises(ValueError, match="2D"):
        histogram(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        equalize(np.zeros((2, 2), dtype=np.int16))


def test_match_histogram_bad_reference() -> None:
    gray = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="reference"):
        match_histogram(gray, np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="reference"):
        match_histogram(gray, np.zeros((10,), dtype=np.float64))
    with pytest.raises(ValueError, match="at least one pixel"):
        match_histogram(gray, np.zeros((256,), dtype=np.int64))


def test_local_equalize_validation() -> None:
    gray = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(TypeError, match="tile"):
        local_equalize(gray, 2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tile"):
        local_equalize(gray, 0)
    with pytest.raises(ValueError, match="clip_limit"):
        local_equalize(gray, 2, clip_limit=1.5)
