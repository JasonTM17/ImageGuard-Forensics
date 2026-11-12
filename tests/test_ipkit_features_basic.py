"""ipkit.features_basic — moments, Haar SAT, LBPH vs OpenCV / hand oracles."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.ipkit.features_basic import (
    central_moment,
    centroid,
    chi_square_distance,
    haar_feature,
    haar_feature_map,
    hu_moments,
    integral_image,
    lbp_image,
    lbph,
    normalized_central_moment,
    raw_moment,
    rect_sum,
    uniform_pattern_lookup,
)


def _symmetric_blob(n: int, center_val: int = 200, bg: int = 10) -> np.ndarray:
    img = np.full((n, n), bg, dtype=np.uint8)
    c = n // 2
    for y in range(n):
        for x in range(n):
            if (x - c) ** 2 + (y - c) ** 2 <= (n // 3) ** 2:
                img[y, x] = center_val
    return img


def test_raw_moment_and_centroid_symmetric_blob() -> None:
    n = 31
    blob = _symmetric_blob(n)
    cx, cy = centroid(blob)
    np.testing.assert_allclose(cx, (n - 1) / 2.0, atol=0.25)
    np.testing.assert_allclose(cy, (n - 1) / 2.0, atol=0.25)
    assert abs(central_moment(blob, 3, 0)) < 5e3
    assert abs(central_moment(blob, 0, 3)) < 5e3
    assert abs(central_moment(blob, 1, 3)) < 5e3


def test_moments_match_cv2() -> None:
    rng = np.random.default_rng(20261112)
    gray = rng.integers(0, 256, (40, 44), dtype=np.uint8)
    m_cv = cv2.moments(gray)
    assert raw_moment(gray, 0, 0) == pytest.approx(m_cv["m00"], rel=1e-9)
    assert raw_moment(gray, 1, 0) == pytest.approx(m_cv["m10"], rel=1e-9)
    assert raw_moment(gray, 0, 1) == pytest.approx(m_cv["m01"], rel=1e-9)
    assert central_moment(gray, 2, 0) == pytest.approx(m_cv["mu20"], rel=1e-6)
    assert normalized_central_moment(gray, 2, 0) == pytest.approx(
        m_cv["nu20"], rel=1e-5
    )
    hu = hu_moments(gray)
    hu_cv = cv2.HuMoments(m_cv).ravel()
    np.testing.assert_allclose(hu, hu_cv, rtol=1e-4, atol=1e-6)


def test_hu_translation_invariance() -> None:
    blob = _symmetric_blob(25)
    shifted = np.zeros((40, 40), dtype=np.uint8)
    shifted[8:33, 10:35] = blob
    np.testing.assert_allclose(
        hu_moments(blob), hu_moments(shifted), rtol=1e-4, atol=1e-5
    )


def test_hu_rotation_and_scale_invariance_discrete() -> None:
    blob = _symmetric_blob(21)
    rot = np.rot90(blob)
    np.testing.assert_allclose(hu_moments(blob), hu_moments(rot), rtol=0.15, atol=0.08)
    up = cv2.resize(blob, (42, 42), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_allclose(hu_moments(blob), hu_moments(up), rtol=0.2, atol=0.12)


def test_integral_rect_sum_vs_slice() -> None:
    rng = np.random.default_rng(11)
    gray = rng.integers(0, 256, (37, 41), dtype=np.uint8)
    sat = integral_image(gray)
    ii = cv2.integral(gray)
    np.testing.assert_allclose(sat, ii, rtol=0, atol=0)
    for _ in range(20):
        h, w = gray.shape
        top = int(rng.integers(0, h - 4))
        left = int(rng.integers(0, w - 4))
        hh = int(rng.integers(1, h - top))
        ww = int(rng.integers(1, w - left))
        direct = float(gray[top : top + hh, left : left + ww].sum())
        assert rect_sum(sat, top, left, hh, ww) == pytest.approx(direct)
        assert rect_sum(sat, top, left, hh, ww) == pytest.approx(
            ii[top + hh, left + ww]
            - ii[top, left + ww]
            - ii[top + hh, left]
            + ii[top, left]
        )


def test_haar_two_h_step_sign() -> None:
    gray = np.zeros((16, 16), dtype=np.uint8)
    gray[:, :8] = 20
    gray[:, 8:] = 200
    sat = integral_image(gray)
    feat = haar_feature(sat, "two_h", 0, 0, 16, 16)
    assert feat < 0
    assert feat == pytest.approx(16 * 8 * (20 - 200))


def test_haar_feature_map_shape() -> None:
    gray = np.random.default_rng(0).integers(0, 256, (24, 24), dtype=np.uint8)
    fm = haar_feature_map(gray, "two_h", 8, 8, 4)
    assert fm.shape == (5, 5)


def test_lbp_constant_all_ones_code() -> None:
    img = np.full((5, 5), 77, dtype=np.uint8)
    codes = lbp_image(img)
    assert int(codes[2, 2]) == 255


def test_lbp_hand_3x3_patch() -> None:
    # Tâm tại (1,1); láng giềng kim đồng hồ; >= tâm → bit 1.
    patch = np.array([[50, 60, 70], [40, 55, 80], [30, 45, 90]], dtype=np.uint8)
    embedded = np.zeros((3, 3), dtype=np.uint8)
    embedded[:] = patch
    got = int(lbp_image(embedded)[1, 1])
    center = 55
    bits = []
    for dr, dc in [
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
    ]:
        bits.append(1 if int(patch[1 + dr, 1 + dc]) >= center else 0)
    expected = sum(b << i for i, b in enumerate(bits))
    assert got == expected


def test_uniform_lookup_59_bins() -> None:
    lut = uniform_pattern_lookup()
    assert lut.shape == (256,)
    assert len(np.unique(lut)) == 59
    assert int(lut.max()) == 58


def test_lbph_length_and_chi_square() -> None:
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    b = a.copy()
    h1 = lbph(a, grid=(8, 8), uniform=True)
    h2 = lbph(b, grid=(8, 8), uniform=True)
    assert h1.shape == (8 * 8 * 59,)
    assert np.isclose(h1.sum(), 1.0)
    assert chi_square_distance(h1, h2) == pytest.approx(0.0)
    rng.shuffle(b)
    h3 = lbph(b, grid=(8, 8), uniform=True)
    d12 = chi_square_distance(h1, h3)
    d21 = chi_square_distance(h3, h1)
    assert d12 == pytest.approx(d21)
    assert d12 > 0


def test_bad_inputs() -> None:
    img = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="zero mass"):
        centroid(np.zeros((3, 3), np.uint8))
    with pytest.raises(ValueError, match="radius"):
        lbp_image(img, radius=2, neighbors=8)
    with pytest.raises(ValueError, match="rectangle exceeds"):
        sat = integral_image(img)
        rect_sum(sat, 0, 0, 10, 10)
