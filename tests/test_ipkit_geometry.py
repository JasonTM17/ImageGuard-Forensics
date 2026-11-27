"""Geometry conventions and exact interpolation, with independent analytic oracles."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

import imageguard.ipkit.geometry as geo
from imageguard.ipkit.geometry import (
    affine,
    rotate,
    sample_bilinear,
    sample_nearest,
    scale,
    translate,
)

_TINY = np.array([[0, 100], [50, 200]], dtype=np.uint8)


def _texture() -> np.ndarray:
    """Ảnh thử nghiệm cố định, có cạnh sắc để so lấy mẫu với warpAffine."""
    rng = np.random.default_rng(20261123)
    return rng.integers(0, 256, size=(17, 23), dtype=np.uint8)


def _assert_within_one(ours: np.ndarray, ref: np.ndarray) -> None:
    """Allow final uint8 rounding when both methods use the same sample locations."""
    assert ours.shape == ref.shape
    delta = np.abs(ours.astype(np.int16) - ref.astype(np.int16))
    assert int(delta.max()) <= 1


def test_public_exports():
    assert geo.__all__ == [
        "affine",
        "rotate",
        "sample_bilinear",
        "sample_nearest",
        "scale",
        "translate",
    ]
    assert geo.sample_nearest is sample_nearest
    assert geo.sample_bilinear is sample_bilinear
    assert geo.affine is affine
    assert geo.rotate is rotate
    assert geo.scale is scale
    assert geo.translate is translate


def test_bilinear_center_of_tiny_image_is_88():
    out = sample_bilinear(_TINY, np.array([0.5]), np.array([0.5]))
    assert int(out[0]) == 88  # (0 + 100 + 50 + 200) / 4 = 87.5 -> rint = 88


def test_bilinear_retains_precision_between_interpolation_table_entries():
    # Exact linear interpolation is 0.9*0 + 0.1*255 = 25.5 -> 26.
    # OpenCV 4 INTER_LINEAR can round x=0.1 to 3/32 and give 24 instead;
    # the teaching implementation must not copy that quantization.
    edge = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    sampled = sample_bilinear(edge, np.array([0.0]), np.array([0.1]))
    np.testing.assert_array_equal(sampled, np.array([26], dtype=np.uint8))


def test_nearest_rounds_half_up_to_column_one():
    out = sample_nearest(_TINY, np.array([0.2]), np.array([0.6]))
    assert int(out[0]) == 100  # floor(0.2+0.5)=0, floor(0.6+0.5)=1


@pytest.mark.parametrize("sampler", (sample_nearest, sample_bilinear))
def test_out_of_bounds_sample_is_zero(sampler):
    rows = np.array([[-1.0, 2.0], [99.0, 0.0]])
    cols = np.array([[0.0, 0.0], [5.0, 2.0]])
    out = sampler(_TINY, rows, cols)
    np.testing.assert_array_equal(out, np.zeros((2, 2), dtype=np.uint8))


def test_translate_zero_nearest_is_identity():
    np.testing.assert_array_equal(
        translate(_TINY, 0, 0, interpolation="nearest"), _TINY
    )


def test_nearest_integer_translation_zero_fills_not_rolls():
    src = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    out = translate(src, 1, 1, interpolation="nearest")
    expected = np.zeros_like(src)
    expected[1:, 1:] = src[:-1, :-1]
    np.testing.assert_array_equal(out, expected)
    assert not np.array_equal(out, np.roll(src, (1, 1), axis=(0, 1)))


def test_nearest_scale_of_constant_image_stays_constant():
    src = np.full((5, 7), 42, dtype=np.uint8)
    # Co mọi chỉ số nguồn đều nằm trong biên: x'∈[0,3] -> x'/0.5 ∈[0,6],
    # y'∈[0,1] -> y'/0.5 ∈[0,2].
    out = scale(src, 0.5, 0.5, interpolation="nearest")
    assert out.shape == (2, 4)  # round(5*0.5), round(7*0.5)
    np.testing.assert_array_equal(out, np.full((2, 4), 42, dtype=np.uint8))


def test_rotate_zero_nearest_is_identity():
    np.testing.assert_array_equal(
        rotate(_TINY, 0.0, interpolation="nearest"), _TINY
    )


def test_bilinear_translate_matches_opencv():
    src = _texture()
    tx, ty = 3.25, -2.5
    matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    ref = cv2.warpAffine(
        src,
        matrix,
        (src.shape[1], src.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    _assert_within_one(translate(src, tx, ty, interpolation="bilinear"), ref)


def _bilinear_surface() -> np.ndarray:
    # On this small domain every value is an integer in [10, 160]. Bilinear
    # interpolation reproduces f(x,y)=10+3x+5y+xy exactly between samples.
    yy, xx = np.indices((9, 11), dtype=np.float64)
    return (10 + 3 * xx + 5 * yy + xx * yy).astype(np.uint8)


def _assert_analytic_surface(ours: np.ndarray, source_x, source_y) -> None:
    inside = (source_x >= 0) & (source_x <= 10) & (source_y >= 0) & (source_y <= 8)
    assert int(inside.sum()) > ours.size // 2, "the oracle must cover most output pixels"
    expected = np.rint(
        10 + 3 * source_x[inside] + 5 * source_y[inside] + source_x[inside] * source_y[inside]
    ).astype(np.uint8)
    np.testing.assert_array_equal(ours[inside], expected)


def test_bilinear_scale_preserves_analytic_surface():
    # warpAffine(INTER_LINEAR) is not a full-precision oracle: OpenCV 4 uses
    # INTER_BITS=5/INTER_TAB_SIZE=32 and a fixed-point interpolation table.
    # High-contrast textures can therefore differ by more than one gray level.
    # https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
    src = _bilinear_surface()
    sx, sy = 1.7, 0.6
    out = scale(src, sx, sy, interpolation="bilinear")
    assert out.shape == (round(src.shape[0] * sy), round(src.shape[1] * sx))
    yy, xx = np.indices(out.shape, dtype=np.float64)
    _assert_analytic_surface(out, xx / sx, yy / sy)


@pytest.mark.parametrize("angle", [-23.0, 23.0])
def test_bilinear_rotate_preserves_analytic_surface(angle):
    src = _bilinear_surface()
    h, w = src.shape[:2]
    out = rotate(src, angle, interpolation="bilinear")
    assert out.shape == src.shape
    yy, xx = np.indices(out.shape, dtype=np.float64)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    theta = math.radians(angle)
    source_x = cx + math.cos(theta) * (xx - cx) - math.sin(theta) * (yy - cy)
    source_y = cy + math.sin(theta) * (xx - cx) + math.cos(theta) * (yy - cy)
    _assert_analytic_surface(out, source_x, source_y)


def test_nearest_integer_translation_matches_opencv_exactly():
    src = _texture()
    tx, ty = 3, -2
    matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    ref = cv2.warpAffine(
        src,
        matrix,
        (src.shape[1], src.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    np.testing.assert_array_equal(
        translate(src, tx, ty, interpolation="nearest"), ref
    )


def test_three_channel_image_is_transformed_per_channel():
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, size=(11, 13), dtype=np.uint8)
    rgb = np.stack([base, (255 - base), (base // 2)], axis=2)
    for fn, args in (
        (translate, (2, -1)),
        (scale, (1.5, 0.75)),
        (rotate, (17.0,)),
    ):
        out = fn(rgb, *args, interpolation="bilinear")
        assert out.ndim == 3 and out.shape[2] == 3
        for channel in range(3):
            single = fn(rgb[:, :, channel], *args, interpolation="bilinear")
            np.testing.assert_array_equal(out[:, :, channel], single)


def test_bad_interpolation_raises():
    for fn, args in (
        (translate, (1, 0)),
        (scale, (2, 2)),
        (rotate, (30,)),
        (affine, (np.eye(2, 3), (5, 5))),
    ):
        with pytest.raises((ValueError, TypeError)):
            fn(_TINY, *args, interpolation="bicubic")


def test_non_finite_values_raise():
    with pytest.raises((ValueError, TypeError)):
        translate(_TINY, np.nan, 0)
    with pytest.raises((ValueError, TypeError)):
        translate(_TINY, 0, np.inf)
    with pytest.raises((ValueError, TypeError)):
        scale(_TINY, np.inf, 1)
    with pytest.raises((ValueError, TypeError)):
        rotate(_TINY, np.nan)
    with pytest.raises((ValueError, TypeError)):
        affine(_TINY, [[np.nan, 0, 0], [0, 1, 0]], (4, 4))


def test_non_positive_scale_raises():
    for sx, sy in ((0, 1), (-1.0, 1), (1, 0.0), (1, -2.0)):
        with pytest.raises((ValueError, TypeError)):
            scale(_TINY, sx, sy)


def test_singular_matrix_raises():
    with pytest.raises((ValueError, TypeError)):
        affine(_TINY, [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], (4, 4))
    with pytest.raises((ValueError, TypeError)):
        affine(_TINY, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], (4, 4))


def test_bad_shape_and_matrix_raise():
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    for bad_shape in ((0, 4), (4, -1), (4,), (4, 4, 4), (4.0, 4), (True, 4)):
        with pytest.raises((ValueError, TypeError)):
            affine(_TINY, identity, bad_shape)
    for bad_matrix in (
        [[1.0, 0.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
    ):
        with pytest.raises((ValueError, TypeError)):
            affine(_TINY, bad_matrix, (4, 4))


def test_affine_rejects_teaching_size_ceiling():
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    with pytest.raises(ValueError, match="4096"):
        affine(_TINY, identity, (4097, 8))
    with pytest.raises(ValueError, match="4_000_000|4000000|pixels"):
        affine(_TINY, identity, (2048, 2048))


def test_huge_coordinates_sample_zero_without_warning():
    # Tọa độ khổng lồ phải trả 0 (ngoài biên) mà không phát RuntimeWarning khi ép int.
    rows = np.array([[1e300, -1e300]])
    cols = np.array([[1e300, 1e300]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out_nearest = sample_nearest(_TINY, rows, cols)
        out_bilinear = sample_bilinear(_TINY, rows, cols)
    expected = np.zeros((1, 2), dtype=np.uint8)
    np.testing.assert_array_equal(out_nearest, expected)
    np.testing.assert_array_equal(out_bilinear, expected)


def test_affine_rejects_bool_matrix_entries():
    with pytest.raises(TypeError):
        affine(_TINY, [[True, 0.0, 0.0], [0.0, 1.0, 0.0]], (4, 4))
    with pytest.raises(TypeError):
        affine(_TINY, np.array([[True, False, False], [False, True, False]]), (4, 4))


def test_geometry_sources_are_pure_numpy():
    root = Path(geo.__file__).resolve().parent
    banned = ("cv2", "scipy", "skimage")
    for name in ("__init__.py", "sampling.py", "transforms.py"):
        text = (root / name).read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{name} chứa {token}"
