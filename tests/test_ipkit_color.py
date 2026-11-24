"""ipkit.color — chuyển đổi RGB/xám/YCbCr/HSV (NumPy thuần, không OpenCV)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imageguard.ipkit.color import (
    hsv_to_rgb,
    rgb_to_gray,
    rgb_to_hsv,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)

_COLOR_MODULE = Path(__file__).resolve().parents[1] / "src" / "imageguard" / "ipkit" / "color.py"


def _pixel(rgb: tuple[int, int, int]) -> np.ndarray:
    return np.array([[list(rgb)]], dtype=np.uint8)


def test_rgb_to_gray_known_luminance() -> None:
    assert int(rgb_to_gray(_pixel((255, 0, 0)))[0, 0]) == 76
    assert int(rgb_to_gray(_pixel((255, 255, 255)))[0, 0]) == 255
    assert int(rgb_to_gray(_pixel((0, 0, 0)))[0, 0]) == 0


@pytest.mark.parametrize(
    "rgb, expected_h",
    [
        ((255, 0, 0), 0.0),
        ((0, 255, 0), 120.0),
        ((0, 0, 255), 240.0),
    ],
)
def test_rgb_to_hsv_primary_hues(rgb: tuple[int, int, int], expected_h: float) -> None:
    hsv = rgb_to_hsv(_pixel(rgb))
    h, s, v = hsv[0, 0]
    assert h == pytest.approx(expected_h, abs=1e-9)
    assert s == pytest.approx(1.0, abs=1e-9)
    assert v == pytest.approx(1.0, abs=1e-9)


_ROUND_TRIP_COLORS: list[tuple[int, int, int]] = [
    (0, 0, 0),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (128, 64, 32),
]


@pytest.mark.parametrize("rgb", _ROUND_TRIP_COLORS)
def test_round_trip_hsv(rgb: tuple[int, int, int]) -> None:
    src = _pixel(rgb)
    back = hsv_to_rgb(rgb_to_hsv(src))
    diff = np.abs(back.astype(np.int16) - src.astype(np.int16)).max()
    assert diff <= 2


@pytest.mark.parametrize("rgb", _ROUND_TRIP_COLORS)
def test_round_trip_ycbcr(rgb: tuple[int, int, int]) -> None:
    src = _pixel(rgb)
    back = ycbcr_to_rgb(rgb_to_ycbcr(src))
    diff = np.abs(back.astype(np.int16) - src.astype(np.int16)).max()
    assert diff <= 2


def test_color_module_has_no_forbidden_imports() -> None:
    text = _COLOR_MODULE.read_text(encoding="utf-8")
    for forbidden in ("cv2", "scipy", "skimage"):
        assert forbidden not in text


def test_2d_rgb_raises_value_error() -> None:
    gray = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        rgb_to_gray(gray)


def test_hsv_wrong_shape_raises_value_error() -> None:
    with pytest.raises(ValueError):
        hsv_to_rgb(np.zeros((4, 4), dtype=np.float64))


def test_hsv_rejects_nonfinite_and_out_of_range() -> None:
    bad = np.array([[[0.0, 0.0, np.nan]]], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        hsv_to_rgb(bad)
    saturated = np.array([[[0.0, 1.5, 0.2]]], dtype=np.float64)
    with pytest.raises(ValueError, match="S must"):
        hsv_to_rgb(saturated)