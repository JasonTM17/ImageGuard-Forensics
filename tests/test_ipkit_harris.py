"""ipkit Harris — đáp ứng góc trên ảnh xám, không OpenCV trong source."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imageguard.ipkit.features_harris import harris_corners, harris_response

_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "imageguard" / "ipkit" / "features_harris.py"
)


def test_flat_image_has_no_corners() -> None:
    gray = np.zeros((32, 32), dtype=np.uint8)
    response = harris_response(gray)
    assert response.shape == gray.shape
    assert response.dtype == np.float64
    assert np.allclose(response, 0.0, atol=1e-3)
    ys, xs = harris_corners(gray)
    assert ys.size == 0 and xs.size == 0


def test_square_corners_are_detected() -> None:
    gray = np.zeros((64, 64), dtype=np.uint8)
    gray[16:48, 16:48] = 255
    ys, xs = harris_corners(gray)
    assert 4 <= ys.size <= 50
    points = np.stack([ys, xs], axis=1)
    for corner in ((16, 16), (16, 47), (47, 16), (47, 47)):
        distance = np.max(np.abs(points - np.array(corner)), axis=1)
        assert np.any(distance <= 8)


def test_invalid_parameters_raise() -> None:
    gray = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        harris_response(gray, k=0.0)
    with pytest.raises(ValueError):
        harris_response(gray, window=4)
    with pytest.raises(TypeError):
        harris_response(gray, window=True)  # type: ignore[arg-type]


def test_corners_are_deterministic() -> None:
    gray = np.zeros((48, 48), dtype=np.uint8)
    gray[10:30, 12:36] = 255
    first = harris_corners(gray)
    second = harris_corners(gray)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_source_has_no_forbidden_imports() -> None:
    text = _SOURCE.read_text(encoding="utf-8")
    for forbidden in ("cv2", "scipy", "skimage"):
        assert forbidden not in text
