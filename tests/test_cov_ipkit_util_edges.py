"""Coverage gaps: ipkit._util validation helpers and edges.prewitt."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.ipkit import _util
from imageguard.ipkit.edges import prewitt


def test_require_uint8_image_gray_and_bgr_uint8_copy() -> None:
    gray = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    out = _util.require_uint8_image(gray)
    assert out.dtype == np.uint8 and out.shape == gray.shape
    bgr = np.stack([gray, gray, gray], axis=2)
    out_bgr = _util.require_uint8_image(bgr, name="bgr")
    assert out_bgr.shape == bgr.shape


def test_require_uint8_image_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="must be 2D gray or H×W×3"):
        _util.require_uint8_image(np.zeros((4,), np.uint8))
    with pytest.raises(ValueError, match="must have 3 channels"):
        _util.require_uint8_image(np.zeros((2, 2, 2), np.uint8))
    with pytest.raises(ValueError, match="must be non-empty"):
        _util.require_uint8_image(np.zeros((0, 3), np.uint8))
    with pytest.raises(ValueError, match="must be uint8"):
        _util.require_uint8_image(np.zeros((2, 2), np.float32))


def test_require_gray_uint8_and_apply_per_channel() -> None:
    rng = np.random.default_rng(20261112)
    gray = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    assert _util.require_gray_uint8(gray).shape == gray.shape

    with pytest.raises(ValueError, match="must be a non-empty 2D"):
        _util.require_gray_uint8(np.zeros((0, 2), np.uint8))
    with pytest.raises(ValueError, match="must be uint8"):
        _util.require_gray_uint8(np.zeros((2, 2), np.int32))

    doubled = _util.apply_per_channel(gray, lambda ch: ch * 2)
    np.testing.assert_array_equal(doubled, gray * 2)

    bgr = np.stack([gray, gray, gray], axis=2)
    per = _util.apply_per_channel(bgr, lambda ch: 255 - ch)
    assert per.shape == bgr.shape
    np.testing.assert_array_equal(per[:, :, 0], 255 - gray)


def test_prewitt_ramp_has_expected_horizontal_gradient() -> None:
    gray = np.tile(np.arange(16, dtype=np.uint8), (8, 1))
    gx, gy = prewitt(gray, "replicate")
    assert gx.shape == gray.shape and gy.shape == gray.shape
    # Horizontal ramp → strong gx near interior, weak gy.
    assert float(np.mean(np.abs(gx[2:-2, 2:-2]))) > float(np.mean(np.abs(gy[2:-2, 2:-2])))
    assert np.isfinite(gx).all() and np.isfinite(gy).all()


def test_prewitt_bad_padding_raises() -> None:
    with pytest.raises(ValueError, match="padding"):
        prewitt(np.zeros((3, 3), np.uint8), "mirror")  # type: ignore[arg-type]
