"""ipkit.noise — determinism, ranges, validation."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.ipkit.noise import poisson, salt_pepper, speckle


def test_salt_pepper_deterministic_uint8() -> None:
    img = np.full((8, 8), 128, dtype=np.uint8)
    a = salt_pepper(img, 0.25, 42)
    b = salt_pepper(img, 0.25, 42)
    assert a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    changed = np.count_nonzero(a != img)
    assert changed == round(0.25 * img.size)


def test_speckle_rgb_shape() -> None:
    base = np.arange(64, dtype=np.uint8).reshape(8, 8)
    img = np.stack([base, base, base], axis=2)
    out = speckle(img, 0.1, 7)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_poisson_zero_stays_zero() -> None:
    img = np.zeros((4, 4), dtype=np.uint8)
    out = poisson(img, 99)
    np.testing.assert_array_equal(out, img)


def test_noise_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="uint8"):
        salt_pepper(np.ones((4, 4), dtype=np.float32), 0.1, 0)
    with pytest.raises(ValueError, match="amount"):
        salt_pepper(np.zeros((2, 2), np.uint8), 1.5, 0)
    with pytest.raises(ValueError, match="sigma"):
        speckle(np.zeros((2, 2), np.uint8), -1.0, 0)
