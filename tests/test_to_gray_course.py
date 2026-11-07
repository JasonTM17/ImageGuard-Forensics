"""Course lock: to_gray uses the OpenCV BGR weights on a flat patch."""

from __future__ import annotations

import numpy as np

from imageguard.core.preprocess import to_gray


def test_to_gray_flat_bgr_locks_uint8_value() -> None:
    bgr = np.zeros((8, 8, 3), np.uint8)
    bgr[:] = (10, 20, 30)
    gray = to_gray(bgr)
    assert gray.shape == (8, 8)
    assert gray.dtype == np.uint8
    assert np.unique(gray).tolist() == [22]
