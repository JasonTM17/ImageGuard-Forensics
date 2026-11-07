"""Course lock: a vertical motion kernel is one measured column."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_ops import motion_kernel


def test_vertical_motion_kernel_is_locked() -> None:
    expected = np.zeros((5, 5), np.float32)
    expected[:, 2] = np.float32(0.20000000298023224)
    np.testing.assert_array_equal(motion_kernel(5, 90.0), expected)
