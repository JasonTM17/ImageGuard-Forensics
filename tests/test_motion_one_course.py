"""Course lock: a one-pixel motion kernel is a single unit weight."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_ops import motion_kernel


def test_unit_motion_kernel_is_one() -> None:
    np.testing.assert_array_equal(
        motion_kernel(1, 0.0), np.array([[1.0]], np.float32)
    )
