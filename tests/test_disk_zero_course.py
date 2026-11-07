"""Course lock: a zero-radius disk kernel is a single unit weight."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_ops import disk_kernel


def test_zero_radius_disk_kernel_is_one() -> None:
    np.testing.assert_array_equal(disk_kernel(0), np.array([[1.0]], np.float32))
