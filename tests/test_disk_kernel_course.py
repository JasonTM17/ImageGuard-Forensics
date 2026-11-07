"""Course lock: a radius-1 disk kernel is the measured five-pixel cross."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_ops import disk_kernel


def test_disk_kernel_radius_one_is_locked() -> None:
    expected = np.array(
        [
            [0.0, 0.20000000298023224, 0.0],
            [0.20000000298023224, 0.20000000298023224, 0.20000000298023224],
            [0.0, 0.20000000298023224, 0.0],
        ],
        np.float32,
    )
    np.testing.assert_array_equal(disk_kernel(1), expected)
