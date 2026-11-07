"""Course lock: a radius-two disk kernel matches the measured weights."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_ops import disk_kernel


def test_radius_two_disk_kernel_is_locked() -> None:
    weight = np.float32(0.05882352963089943)
    zero = np.float32(0.0)
    expected = np.array(
        [
            [zero, zero, weight, zero, zero],
            [weight, weight, weight, weight, weight],
            [weight, weight, weight, weight, weight],
            [weight, weight, weight, weight, weight],
            [zero, zero, weight, zero, zero],
        ],
        np.float32,
    )
    np.testing.assert_array_equal(disk_kernel(2), expected)
