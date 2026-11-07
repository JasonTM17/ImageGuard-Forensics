"""Course lock: a one-pixel Gaussian kernel is a single unit weight."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_ops import gaussian_kernel


def test_unit_gaussian_kernel_is_one() -> None:
    np.testing.assert_array_equal(
        gaussian_kernel(1, 1.0),
        np.array([[1.0]], np.float32),
    )
