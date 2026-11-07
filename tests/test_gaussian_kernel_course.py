"""Course lock: a 3x3 Gaussian kernel matches the measured weights."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_ops import gaussian_kernel


def test_gaussian_kernel_three_is_locked() -> None:
    expected = np.array(
        [
            [0.07511360943317413, 0.12384140491485596, 0.07511360943317413],
            [0.12384140491485596, 0.20417995750904083, 0.12384140491485596],
            [0.07511360943317413, 0.12384140491485596, 0.07511360943317413],
        ],
        np.float32,
    )
    np.testing.assert_array_equal(gaussian_kernel(3, 1.0), expected)
