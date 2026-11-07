"""Course lock: nearest mask alignment repeats each source pixel."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_metrics import align_mask


def test_align_mask_nearest_repeats_the_two_by_two() -> None:
    source = np.array([[0, 255], [128, 0]], np.uint8)
    aligned = align_mask(source, (4, 4))
    expected = np.array(
        [
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [128, 128, 0, 0],
            [128, 128, 0, 0],
        ],
        np.uint8,
    )
    np.testing.assert_array_equal(aligned, expected)
