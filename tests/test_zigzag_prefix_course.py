"""Course lock: the first ten 8x8 zigzag indices."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.dct_blocks import zigzag_indices


def test_zigzag_prefix_of_eight_is_locked() -> None:
    np.testing.assert_array_equal(
        zigzag_indices(8, 10),
        np.array([0, 8, 1, 2, 9, 16, 24, 17, 10, 3]),
    )
