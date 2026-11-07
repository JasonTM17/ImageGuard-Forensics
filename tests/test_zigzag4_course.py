"""Course lock: the first six zigzag indices of a 4x4 block."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.dct_blocks import zigzag_indices


def test_zigzag_prefix_of_four_is_locked() -> None:
    np.testing.assert_array_equal(
        zigzag_indices(4, 6),
        np.array([0, 4, 1, 2, 5, 8]),
    )