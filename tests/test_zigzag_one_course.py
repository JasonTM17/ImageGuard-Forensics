"""Course lock: the first zigzag index of a 1 by 1 block is zero."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.dct_blocks import zigzag_indices


def test_one_by_one_zigzag_is_zero() -> None:
    np.testing.assert_array_equal(zigzag_indices(1, 1), np.array([0], np.int64))
