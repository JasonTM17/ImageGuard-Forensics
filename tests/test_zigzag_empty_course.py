"""Course lock: a zero coefficient count yields an empty zigzag."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.dct_blocks import zigzag_indices


def test_zero_coefficient_zigzag_is_empty() -> None:
    np.testing.assert_array_equal(zigzag_indices(8, 0), np.empty(0, np.float64))
