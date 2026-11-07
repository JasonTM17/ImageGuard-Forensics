"""Course lock: a flat field has a zero block-standard-deviation map."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.dct_blocks import block_std_map


def test_flat_block_std_map_is_zero() -> None:
    std = block_std_map(np.full((6, 6), 9, np.uint8), 3)
    np.testing.assert_array_equal(std, np.zeros((4, 4), np.float32))
