"""Course lock: a 4 by 4 ramp has one block standard deviation."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.dct_blocks import block_std_map


def test_four_by_four_ramp_std_is_uniform() -> None:
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    expected = np.full((3, 3), np.float32(2.0615527629852295), np.float32)
    np.testing.assert_array_equal(block_std_map(gray, 2), expected)
