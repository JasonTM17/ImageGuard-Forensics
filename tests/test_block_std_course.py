"""Course lock: block_std_map shape and value on a tiled 8×8 ramp."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.dct_blocks import block_std_map

_LOCKED_STD = 18.47295379638672


def test_block_std_map_tiled_arange_locks_shape_and_value() -> None:
    gray = np.tile(np.arange(64, dtype=np.uint8).reshape(8, 8), (2, 2))
    std = block_std_map(gray, 8)

    assert std.shape == (9, 9)
    assert float(std[0, 0]) == _LOCKED_STD
