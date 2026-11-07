"""Course lock: an unrecognized fuse mode stays a union when closing is off."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.localize.fusion_masks import fuse_masks


def test_unknown_fuse_mode_is_a_union() -> None:
    left = np.zeros((4, 4), np.uint8)
    left[0:2, 0:2] = 255
    right = np.zeros((4, 4), np.uint8)
    right[2:4, 2:4] = 255
    fused = fuse_masks([left, right], mode="bogus", close_k=1)
    expected = np.zeros((4, 4), np.uint8)
    expected[0:2, 0:2] = 255
    expected[2:4, 2:4] = 255
    np.testing.assert_array_equal(fused, expected)
