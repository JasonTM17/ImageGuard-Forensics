"""Course lock: intersection without closing keeps the shared column."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.localize.fusion_masks import fuse_masks


def test_intersection_without_close_keeps_two_pixels() -> None:
    left = np.zeros((4, 4), np.uint8)
    right = np.zeros((4, 4), np.uint8)
    left[0:3, 0:2] = 255
    right[1:4, 1:3] = 255
    fused = fuse_masks([left, right], mode="all", close_k=1)
    expected = np.zeros((4, 4), np.uint8)
    expected[1:3, 1] = 255
    np.testing.assert_array_equal(fused, expected)
