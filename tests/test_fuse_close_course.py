"""Course lock: morphological close of two sparse masks."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.localize.fusion_masks import fuse_masks


def test_close_expands_the_corner_pixel_only() -> None:
    corner = np.zeros((7, 7), np.uint8)
    corner[1, 1] = 255
    center = np.zeros((7, 7), np.uint8)
    center[3, 3] = 255
    fused = fuse_masks([corner, center], mode="any", close_k=3)
    expected = np.zeros((7, 7), np.uint8)
    expected[0:2, 0:2] = 255
    expected[3, 3] = 255
    np.testing.assert_array_equal(fused, expected)
