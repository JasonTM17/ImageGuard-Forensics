"""Course lock: an even close size leaves a lone pixel in place."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.localize.fusion_masks import fuse_masks


def test_even_close_kernel_keeps_a_lone_pixel() -> None:
    pixel = np.zeros((5, 5), np.uint8)
    pixel[2, 2] = 255
    fused = fuse_masks([pixel], mode="any", close_k=2)
    np.testing.assert_array_equal(fused, pixel)
