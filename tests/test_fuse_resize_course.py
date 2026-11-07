"""Course lock: fuse_masks resizes later masks to the first valid shape."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.localize.fusion_masks import fuse_masks


def test_fuse_masks_resizes_to_first_mask_shape() -> None:
    small = np.zeros((40, 40), dtype=np.uint8)
    small[5:15, 5:15] = 255
    large = np.zeros((80, 80), dtype=np.uint8)
    large[10:30, 10:30] = 255

    union = fuse_masks([small, large], "any", close_k=0)
    intersection = fuse_masks([small, large], "all", close_k=0)

    assert union is not None and intersection is not None
    assert union.shape == (40, 40)
    assert intersection.shape == (40, 40)
    assert int(np.count_nonzero(union)) == 100
    assert int(np.count_nonzero(intersection)) == 100
