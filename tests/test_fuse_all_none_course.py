"""Course lock: fuse_masks intersection with one None keeps the lone mask."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.localize.fusion_masks import fuse_masks


def test_fuse_masks_all_keeps_single_non_none_mask() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 255

    out = fuse_masks([None, mask], "all", close_k=0)

    assert out is not None
    assert out.shape == (64, 64)
    assert int(np.count_nonzero(out)) == 100
