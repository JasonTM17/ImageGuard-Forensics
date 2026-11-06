"""Course lock: fuse_masks keeps a lone non-None mask under union mode."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.localize.fusion_masks import fuse_masks


def test_fuse_masks_any_keeps_single_non_none_mask() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 255

    out = fuse_masks([mask, None], "any", close_k=0)

    assert out is not None
    assert out.shape == (64, 64)
    assert int(np.count_nonzero(out)) == 100
    assert fuse_masks([None, None], "any", close_k=0) is None
