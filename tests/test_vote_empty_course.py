"""Course lock: voting with no pairs stays an empty mask."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.voting import vote_mask


def test_vote_mask_without_pairs_is_empty() -> None:
    pos = np.array([[0, 0], [40, 0]], np.int32)
    mask, heat, stats = vote_mask(np.empty((0, 2), np.int32), pos, 8, (16, 16), 2)
    assert mask.dtype == np.uint8
    assert heat.dtype == np.float32
    assert int(mask.sum()) == 0
    assert float(heat.sum()) == 0.0
    assert stats == {"n_pairs": 0, "n_shifts_kept": 0}
