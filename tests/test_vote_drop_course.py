"""Course lock: one shift below the vote threshold paints nothing."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.voting import vote_mask


def test_one_shift_below_min_votes_is_dropped() -> None:
    pos = np.array([[0, 0], [8, 0]], np.int32)
    pairs = np.array([[0, 1]], np.int32)
    mask, heat, stats = vote_mask(pairs, pos, 4, (16, 16), 2)
    np.testing.assert_array_equal(mask, np.zeros((16, 16), np.uint8))
    np.testing.assert_array_equal(heat, np.zeros((16, 16), np.float32))
    assert stats == {"n_pairs": 1, "n_shifts_kept": 0, "top_shift_votes": 1}