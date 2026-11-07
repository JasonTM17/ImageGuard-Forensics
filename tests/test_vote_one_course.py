"""Course lock: one kept shift paints two closed 4x4 blocks."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.voting import vote_mask


def test_one_shift_paints_two_blocks() -> None:
    pos = np.array([[0, 0], [8, 0]], np.int32)
    pairs = np.array([[0, 1]], np.int32)
    mask, heat, stats = vote_mask(pairs, pos, 4, (16, 16), 1)
    expected = np.zeros((16, 16), np.uint8)
    expected[0:4, 0:4] = 255
    expected[8:12, 0:4] = 255
    np.testing.assert_array_equal(mask, expected)
    assert float(heat.max()) == 1.0
    assert stats == {"n_pairs": 1, "n_shifts_kept": 1, "top_shift_votes": 1}
