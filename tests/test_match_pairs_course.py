"""Course lock: empty features yield no pairs and one far duplicate is kept."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.voting import match_pairs


def test_empty_features_and_one_far_pair() -> None:
    empty = match_pairs(
        np.zeros((0, 4), np.int32),
        np.zeros((0, 2), np.int32),
        0.5,
        16,
    )
    assert empty.shape == (0, 2)
    assert empty.dtype == np.int32
    feats = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [9, 9, 9, 9]], np.int32)
    pos = np.array([[0, 0], [40, 0], [0, 40]], np.int32)
    pairs = match_pairs(feats, pos, sim_thresh=0.1, min_block_dist=16, n_neighbors=2)
    np.testing.assert_array_equal(pairs, np.array([[0, 1]], np.int32))
