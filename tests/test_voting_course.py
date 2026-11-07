"""Deterministic course cases for block shift matching and mask voting."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.forensics.block.voting import match_pairs, vote_mask


def _shift_case() -> tuple[np.ndarray, np.ndarray]:
    feats = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [80.0, -20.0],
        ],
        dtype=np.float32,
    )
    pos = np.array(
        [
            [0, 0],
            [24, 0],
            [8, 40],
        ],
        dtype=np.int32,
    )
    return feats, pos


def test_match_pairs_selects_equal_features_at_far_shift_only() -> None:
    feats, pos = _shift_case()

    pairs = match_pairs(
        feats,
        pos,
        sim_thresh=0.01,
        min_block_dist=16.0,
    )

    np.testing.assert_array_equal(pairs, np.array([[0, 1]], dtype=np.int32))
    np.testing.assert_array_equal(pos[pairs[0, 1]] - pos[pairs[0, 0]], [24, 0])


def test_vote_mask_threshold_controls_both_shift_footprints() -> None:
    _, pos = _shift_case()
    pairs = np.array([[0, 1]], dtype=np.int32)

    mask, _, stats = vote_mask(
        pairs,
        pos,
        B=8,
        shape=(40, 48),
        min_votes=1,
    )

    assert stats["n_shifts_kept"] == 1
    assert np.all(mask[0:8, 0:8] == 255)
    assert np.all(mask[24:32, 0:8] == 255)

    empty_mask, _, empty_stats = vote_mask(
        pairs,
        pos,
        B=8,
        shape=(40, 48),
        min_votes=2,
    )

    assert not np.any(empty_mask)
    assert empty_stats["n_shifts_kept"] == 0


@pytest.mark.parametrize("translation", [(24, 0), (0, 24), (24, -16)])
def test_vote_mask_is_invariant_to_pair_endpoint_order(translation: tuple[int, int]) -> None:
    source = np.array([[8, 16], [8, 17], [9, 16], [9, 17]], dtype=np.int32)
    pos = np.concatenate((source, source + np.array(translation, dtype=np.int32)))
    pairs = np.array([[0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.int32)
    mixed_order = pairs.copy()
    mixed_order[1::2] = mixed_order[1::2, ::-1]

    expected_mask, expected_heat, expected_stats = vote_mask(pairs, pos, 8, (64, 64), 4)
    mask, heat, stats = vote_mask(mixed_order, pos, 8, (64, 64), 4)

    assert expected_stats["n_shifts_kept"] == 1
    assert stats == expected_stats
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(heat, expected_heat)
