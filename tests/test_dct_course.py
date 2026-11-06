"""Focused tests for the block DCT feature helpers."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.dct_blocks import (
    dct_basis,
    extract_features,
    zigzag_indices,
)


def test_dct_basis_is_orthonormal() -> None:
    basis = dct_basis(8)
    identity = np.eye(8)

    np.testing.assert_allclose(basis @ basis.T, identity, atol=1e-12)
    np.testing.assert_allclose(basis.T @ basis, identity, atol=1e-12)


def test_zigzag_indices_include_dc_and_have_unique_valid_prefix() -> None:
    np.testing.assert_array_equal(zigzag_indices(8, 1), np.array([0]))

    n_coeffs = 32
    indices = zigzag_indices(8, n_coeffs)
    prefix = indices[:n_coeffs]

    assert np.all((prefix >= 0) & (prefix < 64))
    assert np.unique(prefix).size == n_coeffs


def test_flat_gray_block_features_are_filtered() -> None:
    flat = np.full((8, 8), 127, dtype=np.uint8)

    features, positions = extract_features(
        flat, B=8, n_coeffs=4, flat_std_eps=5.0
    )

    assert features.shape == (0, 4)
    assert positions.shape == (0, 2)


def test_identical_textured_blocks_have_equal_quantized_features() -> None:
    rng = np.random.default_rng(41)
    gray = rng.integers(0, 256, size=(16, 32), dtype=np.uint8)
    patch = rng.integers(0, 256, size=(8, 8), dtype=np.uint8)
    first = (3, 3)
    second = (3, 19)
    gray[first[0] : first[0] + 8, first[1] : first[1] + 8] = patch
    gray[second[0] : second[0] + 8, second[1] : second[1] + 8] = patch

    features, positions = extract_features(
        gray, B=8, n_coeffs=16, flat_std_eps=5.0
    )
    first_row = np.flatnonzero(np.all(positions == first, axis=1))
    second_row = np.flatnonzero(np.all(positions == second, axis=1))

    assert first_row.size == 1
    assert second_row.size == 1
    np.testing.assert_array_equal(features[first_row[0]], features[second_row[0]])


def test_one_level_high_frequency_change_preserves_quantized_dc_value() -> None:
    rows, cols = np.indices((8, 8))
    block = (((rows + cols) % 2) * 255).astype(np.uint8)
    changed = block.copy()
    changed[7, 7] = 1

    before, _ = extract_features(block, B=8, n_coeffs=1, flat_std_eps=5.0)
    after, _ = extract_features(changed, B=8, n_coeffs=1, flat_std_eps=5.0)

    assert int(before[0, 0]) == 1020
    assert int(after[0, 0]) == 1020
