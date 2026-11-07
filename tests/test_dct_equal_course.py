"""Course lock: identical 8×8 patches yield equal quantized DCT features."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.dct_blocks import extract_features

_LOCKED = np.array([893, 33, -21, 8, -3, 37, 67, -32], dtype=np.int32)


def test_identical_patches_share_locked_quantized_features() -> None:
    rng = np.random.default_rng(41)
    gray = rng.integers(0, 256, (16, 24), dtype=np.uint8)
    patch = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    gray[0:8, 0:8] = patch
    gray[0:8, 8:16] = patch

    features, positions = extract_features(
        gray, B=8, n_coeffs=8, flat_std_eps=5.0
    )
    first = np.flatnonzero(np.all(positions == (0, 0), axis=1))
    second = np.flatnonzero(np.all(positions == (0, 8), axis=1))
    assert first.size == 1 and second.size == 1
    np.testing.assert_array_equal(features[first[0]], features[second[0]])
    np.testing.assert_array_equal(features[first[0]], _LOCKED)
