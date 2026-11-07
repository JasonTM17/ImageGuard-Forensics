"""Course lock: an 8x8 ramp keeps one quantized DCT row."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.dct_blocks import extract_features


def test_arange_block_quantized_features_are_locked() -> None:
    ramp = np.arange(64, dtype=np.uint8).reshape(8, 8)
    features, positions = extract_features(
        ramp, B=8, n_coeffs=6, flat_std_eps=5.0
    )
    np.testing.assert_array_equal(
        features, np.array([[252, -117, -15, 0, 0, 0]], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        positions, np.array([[0, 0]], dtype=np.int32)
    )
