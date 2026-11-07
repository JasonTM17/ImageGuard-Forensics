"""Course lock: the 2x2 DCT basis is the measured orthonormal matrix."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.dct_blocks import dct_basis


def test_dct_basis_two_matches_measured_matrix() -> None:
    expected = np.array(
        [
            [0.7071067811865476, 0.7071067811865476],
            [0.7071067811865476, -0.7071067811865475],
        ]
    )
    np.testing.assert_array_equal(dct_basis(2), expected)
