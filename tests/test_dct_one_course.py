"""Course lock: the 1 by 1 DCT basis is a single one."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.block.dct_blocks import dct_basis


def test_dct_basis_one_is_a_single_one() -> None:
    np.testing.assert_array_equal(dct_basis(1), np.array([[1.0]]))
