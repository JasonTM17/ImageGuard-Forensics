"""Course lock: identical masks have boundary F1 of one."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.modelbench import boundary_f1


def test_identical_masks_have_boundary_f1_one() -> None:
    mask = np.zeros((8, 8), np.uint8)
    mask[1:4, 1:4] = 255
    assert boundary_f1(mask, mask.copy(), tol=1) == 1.0