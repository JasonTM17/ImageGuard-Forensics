"""Course lock: drawing no inliers copies the image and changes no pixels."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.keypoint.viz import draw_pairs


def test_draw_pairs_empty_inliers_is_an_unchanged_copy() -> None:
    base = np.zeros((8, 8, 3), np.uint8)
    out = draw_pairs(
        base,
        [],
        np.empty((0, 2), np.int32),
        np.empty(0, np.int32),
    )
    assert out is not base
    assert out.shape == (8, 8, 3)
    assert int(np.abs(out.astype(int) - base.astype(int)).sum()) == 0
