"""Course lock: draw_pairs paints verified correspondences on a seed-23 image."""

from __future__ import annotations

import cv2
import numpy as np

from imageguard.forensics.keypoint.viz import draw_pairs

_LOCKED_CHANGED = 716


def test_draw_pairs_changes_locked_pixel_count() -> None:
    rng = np.random.default_rng(23)
    base = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    keypoints = [
        cv2.KeyPoint(10.0, 10.0, 1.0),
        cv2.KeyPoint(60.0, 50.0, 1.0),
    ]
    pairs = np.array([[0, 1]], dtype=np.int32)
    out = draw_pairs(base, keypoints, pairs, np.array([0], dtype=np.int32))

    assert out.shape == (80, 80, 3)
    assert out.dtype == np.uint8
    assert not np.array_equal(out, base)
    assert int(np.count_nonzero(out != base)) == _LOCKED_CHANGED
