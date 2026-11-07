"""Course lock: build_mask paints a nonzero region for one inlier pair."""

from __future__ import annotations

import cv2
import numpy as np

from imageguard.forensics.keypoint.mask_builder import build_mask

_LOCKED_NONZERO = 1053


def test_mask_builder_one_inlier_pair_locks_nonzero_count() -> None:
    keypoints = [
        cv2.KeyPoint(10.0, 10.0, 1.0),
        cv2.KeyPoint(50.0, 40.0, 1.0),
    ]
    pairs = np.array([[0, 1]], dtype=np.int32)
    mask, lines = build_mask(
        keypoints, pairs, np.array([0], dtype=np.int32), (80, 80)
    )

    assert mask.shape == (80, 80)
    assert lines.shape == (80, 80)
    assert int(np.count_nonzero(mask)) == _LOCKED_NONZERO
