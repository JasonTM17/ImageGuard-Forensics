"""Course lock: a single correspondence cannot form a geometric cluster."""

from __future__ import annotations

import cv2
import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.forensics.keypoint.geo_verify import geometric_verify


def test_geometric_verify_one_pair_yields_zero_clusters() -> None:
    pairs = np.array([[0, 1]], dtype=np.int32)
    keypoints = [
        cv2.KeyPoint(0.0, 0.0, 1.0),
        cv2.KeyPoint(10.0, 0.0, 1.0),
    ]
    result = geometric_verify(pairs, keypoints, load_thresholds())

    assert result.n_clusters == 0
    assert len(result.inlier_pairs) == 0
