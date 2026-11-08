"""Geometric verification keeps translated keypoint pairs and drops outliers."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.core.config import load_thresholds
from imageguard.forensics.keypoint.geo_verify import geometric_verify


def test_geometric_verify_keeps_translation_pairs_and_rejects_outliers() -> None:
    # Eight source points span the plane; their matching keypoints are shifted
    # by +40 px in x. Two unrelated correspondences exercise rejection.
    source = np.array(
        [
            [10, 10],
            [55, 15],
            [20, 50],
            [70, 75],
            [95, 25],
            [110, 90],
            [40, 115],
            [125, 55],
            [30, 145],
            [150, 115],
        ],
        dtype=np.float32,
    )
    destination = source.copy()
    destination[:8, 0] += 40.0
    destination[8:] = [[175, 6], [6, 190]]

    keypoints = [
        cv2.KeyPoint(float(x), float(y), 1.0)
        for x, y in np.vstack((source, destination))
    ]
    pairs = np.column_stack((np.arange(10), np.arange(10, 20))).astype(np.int32)
    translation_pair_indexes = np.arange(8, dtype=np.int32)
    outlier_pair_indexes = np.arange(8, 10, dtype=np.int32)

    result = geometric_verify(pairs, keypoints, load_thresholds())

    verified = set(result.inlier_pairs.tolist())
    assert set(translation_pair_indexes.tolist()).issubset(verified)
    assert set(outlier_pair_indexes.tolist()).isdisjoint(verified)
    assert result.n_clusters >= 1


def test_geometric_verify_rejects_nonfinite_affine_inliers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OpenCV can return a NaN affine and mark all three collinear pairs as inliers.
    # Stub the native result so the regression is deterministic across OpenCV builds.
    model = np.full((2, 3), np.nan)
    inliers = np.ones((3, 1), dtype=np.uint8)
    monkeypatch.setattr(cv2, "estimateAffine2D", lambda *args, **kwargs: (model, inliers))
    source = [(0, 0), (5, 0), (10, 0)]
    target = [(x + 100, y + 50) for x, y in source]
    keypoints = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in source + target]
    pairs = np.array([[0, 3], [1, 4], [2, 5]], dtype=np.int32)

    result = geometric_verify(pairs, keypoints, load_thresholds())

    assert result.inlier_pairs.size == 0
    assert result.n_clusters == 0
    assert result.models == []
