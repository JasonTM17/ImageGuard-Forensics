"""Course test: geometric_verify returns an empty result for zero pairs."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.forensics.keypoint.geo_verify import geometric_verify


def test_geometric_verify_empty_pairs_yields_zero_clusters() -> None:
    pairs = np.zeros((0, 2), dtype=np.int32)
    result = geometric_verify(pairs, [], load_thresholds())

    assert result.n_clusters == 0
    assert len(result.inlier_pairs) == 0
