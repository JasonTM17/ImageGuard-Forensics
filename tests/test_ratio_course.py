"""Lowe ratio: self_match keeps only distinctive nearest neighbours.

Hand-built 5×2 float32 descriptor line (L2):
    d0=0   d1=2   d2=100   d3=102   d4=50.5
  - query 0: nn d1 @ 2.0 vs runner-up d4 @ 50.5 → ratio 0.04 << ρ → kept.
  - query 4: nn d1 @ 48.5 vs runner-up d2 @ 49.5 → ratio 0.98 ≥ ρ → dropped.
Keypoints sit ≥50 px apart, so min_match_dist (20 px) never filters a pair —
only the Lowe ratio decides. pts[4] is left of pts[1], so the failing pair
canonicalises to (4, 1). ρ and min_match_dist come from load_thresholds();
ratio_rho is not patched.
"""

from __future__ import annotations

import cv2
import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.forensics.keypoint.matchers import self_match


def _ratio_case() -> tuple[list[cv2.KeyPoint], np.ndarray]:
    desc = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [100.0, 0.0],
            [102.0, 0.0],
            [50.5, 0.0],
        ],
        dtype=np.float32,
    )
    pts = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0), (50.0, 0.0)]
    kps = [cv2.KeyPoint(x, y, 1.0) for x, y in pts]
    return kps, desc


def test_lowe_ratio_keeps_clear_match_and_drops_ambiguous_one() -> None:
    kps, desc = _ratio_case()
    cfg = load_thresholds()
    rho = float(cfg.value("keypoint", "ratio_rho"))
    min_dist = float(cfg.value("keypoint", "min_match_dist"))

    result = self_match(kps, desc, cfg)
    pairs = {tuple(int(v) for v in pair) for pair in result.pairs}

    # Query 0: nearest neighbour is ~13× closer than the runner-up → kept.
    assert (0, 1) in pairs
    # Query 4: nearest and runner-up are nearly tied (48.5 vs 49.5) → dropped.
    # Both orientations are asserted so a canonicalisation change cannot hide it.
    assert (4, 1) not in pairs
    assert (1, 4) not in pairs
    assert 48.5 / 49.5 >= rho  # sanity: the case really straddles the threshold

    # The drop is the ratio test, not the spatial filter: (4, 1) keypoints are
    # 50 px apart, well past min_match_dist.
    assert float(np.hypot(*(np.array(kps[4].pt) - np.array(kps[1].pt)))) >= min_dist

    # Exactly one of the five candidate nearest-neighbour pairs was removed.
    assert result.stats["after_ratio_spatial"] == 4
    assert result.stats["unique_pairs"] == 2
