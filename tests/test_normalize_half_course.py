"""Course lock: normalizing a 2x2 map scales the peak to one."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.localize.heatmap import normalize_map


def test_normalize_map_half_and_peak() -> None:
    heat = np.array([[0, 2], [4, 0]], np.float32)
    expected = np.array([[0.0, 0.5], [1.0, 0.0]], np.float32)
    np.testing.assert_array_equal(normalize_map(heat), expected)
