"""Course lock: overlay_heatmap keeps shape and only tints the hot region."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.localize.heatmap import normalize_map, overlay_heatmap


def test_overlay_heatmap_hot_region_changes_cold_region_unchanged() -> None:
    rng = np.random.default_rng(23)
    base = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    heat = np.zeros((160, 160), dtype=np.float32)
    heat[60:100, 60:100] = 1.0

    out = overlay_heatmap(base, heat, alpha=0.45)

    assert out.shape == (160, 160, 3)
    assert out.dtype == np.uint8
    assert not np.array_equal(out[80, 80], base[80, 80])
    assert np.array_equal(out[10, 10], base[10, 10])
    assert float(normalize_map(np.zeros((4, 4), dtype=np.float32)).sum()) == 0.0
