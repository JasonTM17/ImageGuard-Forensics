"""Course lock: overlay_mask tints only the masked region."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.localize.heatmap import overlay_mask


def test_overlay_mask_cold_pixel_unchanged_hot_pixel_changes() -> None:
    rng = np.random.default_rng(23)
    base = rng.integers(0, 256, (120, 120, 3), dtype=np.uint8)
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[40:80, 40:80] = 255

    out = overlay_mask(base, mask)

    assert out.shape == (120, 120, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out[5, 5], base[5, 5])
    assert not np.array_equal(out[60, 60], base[60, 60])
