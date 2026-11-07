"""Course lock: keypoint.viz.overlay_mask tints a seed-23 square."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.keypoint.viz import overlay_mask

_LOCKED_CHANGED = 1188


def test_viz_overlay_mask_locks_changed_pixel_count() -> None:
    rng = np.random.default_rng(23)
    base = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255

    out = overlay_mask(base, mask)

    assert out.shape == (64, 64, 3)
    assert int(np.count_nonzero(out != base)) == _LOCKED_CHANGED
