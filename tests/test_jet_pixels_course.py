"""Course lock: jet colormap endpoints on a two-by-two heat map."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.localize.heatmap import render_heatmap


def test_jet_endpoints_lock_bgr_pixels() -> None:
    heat = np.array([[0.0, 1.0], [1.0, 0.0]], np.float32)
    rendered = render_heatmap(heat)
    np.testing.assert_array_equal(
        rendered,
        np.array(
            [
                [[128, 0, 0], [0, 0, 128]],
                [[0, 0, 128], [128, 0, 0]],
            ],
            dtype=np.uint8,
        ),
    )
