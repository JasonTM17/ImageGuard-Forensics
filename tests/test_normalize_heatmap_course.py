"""Course regression tests for heatmap normalization and rendering."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.localize.heatmap import normalize_map, render_heatmap


def test_normalize_map_scales_values_to_unit_range() -> None:
    heat = np.array([[0, 2], [4, 6]], dtype=np.float32)

    normalized = normalize_map(heat)

    assert normalized.max() == 1.0
    np.testing.assert_allclose(
        normalized,
        np.array([[0, 1 / 3], [2 / 3, 1]], dtype=np.float32),
        atol=1e-6,
    )


def test_normalize_map_keeps_zero_input_zero() -> None:
    normalized = normalize_map(np.zeros((3, 3), dtype=np.float32))

    assert normalized.sum() == 0


def test_render_heatmap_returns_bgr_uint8() -> None:
    heat = np.array([[0, 2], [4, 6]], dtype=np.float32)

    rendered = render_heatmap(normalize_map(heat))

    assert rendered.shape == (2, 2, 3)
    assert rendered.dtype == np.uint8
