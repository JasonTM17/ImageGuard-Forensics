"""Heatmap/mask render + fusion semantics."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.localize.fusion_masks import fuse_masks
from imageguard.forensics.localize.heatmap import (
    normalize_map,
    overlay_heatmap,
    overlay_mask,
)


def test_normalize_map_bounds() -> None:
    e = np.array([[0, 5], [10, 2]], np.float32)
    n = normalize_map(e)
    assert n.max() == 1.0 and n.min() == 0.0
    z = normalize_map(np.zeros((4, 4), np.float32))
    assert z.max() == 0.0  # không NaN


def test_overlay_heatmap_alpha() -> None:
    bgr = np.full((100, 100, 3), 100, np.uint8)
    heat = np.zeros((100, 100), np.float32)
    heat[40:60, 40:60] = 1.0
    out = overlay_heatmap(bgr, heat, alpha=0.5)
    assert out.shape == bgr.shape
    # Vùng nóng đổi màu, vùng lạnh ~ giữ nguyên.
    assert not np.allclose(out[50, 50], bgr[50, 50])
    assert np.allclose(out[5, 5], bgr[5, 5], atol=2)


def test_overlay_mask_binary() -> None:
    bgr = np.zeros((50, 50, 3), np.uint8)
    mask = np.zeros((50, 50), np.uint8)
    mask[10:20, 10:20] = 255
    out = overlay_mask(bgr, mask)
    assert out[15, 15, 2] > 0  # kênh đỏ được tô


def test_fuse_any_all() -> None:
    a = np.zeros((40, 40), np.uint8); a[0:20, :] = 255
    b = np.zeros((40, 40), np.uint8); b[10:30, :] = 255
    uni = fuse_masks([a, b], "any")
    inter = fuse_masks([a, b], "all", close_k=0)
    assert (uni > 0).sum() > (inter > 0).sum()
    # intersection: chỉ dải 10:20 chung
    assert (inter[5, :] == 0).all()
    assert (inter[15, :] == 255).all()
    assert fuse_masks([None, None]) is None
