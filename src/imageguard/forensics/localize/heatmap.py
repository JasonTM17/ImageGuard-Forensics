"""Heatmap renderer thống nhất: normalize [0,1] → colormap → alpha overlay."""

from __future__ import annotations

import cv2
import numpy as np


def normalize_map(evidence: np.ndarray) -> np.ndarray:
    """Mọi evidence map → float32 [0,1]. max=0 → toàn 0 (không nhiễu NaN)."""
    e = evidence.astype(np.float32)
    m = e.max()
    return e / m if m > 0 else e


def render_heatmap(heat: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """float [0,1] → BGR uint8 colormap."""
    return cv2.applyColorMap((normalize_map(heat) * 255).astype(np.uint8), colormap)


def overlay_heatmap(bgr: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend colormap lên ảnh gốc; chỉ vùng heat>0 được tô."""
    if heat.shape[:2] != bgr.shape[:2]:
        heat = cv2.resize(heat, (bgr.shape[1], bgr.shape[0]))
    hm = render_heatmap(heat)
    norm = normalize_map(heat)
    w = (norm * alpha)[:, :, None]
    return (bgr * (1 - w) + hm * w).astype(np.uint8)


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.4) -> np.ndarray:
    out = bgr.copy()
    out[mask > 0] = color
    return cv2.addWeighted(bgr, 1 - alpha, out, alpha, 0)
