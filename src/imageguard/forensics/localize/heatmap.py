"""Heatmap renderer thống nhất: normalize [0,1] → colormap → alpha overlay."""

from __future__ import annotations

import cv2
import numpy as np


def normalize_map(evidence: np.ndarray) -> np.ndarray:
    """Mọi evidence map → float32 [0,1]. max=0 → toàn 0 (không nhiễu NaN)."""
    e = np.nan_to_num(evidence.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    e = np.maximum(e, 0.0)
    m = float(e.max()) if e.size else 0.0
    return e / m if m > 0 else e


def _to_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[2] == 0:
            raise ValueError(f"Expected non-empty channel axis, got shape {a.shape}")
        if a.shape[2] == 1:
            return a[:, :, 0]
        return np.max(a, axis=2)
    raise ValueError(f"Expected 2D or 3D map, got shape {a.shape}")


def _ensure_u8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if np.iscomplexobj(arr) or arr.dtype.kind not in "biuf":
        raise ValueError(f"Unsupported image dtype: {arr.dtype}")
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _ensure_bgr3(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        u8 = _ensure_u8(a)
        if u8.size == 0:
            return np.zeros((u8.shape[0], u8.shape[1], 3), dtype=np.uint8)
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    if a.ndim == 3 and a.shape[2] == 1:
        u8 = _ensure_u8(a[:, :, 0])
        if u8.size == 0:
            return np.zeros((u8.shape[0], u8.shape[1], 3), dtype=np.uint8)
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    if a.ndim == 3 and a.shape[2] == 4:
        return _ensure_u8(a[:, :, :3])
    if a.ndim == 3 and a.shape[2] == 3:
        return _ensure_u8(a)
    raise ValueError(f"Expected 2D or 3/4-channel image, got shape {a.shape}")


def render_heatmap(heat: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """float [0,1] → BGR uint8 colormap."""
    norm = normalize_map(_to_2d(heat))
    if norm.size == 0:
        return np.zeros((norm.shape[0], norm.shape[1], 3), dtype=np.uint8)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)


def overlay_heatmap(bgr: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend colormap lên ảnh gốc; chỉ vùng heat>0 được tô."""
    base = _ensure_bgr3(bgr)
    norm = normalize_map(_to_2d(heat))
    if base.size == 0 or norm.size == 0:
        return base.copy()
    if norm.shape[:2] != base.shape[:2]:
        norm = cv2.resize(norm, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
        norm = np.clip(norm, 0.0, 1.0)
    hm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    w = (norm * alpha)[:, :, None]
    return np.clip(base * (1 - w) + hm * w, 0, 255).astype(np.uint8)


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.4) -> np.ndarray:
    base = _ensure_bgr3(bgr)
    m = _to_2d(mask)
    if base.size == 0 or m.size == 0:
        return base.copy()
    if m.shape[:2] != base.shape[:2]:
        m = cv2.resize(
            (m > 0).astype(np.uint8),
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    out = base.copy()
    out[m > 0] = color
    return cv2.addWeighted(base, 1 - alpha, out, alpha, 0)
