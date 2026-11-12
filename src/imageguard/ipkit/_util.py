"""Input validation helpers for ipkit (view/demo operators)."""

from __future__ import annotations

import numpy as np

_UINT8 = (np.uint8,)


def require_uint8_image(img: np.ndarray, *, name: str = "img") -> np.ndarray:
    """Return a uint8 copy; accept 2D gray or H×W×3."""
    arr = np.asarray(img)
    if arr.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2D gray or H×W×3")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError(f"{name} must have 3 channels when 3D")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if arr.dtype not in _UINT8:
        raise ValueError(f"{name} must be uint8")
    return arr


def require_gray_uint8(gray: np.ndarray, *, name: str = "gray") -> np.ndarray:
    """Return a 2D uint8 gray image."""
    arr = np.asarray(gray)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if arr.dtype not in _UINT8:
        raise ValueError(f"{name} must be uint8")
    return arr


def apply_per_channel(img: np.ndarray, fn) -> np.ndarray:
    """Apply ``fn(channel)`` on 2D slices; pass-through for 2D."""
    if img.ndim == 2:
        return fn(img)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = fn(img[:, :, c])
    return out
