"""Gradient and edge operators via 2D convolution (NumPy only)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from imageguard.ipkit._util import apply_per_channel, require_uint8_image

PaddingMode = Literal["zero", "replicate"]

_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
_PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
_PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
_LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def _check_padding(padding: str) -> PaddingMode:
    if padding not in ("zero", "replicate"):
        raise ValueError("padding must be 'zero' or 'replicate'")
    return padding  # type: ignore[return-value]


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    padding: PaddingMode = "replicate",
) -> np.ndarray:
    """2D correlation with ``kernel`` (odd sizes); ``padding`` zero or replicate."""
    _check_padding(padding)
    ker = np.asarray(kernel, dtype=np.float64)
    if ker.ndim != 2 or ker.shape[0] % 2 == 0 or ker.shape[1] % 2 == 0:
        raise ValueError("kernel must be 2D with odd height and width")
    if ker.size == 0:
        raise ValueError("kernel must be non-empty")

    src = np.asarray(image, dtype=np.float64)
    if src.ndim != 2 or src.size == 0:
        raise ValueError("image must be a non-empty 2D array")

    ph, pw = ker.shape[0] // 2, ker.shape[1] // 2
    mode = "constant" if padding == "zero" else "edge"
    pad_kw = {"constant_values": 0.0} if padding == "zero" else {}
    padded = np.pad(src, ((ph, ph), (pw, pw)), mode=mode, **pad_kw)

    h, w = src.shape
    out = np.zeros((h, w), dtype=np.float64)
    kh, kw = ker.shape
    for yi in range(kh):
        for xi in range(kw):
            out += ker[yi, xi] * padded[yi : yi + h, xi : xi + w]
    return out.astype(np.float32)


def sobel(img: np.ndarray, padding: PaddingMode = "replicate") -> np.ndarray:
    """Sobel gradients; returns ``(gx, gy)`` as float32 (per channel if RGB)."""
    source = require_uint8_image(img)
    pad = _check_padding(padding)

    def _one(ch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = convolve2d(ch, _SOBEL_X, pad)
        gy = convolve2d(ch, _SOBEL_Y, pad)
        return gx, gy

    if source.ndim == 2:
        return _one(source)
    gxs, gys = [], []
    for c in range(3):
        gx, gy = _one(source[:, :, c])
        gxs.append(gx)
        gys.append(gy)
    return np.stack(gxs, axis=2), np.stack(gys, axis=2)


def prewitt(img: np.ndarray, padding: PaddingMode = "replicate") -> np.ndarray:
    """Prewitt gradients; returns ``(gx, gy)``."""
    source = require_uint8_image(img)
    pad = _check_padding(padding)

    def _one(ch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return convolve2d(ch, _PREWITT_X, pad), convolve2d(ch, _PREWITT_Y, pad)

    if source.ndim == 2:
        return _one(source)
    gxs, gys = [], []
    for c in range(3):
        gx, gy = _one(source[:, :, c])
        gxs.append(gx)
        gys.append(gy)
    return np.stack(gxs, axis=2), np.stack(gys, axis=2)


def laplacian(img: np.ndarray, padding: PaddingMode = "replicate") -> np.ndarray:
    """Discrete Laplacian (4-neighbor), float32."""
    source = require_uint8_image(img)
    pad = _check_padding(padding)
    return apply_per_channel(
        source, lambda ch: convolve2d(ch, _LAPLACIAN, pad)
    )


def gradient_magnitude(img: np.ndarray, padding: PaddingMode = "replicate") -> np.ndarray:
    """``sqrt(gx^2 + gy^2)`` from Sobel, float32."""
    gx, gy = sobel(img, padding)
    return np.sqrt(gx * gx + gy * gy, dtype=np.float32)
