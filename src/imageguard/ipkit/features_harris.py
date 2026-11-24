"""Điểm góc Harris — Chương 6, toán tử học tập thuần NumPy.

M = [[A, C], [C, B]] với A, B, C là trung bình Gaussian của Ix², Iy², Ix·Iy.
R = det(M) - k·trace(M)². Module này không được detector forensics gọi và
không đổi verdict.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from imageguard.ipkit._util import require_gray_uint8
from imageguard.ipkit.edges import convolve2d, sobel
from imageguard.ipkit.spatial import gaussian_kernel


def _real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)  # type: ignore[arg-type]
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _odd_int(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if number < minimum or number % 2 == 0:
        raise ValueError(f"{name} must be an odd integer >= {minimum}")
    return number


def harris_response(
    gray: np.ndarray,
    *,
    k: float = 0.04,
    sigma: float = 1.0,
    window: int = 3,
) -> np.ndarray:
    """Trả bản đồ đáp ứng Harris float64, cùng kích thước ảnh xám."""
    source = require_gray_uint8(gray)
    sensitivity = _real(k, name="k")
    if not 0.0 < sensitivity < 0.25:
        raise ValueError("k must satisfy 0 < k < 0.25")
    smooth = _real(sigma, name="sigma")
    if smooth <= 0.0:
        raise ValueError("sigma must be positive")
    size = _odd_int(window, name="window", minimum=3)

    gx, gy = sobel(source, "replicate")
    kernel = gaussian_kernel(size, smooth)
    ix = gx.astype(np.float64)
    iy = gy.astype(np.float64)
    a = convolve2d(ix * ix, kernel, "replicate").astype(np.float64)
    b = convolve2d(iy * iy, kernel, "replicate").astype(np.float64)
    c = convolve2d(ix * iy, kernel, "replicate").astype(np.float64)
    trace = a + b
    return (a * b - c * c) - sensitivity * trace * trace


def _peaks(response: np.ndarray, nms: int) -> np.ndarray:
    if nms == 1:
        return np.ones(response.shape, dtype=bool)
    pad = nms // 2
    padded = np.pad(response, pad, mode="constant", constant_values=-np.inf)
    windows = sliding_window_view(padded, (nms, nms))
    return response >= windows.max(axis=(-2, -1))


def harris_corners(
    gray: np.ndarray,
    *,
    k: float = 0.04,
    sigma: float = 1.0,
    window: int = 3,
    rel_threshold: float = 0.01,
    nms: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Trả ``(ys, xs)`` int64 của cực trị cục bộ, sắp theo hàng rồi cột."""
    level = _real(rel_threshold, name="rel_threshold")
    if not 0.0 <= level <= 1.0:
        raise ValueError("rel_threshold must be between 0 and 1")
    radius = _odd_int(nms, name="nms", minimum=1)
    response = harris_response(gray, k=k, sigma=sigma, window=window)
    peak = float(response.max()) if response.size else 0.0
    if peak <= 0.0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    keep = _peaks(response, radius) & (response > level * peak)
    ys, xs = np.nonzero(keep)
    order = np.lexsort((xs, ys))
    return ys[order].astype(np.int64), xs[order].astype(np.int64)
