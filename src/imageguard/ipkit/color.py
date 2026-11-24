"""Chuyển đổi không gian màu RGB — xám, YCbCr, HSV (Chương Xử lý ảnh màu).

Các hàm dùng NumPy thuần cho mục đích giảng dạy; không ảnh hưởng phán đoán
pháp y trong ImageGuard. Ảnh RGB đầu vào là uint8 H×W×3.
"""

from __future__ import annotations

import numpy as np


def _require_rgb_uint8(image: np.ndarray, *, name: str = "image") -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{name} must be H×W×3 RGB")
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _require_hsv_float(hsv: np.ndarray, *, name: str = "hsv") -> np.ndarray:
    arr = np.asarray(hsv, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
        raise ValueError(f"{name} must be non-empty H×W×3")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} values must be finite")
    saturation = arr[..., 1]
    value = arr[..., 2]
    if np.any(saturation < 0.0) or np.any(saturation > 1.0):
        raise ValueError("S must be in [0, 1]")
    if np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("V must be in [0, 1]")
    return arr


def _clip_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Ảnh xám BT.601: Y = 0.299 R + 0.587 G + 0.114 B (làm tròn, uint8)."""
    rgb = _require_rgb_uint8(image)
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return _clip_uint8(y)


def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """RGB uint8 -> YCbCr JPEG full-range (Y, Cb, Cr), uint8."""
    rgb = _require_rgb_uint8(image)
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    out = np.stack([y, cb, cr], axis=-1)
    return _clip_uint8(out)


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    """YCbCr uint8 -> RGB uint8 (công thức nghịch đảo JPEG full-range)."""
    ycc = _require_rgb_uint8(image)
    y = ycc[..., 0].astype(np.float64)
    cb = ycc[..., 1].astype(np.float64)
    cr = ycc[..., 2].astype(np.float64)
    cb_off = cb - 128.0
    cr_off = cr - 128.0
    r = y + 1.402 * cr_off
    g = y - 0.344136 * cb_off - 0.714136 * cr_off
    b = y + 1.772 * cb_off
    out = np.stack([r, g, b], axis=-1)
    return _clip_uint8(out)


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """RGB uint8 -> HSV float64: H ∈ [0, 360), S và V ∈ [0, 1]."""
    rgb = _require_rgb_uint8(image)
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    v = max_c / 255.0
    s = np.divide(delta, max_c, out=np.zeros_like(delta, dtype=np.float64), where=max_c > 0.0)

    h = np.zeros_like(r, dtype=np.float64)
    mask = delta > 0.0
    safe_delta = np.where(mask, delta, 1.0)

    max_is_r = mask & (max_c == r)
    max_is_g = mask & (max_c == g) & ~max_is_r
    max_is_b = mask & ~max_is_r & ~max_is_g

    h = np.where(
        max_is_r,
        60.0 * np.mod((g - b) / safe_delta, 6.0),
        h,
    )
    h = np.where(
        max_is_g,
        60.0 * ((b - r) / safe_delta + 2.0),
        h,
    )
    h = np.where(
        max_is_b,
        60.0 * ((r - g) / safe_delta + 4.0),
        h,
    )
    h = np.where(h < 0.0, h + 360.0, h)

    return np.stack([h, s, v], axis=-1).astype(np.float64)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """HSV float64 H×W×3 -> RGB uint8 (6 sector chuẩn giáo trình)."""
    arr = _require_hsv_float(hsv)
    h = arr[..., 0] % 360.0
    s = arr[..., 1]
    v = arr[..., 2]

    h_prime = h / 60.0
    sector = np.floor(h_prime).astype(np.int64) % 6
    f = h_prime - np.floor(h_prime)

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    shape = arr.shape[:-1]
    r = np.zeros(shape, dtype=np.float64)
    g = np.zeros(shape, dtype=np.float64)
    b = np.zeros(shape, dtype=np.float64)

    for i, (rv, gv, bv) in enumerate(
        [
            (v, t, p),
            (q, v, p),
            (p, v, t),
            (p, q, v),
            (t, p, v),
            (v, p, q),
        ]
    ):
        m = sector == i
        r = np.where(m, rv, r)
        g = np.where(m, gv, g)
        b = np.where(m, bv, b)

    zero_sv = (s == 0.0) | (v == 0.0)
    r = np.where(zero_sv, v, r)
    g = np.where(zero_sv, v, g)
    b = np.where(zero_sv, v, b)

    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return _clip_uint8(rgb)