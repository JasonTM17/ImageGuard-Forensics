"""Global and adaptive thresholding (NumPy only)."""

from __future__ import annotations

import numpy as np

from imageguard.ipkit._util import require_gray_uint8


def _odd_block(block: int) -> int:
    if isinstance(block, bool) or not isinstance(block, (int, np.integer)):
        raise TypeError("block must be an integer")
    b = int(block)
    if b < 3 or b % 2 == 0:
        raise ValueError("block must be an odd integer >= 3")
    return b


def otsu_threshold(gray: np.ndarray) -> int:
    """Otsu optimal threshold for 8-bit gray; returns integer in [0, 255]."""
    img = require_gray_uint8(gray)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = float(img.size)
    if total == 0:
        raise ValueError("empty image")
    if hist.sum() <= 0:
        return 0

    probs = hist / total
    bins = np.arange(256, dtype=np.float64)
    omega = np.cumsum(probs)
    mu = np.cumsum(probs * bins)
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = np.where(
            denom > 0,
            (mu_t * omega - mu) ** 2 / denom,
            -1.0,
        )
    best = int(np.argmax(sigma_b2))
    return best


def adaptive_mean_threshold(gray: np.ndarray, block: int, c: float) -> np.ndarray:
    """Local mean threshold: pixel > (mean_block - c) -> 255 else 0."""
    img = require_gray_uint8(gray)
    b = _odd_block(block)
    offset = float(c)
    if not np.isfinite(offset):
        raise ValueError("c must be finite")

    h, w = img.shape
    r = b // 2
    src = img.astype(np.float64)
    padded = np.pad(src, r, mode="edge")
    out = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            window = padded[y : y + b, x : x + b]
            mean = float(window.sum()) / float(b * b)
            out[y, x] = 255 if src[y, x] > mean - offset else 0
    return out
