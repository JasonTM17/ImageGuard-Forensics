"""Binary/grayscale morphological operators (NumPy only)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from imageguard.ipkit._util import apply_per_channel, require_uint8_image

ShapeKind = Literal["rect", "cross", "ellipse"]


def structuring_element(kind: ShapeKind, ksize: int) -> np.ndarray:
    """Binary structuring element matching common OpenCV shapes."""
    if kind not in ("rect", "cross", "ellipse"):
        raise ValueError("kind must be 'rect', 'cross', or 'ellipse'")
    if isinstance(ksize, bool) or not isinstance(ksize, (int, np.integer)):
        raise TypeError("ksize must be an integer")
    k = int(ksize)
    if k < 1 or k % 2 == 0:
        raise ValueError("ksize must be a positive odd integer")

    se = np.zeros((k, k), dtype=np.uint8)
    c = k // 2
    if kind == "rect":
        se[:] = 1
    elif kind == "cross":
        se[c, :] = 1
        se[:, c] = 1
    else:
        for y in range(k):
            for x in range(k):
                if ((x - c) / max(c, 1)) ** 2 + ((y - c) / max(c, 1)) ** 2 <= 1.0:
                    se[y, x] = 1
    return se


def _erode_gray(channel: np.ndarray, se: np.ndarray) -> np.ndarray:
    h, w = channel.shape
    kh, kw = se.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(
        channel, ((ph, ph), (pw, pw)), mode="constant", constant_values=255
    )
    active = np.argwhere(se.astype(bool))
    if active.size == 0:
        return np.full((h, w), 255, dtype=channel.dtype)
    out: np.ndarray | None = None
    for ky, kx in active:
        slab = pad[ky : ky + h, kx : kx + w]
        out = slab if out is None else np.minimum(out, slab)
    assert out is not None
    return out.astype(channel.dtype, copy=False)


def _dilate_gray(channel: np.ndarray, se: np.ndarray) -> np.ndarray:
    h, w = channel.shape
    kh, kw = se.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(channel, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)
    active = np.argwhere(se.astype(bool))
    if active.size == 0:
        return np.zeros((h, w), dtype=channel.dtype)
    out: np.ndarray | None = None
    for ky, kx in active:
        slab = pad[ky : ky + h, kx : kx + w]
        out = slab if out is None else np.maximum(out, slab)
    assert out is not None
    return out.astype(channel.dtype, copy=False)


def _check_se(se: np.ndarray) -> np.ndarray:
    elem = np.asarray(se)
    if elem.ndim != 2 or elem.size == 0 or elem.shape[0] % 2 == 0 or elem.shape[1] % 2 == 0:
        raise ValueError("se must be a non-empty 2D structuring element with odd sizes")
    if np.any(elem > 1):
        raise ValueError("se must be binary (0/1)")
    return (elem != 0).astype(np.uint8)


def erode(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Grayscale erosion (minimum under ``se``)."""
    source = require_uint8_image(img)
    elem = _check_se(se)
    return apply_per_channel(source, lambda ch: _erode_gray(ch, elem))


def dilate(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Grayscale dilation (maximum under ``se``)."""
    source = require_uint8_image(img)
    elem = _check_se(se)
    return apply_per_channel(source, lambda ch: _dilate_gray(ch, elem))


def open_(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Opening: dilate(erode)."""
    return dilate(erode(img, se), se)


def close_(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Closing: erode(dilate)."""
    return erode(dilate(img, se), se)


def gradient(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Morphological gradient: dilate - erode (clipped uint8)."""
    d = dilate(img, se).astype(np.int16)
    e = erode(img, se).astype(np.int16)
    return np.clip(d - e, 0, 255).astype(np.uint8)


def tophat(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """White top-hat: image - open(image)."""
    base = require_uint8_image(img)
    opened = open_(base, se).astype(np.int16)
    return np.clip(base.astype(np.int16) - opened, 0, 255).astype(np.uint8)
