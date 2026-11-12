"""
Mô tả HOG — Histogram of Oriented Gradients (Chương 6, HCMUTE dunghv).

Gradient central difference [-1,0,1], histogram 9 bin / ô 8×8, chuẩn hóa khối 2×2 ô
stride 1 (tương đương 16×16 pixel), độ dài 64×128 → 3780 (slide Dalal–Triggs).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from imageguard.ipkit._util import require_gray_uint8
from imageguard.ipkit.edges import convolve2d

NormKind = Literal["L2", "L2-Hys"]

# Kernel 3×3: bằng [-1,0,1] theo trục x và y (đủ lẻ cho convolve2d)
_GRAD_X = np.array(
    [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64
)
_GRAD_Y = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)

_HYS_CLIP = 0.2
_L2_EPS = 1e-6


def _check_cell_block_bins(cell: int, block: int, bins: int) -> tuple[int, int, int]:
    if isinstance(cell, bool) or not isinstance(cell, (int, np.integer)):
        raise TypeError("cell must be an integer")
    if isinstance(block, bool) or not isinstance(block, (int, np.integer)):
        raise TypeError("block must be an integer")
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)):
        raise TypeError("bins must be an integer")
    c, b, k = int(cell), int(block), int(bins)
    if c < 1 or b < 1 or k < 2:
        raise ValueError("cell and block must be >= 1 and bins >= 2")
    return c, b, k


def hog_feature_length(
    shape: tuple[int, int] | list[int],
    *,
    cell: int = 8,
    block: int = 2,
    bins: int = 9,
) -> int:
    """
    Độ dài vector HOG theo công thức slide: (#block_y)(#block_x)(block²)(bins).

    Với 128×64 (H×W), cell=8, block=2: 15×7×4×9 = 3780.

    Parameters
    ----------
    shape :
        (chiều cao, chiều rộng) pixel, chia hết cho ``cell``.
    cell, block, bins :
        Kích thước ô, số ô mỗi chiều trong khối, số bin hướng.

    Returns
    -------
    int
        Số phần tử descriptor.

    Example
    -------
    >>> hog_feature_length((128, 64), cell=8, block=2, bins=9)
    3780
    """
    if len(shape) != 2:
        raise ValueError("shape must be (height, width)")
    h, w = int(shape[0]), int(shape[1])
    c, b, k = _check_cell_block_bins(cell, block, bins)
    if h % c != 0 or w % c != 0:
        raise ValueError("image height and width must be multiples of cell")
    n_cy, n_cx = h // c, w // c
    if n_cy < b or n_cx < b:
        raise ValueError("image too small for given cell and block")
    n_by = n_cy - b + 1
    n_bx = n_cx - b + 1
    return n_by * n_bx * (b * b) * k


def _gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = gray.astype(np.float64)
    # Đạo hàm trung tâm; biên replicate theo convolve2d mặc định
    gx = convolve2d(src, _GRAD_X, "replicate")
    gy = convolve2d(src, _GRAD_Y, "replicate")
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def _orientation_deg(gx: np.ndarray, gy: np.ndarray, *, signed: bool) -> np.ndarray:
    ang = np.degrees(np.arctan2(gy, gx))
    if signed:
        # [-180, 180)
        return np.mod(ang + 180.0, 360.0) - 180.0
    # Không dấu: [0, 180) như slide pedestrian 9 bin
    return np.mod(ang, 180.0)


def _accumulate_cell_histograms(
    mag: np.ndarray,
    orient_deg: np.ndarray,
    *,
    cell: int,
    bins: int,
    signed: bool,
) -> np.ndarray:
    h, w = mag.shape
    n_cy, n_cx = h // cell, w // cell
    hist = np.zeros((n_cy, n_cx, bins), dtype=np.float64)
    half = 180.0 if not signed else 360.0
    bin_w = half / float(bins)

    # Duyệt pixel để minh họa; ảnh test nhỏ nên vẫn nhanh
    for y in range(h):
        cy = y // cell
        if cy >= n_cy:
            continue
        for x in range(w):
            cx = x // cell
            if cx >= n_cx:
                continue
            m = mag[y, x]
            if m <= 0:
                continue
            o = float(orient_deg[y, x])
            if signed:
                o = o % 360.0
            else:
                o = o % 180.0
            # Vị trí liên tục trong trục bin để tách phiếu sang hai bin lân cận
            continuous = o / bin_w
            b0 = int(continuous) % bins
            b1 = (b0 + 1) % bins
            w1 = continuous - np.floor(continuous)
            w0 = 1.0 - w1
            hist[cy, cx, b0] += m * w0
            hist[cy, cx, b1] += m * w1
    return hist


def _normalize_block(vec: np.ndarray, norm: NormKind) -> np.ndarray:
    out = vec.astype(np.float64, copy=True)
    nrm = float(np.linalg.norm(out))
    if nrm <= _L2_EPS:
        return out
    out /= nrm
    if norm == "L2":
        return out
    # L2-Hys: clip rồi chuẩn hóa L2 lần hai (Dalal–Triggs)
    out = np.minimum(out, _HYS_CLIP)
    nrm2 = float(np.linalg.norm(out))
    if nrm2 > _L2_EPS:
        out /= nrm2
    return out


def hog_descriptor(
    gray: np.ndarray,
    *,
    cell: int = 8,
    block: int = 2,
    bins: int = 9,
    signed: bool = False,
    norm: NormKind = "L2-Hys",
) -> np.ndarray:
    """
    Vector HOG cho ảnh xám (Ch.6 — bước gradient → histogram ô → chuẩn hóa khối).

    Ảnh hằng số cho gradient 0 và toàn bộ descriptor 0 (đã ghi nhận).

    Parameters
    ----------
    gray :
        Ảnh uint8 H×W.
    cell :
        Ô lưới pixel (mặc định 8).
    block :
        Số ô theo mỗi chiều trong khối chuẩn hóa (mặc định 2 → 16×16 px).
    bins :
        Số bin hướng (9 không dấu 0°–180°).
    signed :
        False: không dấu; True: ±180°.
    norm :
        ``L2`` hoặc ``L2-Hys``.

    Returns
    -------
    np.ndarray
        Vector float64 1D.

    Example
    -------
    >>> img = np.zeros((128, 64), dtype=np.uint8)
    >>> hog_descriptor(img).shape[0]
    3780
    """
    img = require_gray_uint8(gray)
    c, b, k = _check_cell_block_bins(cell, block, bins)
    if norm not in ("L2", "L2-Hys"):
        raise ValueError("norm must be 'L2' or 'L2-Hys'")
    h, w = img.shape
    if h % c != 0 or w % c != 0:
        raise ValueError("image height and width must be multiples of cell")
    n_cy, n_cx = h // c, w // c
    if n_cy < b or n_cx < b:
        raise ValueError("image too small for given cell and block")

    gx, gy, mag = _gradients(img)
    orient = _orientation_deg(gx, gy, signed=signed)
    cell_hists = _accumulate_cell_histograms(mag, orient, cell=c, bins=k, signed=signed)

    n_by = n_cy - b + 1
    n_bx = n_cx - b + 1
    block_dim = b * b * k
    features: list[np.ndarray] = []

    # Khối trượt stride 1 ô (overlap)
    for by in range(n_by):
        for bx in range(n_bx):
            patch = cell_hists[by : by + b, bx : bx + b, :].reshape(block_dim)
            features.append(_normalize_block(patch, norm))

    return np.concatenate(features, axis=0)


def hog_visualize(
    gray: np.ndarray,
    *,
    cell: int = 8,
    bins: int = 9,
    signed: bool = False,
) -> np.ndarray:
    """
    Ảnh uint8 hiển thị hướng chủ đạo mỗi ô (0–255 tỉ lệ độ lớn histogram).

    Parameters
    ----------
    gray :
        Ảnh xám uint8.
    cell, bins, signed :
        Giống ``hog_descriptor`` (không dùng block).

    Returns
    -------
    np.ndarray
        uint8 cùng kích thước đầu vào.

    Example
    -------
    >>> g = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (32, 1))
    >>> v = hog_visualize(g, cell=8)
    >>> v.shape == g.shape
    True
    """
    img = require_gray_uint8(gray)
    c, _, k = _check_cell_block_bins(cell, 2, bins)
    h, w = img.shape
    if h % c != 0 or w % c != 0:
        raise ValueError("image height and width must be multiples of cell")

    gx, gy, mag = _gradients(img)
    orient = _orientation_deg(gx, gy, signed=signed)
    cell_hists = _accumulate_cell_histograms(mag, orient, cell=c, bins=k, signed=signed)

    n_cy, n_cx = h // c, w // c
    out = np.zeros((h, w), dtype=np.uint8)

    # Vẽ hướng mạnh nhất trong mỗi ô lên toàn bộ vùng ô
    for cy in range(n_cy):
        for cx in range(n_cx):
            strength = float(cell_hists[cy, cx].max())
            if strength <= 0:
                continue
            peak = int(np.argmax(cell_hists[cy, cx]))
            val = min(255, int(255.0 * strength / (strength + 1.0)))
            y0, y1 = cy * c, (cy + 1) * c
            x0, x1 = cx * c, (cx + 1) * c
            out[y0:y1, x0:x1] = np.uint8((peak * (255 // max(k - 1, 1)) + val) // 2)
    return out
