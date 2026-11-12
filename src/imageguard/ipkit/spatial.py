"""Bộ lọc không gian (spatial domain) — Chương 2, HCMUTE (dunghv).

Phần 4–6: lọc tuyến tính (hộp, Gaussian), lọc thống kê bậc (trung vị,
min, max, midpoint), làm sắc nét Laplace, unsharp masking và high-boost.
Triển khai NumPy thuần; tích chập qua ``imageguard.ipkit.edges.convolve2d``.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from imageguard.ipkit._util import apply_per_channel, require_uint8_image
from imageguard.ipkit.edges import convolve2d

# Ma trận Laplace 4-lân cận, tâm âm — cùng định nghĩa Chương 2 / edges.laplacian.
_LAPLACIAN_NEG_CENTER = np.array(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

_PADDING = "replicate"


def _odd_size(size: int, *, name: str = "size") -> int:
    """Kiểm tra kích thước cửa sổ lẻ, dương (theo slide bộ lọc)."""
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    s = int(size)
    if s < 1 or s % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")
    return s


def _check_sigma(sigma: float) -> float:
    s = float(sigma)
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("sigma must be a finite positive number")
    return s


def _check_sharpen_c(c: float) -> float:
    cf = float(c)
    if cf not in (-1.0, 1.0):
        raise ValueError("c must be -1 or +1 per Ch.2 Laplacian sign convention")
    return cf


def _uint8_from_float(out: np.ndarray) -> np.ndarray:
    # Ép về [0,255] trước khi xuất uint8 như ảnh gốc.
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def box_kernel(size: int) -> np.ndarray:
    """Tạo kernel hộp (box) m×m, tất cả hệ số bằng nhau và tổng bằng 1.

    Chương 2 — Smoothing: lowpass khi mọi hệ số trong mặt nạ đều bằng nhau.

    Parameters
    ----------
    size:
        Cạnh kernel lẻ (3, 5, ...).

    Returns
    -------
    np.ndarray
        Ma trận ``float64`` kích thước ``(size, size)``, chuẩn hóa ``sum == 1``.

    Examples
    --------
    >>> k = box_kernel(3)
    >>> k.shape
    (3, 3)
    >>> np.isclose(k.sum(), 1.0)
    True
    """
    s = _odd_size(size, name="size")
    ker = np.ones((s, s), dtype=np.float64)
    # Chia đều để tổng trọng số = 1 (trung bình cộng trong cửa sổ).
    ker /= float(s * s)
    return ker


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Kernel Gaussian 2D lấy mẫu rời, chuẩn hóa tổng bằng 1.

    Chương 2 — Gaussian filter: ``w(x,y) ∝ exp(-(x²+y²)/(2σ²))``.

    Parameters
    ----------
    size:
        Cạnh kernel lẻ.
    sigma:
        Độ lệch chuẩn (σ) theo slide.

    Returns
    -------
    np.ndarray
        Kernel ``float64`` kích thước ``(size, size)``.

    Examples
    --------
    >>> g = gaussian_kernel(3, sigma=1.0)
    >>> np.isclose(g.sum(), 1.0)
    True
    """
    s = _odd_size(size, name="size")
    sig = _check_sigma(sigma)
    c = s // 2
    # Tọa độ lưới tâm tại (0,0) để công thức khớp slide.
    coords = np.arange(s, dtype=np.float64) - float(c)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    # Hàm Gaussian isotropic: trọng số giảm theo khoảng cách bình phương tới tâm.
    ker = np.exp(-(xx * xx + yy * yy) / (2.0 * sig * sig))
    total = float(ker.sum())
    if total <= 0.0:
        raise ValueError("degenerate Gaussian kernel")
    ker /= total
    return ker


def mean_filter(img: np.ndarray, size: int) -> np.ndarray:
    """Lọc trung bình (box / averaging lowpass) trên ảnh uint8.

    Chương 2 — thay mỗi pixel bằng trung bình cường độ trong lân cận m×m;
    tích chập với ``box_kernel``, viền ``replicate``.

    Parameters
    ----------
    img:
        Ảnh xám hoặc H×W×3, ``uint8``.
    size:
        Cạnh cửa sổ lẻ.

    Returns
    -------
    np.ndarray
        Ảnh đã làm mịn, cùng shape và ``uint8``.

    Examples
    --------
    >>> import numpy as np
    >>> out = mean_filter(np.full((5, 5), 100, np.uint8), 3)
    >>> int(out[2, 2])
    100
    """
    source = require_uint8_image(img)
    ker = box_kernel(size)

    def _one(ch: np.ndarray) -> np.ndarray:
        # Tích chập tuyến tính (tương đương correlation với kernel đối xứng).
        filtered = convolve2d(ch, ker, _PADDING)
        return _uint8_from_float(filtered)

    return apply_per_channel(source, _one)


def gaussian_filter(img: np.ndarray, size: int, sigma: float) -> np.ndarray:
    """Lọc Gaussian lowpass; viền ``replicate`` như các bộ lọc tuyến tính khác.

    Chương 2 — làm mịn bằng kernel Gaussian m×m với độ lệch chuẩn σ.

    Parameters
    ----------
    img:
        Ảnh xám hoặc RGB ``uint8``.
    size:
        Cạnh kernel lẻ.
    sigma:
        Tham số σ của Gaussian.

    Returns
    -------
    np.ndarray
        Ảnh đã lọc, ``uint8``.

    Examples
    --------
    >>> import numpy as np
    >>> g = gaussian_filter(np.zeros((4, 4), np.uint8), 3, 1.0)
    >>> g.dtype == np.uint8
    True
    """
    source = require_uint8_image(img)
    ker = gaussian_kernel(size, sigma)

    def _one(ch: np.ndarray) -> np.ndarray:
        filtered = convolve2d(ch, ker, _PADDING)
        return _uint8_from_float(filtered)

    return apply_per_channel(source, _one)


def _order_statistic_filter(
    img: np.ndarray,
    size: int,
    reducer,
) -> np.ndarray:
    """Lõi lọc phi tuyến: cửa sổ trượt trên ảnh đã pad viền replicate."""
    s = _odd_size(size, name="size")
    r = s // 2

    def _one(ch: np.ndarray) -> np.ndarray:
        src = ch.astype(np.float64)
        # Pad viền để cửa sổ luôn đủ pixel (giống xử lý biên replicate).
        padded = np.pad(src, r, mode="edge")
        # Mỗi vị trí (i,j) nhìn thấy khối s×s liền kề — không cần vòng lặp Python.
        windows = sliding_window_view(padded, (s, s))
        # Giảm theo trục cửa sổ để ra một giá trị đầu ra / pixel.
        reduced = reducer(windows)
        return _uint8_from_float(reduced)

    return apply_per_channel(img, _one)


def median_filter(img: np.ndarray, size: int) -> np.ndarray:
    """Lọc trung vị: thay pixel tâm bằng median trong cửa sổ (Chương 2, order-statistic).

    Parameters
    ----------
    img:
        Ảnh ``uint8`` xám hoặc RGB.
    size:
        Cạnh cửa sổ lẻ.

    Returns
    -------
    np.ndarray
        Ảnh đã lọc; hiệu quả loại nhiễu muối tiêu khi ``size`` đủ lớn.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[0, 255, 0]], dtype=np.uint8)
    >>> y = median_filter(x, 3)
    >>> y.dtype == np.uint8
    True
    """
    source = require_uint8_image(img)

    def _median(windows: np.ndarray) -> np.ndarray:
        # Median = phần tử 50% sau khi xếp hạng — đặc trưng của lọc thống kê bậc.
        return np.median(windows, axis=(-2, -1))

    return _order_statistic_filter(source, size, _median)


def min_filter(img: np.ndarray, size: int) -> np.ndarray:
    """Lọc min (erosion grayscale trong cửa sổ) — Chương 2, order-statistic.

    Parameters
    ----------
    img:
        Ảnh ``uint8``.
    size:
        Cửa sổ lẻ.

    Returns
    -------
    np.ndarray
        Ảnh với mỗi pixel bằng giá trị nhỏ nhất trong lân cận.

    Examples
    --------
    >>> import numpy as np
    >>> m = min_filter(np.array([[10, 50]], dtype=np.uint8), 3)
    >>> m.shape == (1, 2)
    True
    """
    source = require_uint8_image(img)
    return _order_statistic_filter(source, size, lambda w: np.min(w, axis=(-2, -1)))


def max_filter(img: np.ndarray, size: int) -> np.ndarray:
    """Lọc max trong cửa sổ — Chương 2, order-statistic.

    Parameters
    ----------
    img:
        Ảnh ``uint8``.
    size:
        Cửa sổ lẻ.

    Returns
    -------
    np.ndarray
        Pixel đầu ra là max của lân cận.

    Examples
    --------
    >>> import numpy as np
    >>> m = max_filter(np.array([[10, 50]], dtype=np.uint8), 3)
    >>> m.shape == (1, 2)
    True
    """
    source = require_uint8_image(img)
    return _order_statistic_filter(source, size, lambda w: np.max(w, axis=(-2, -1)))


def midpoint_filter(img: np.ndarray, size: int) -> np.ndarray:
    """Lọc midpoint: (min + max) / 2 trong cửa sổ — Chương 2, order-statistic.

    Parameters
    ----------
    img:
        Ảnh ``uint8``.
    size:
        Cửa sổ lẻ.

    Returns
    -------
    np.ndarray
        Ảnh đã lọc.

    Examples
    --------
    >>> import numpy as np
    >>> z = midpoint_filter(np.full((3, 3), 40, np.uint8), 3)
    >>> int(z[1, 1]) == 40
    True
    """
    source = require_uint8_image(img)

    def _mid(windows: np.ndarray) -> np.ndarray:
        lo = np.min(windows, axis=(-2, -1))
        hi = np.max(windows, axis=(-2, -1))
        # Trung điểm min–max: bộ lọc thống kê bậc khác median.
        return 0.5 * (lo + hi)

    return _order_statistic_filter(source, size, _mid)


def laplacian_sharpen(img: np.ndarray, c: float = -1.0) -> np.ndarray:
    """Làm sắc nét bằng Laplace: ``g = f + c · (L * f)`` với ma trận tâm âm.

    Chương 2 — Second derivatives / Laplacian:
    với kernel Laplace tâm **âm** (hệ số -4 ở tâm, như slide và ``edges.laplacian``),
    slide quy ước ``c = -1``; kernel tâm dương thì ``c = +1``.
    Hàm này dùng kernel tâm âm, mặc định ``c=-1``.

    Parameters
    ----------
    img:
        Ảnh ``uint8``.
    c:
        ``-1`` (kernel tâm âm) hoặc ``+1`` (kernel tâm dương nếu đổi ma trận).

    Returns
    -------
    np.ndarray
        Ảnh làm sắc nét, ``uint8``.

    Examples
    --------
    >>> import numpy as np
    >>> edge = laplacian_sharpen(
    ...     np.tile(np.array([0, 255], dtype=np.uint8), (1, 8)), c=-1.0
    ... )
    >>> edge.dtype == np.uint8
    True
    """
    source = require_uint8_image(img)
    coeff = _check_sharpen_c(c)

    def _one(ch: np.ndarray) -> np.ndarray:
        lap = convolve2d(ch, _LAPLACIAN_NEG_CENTER, _PADDING)
        # Cộng thêm Laplace (c có dấu) để nhấn biên, vùng phẳng gần như giữ nguyên.
        sharp = ch.astype(np.float64) + coeff * lap.astype(np.float64)
        return _uint8_from_float(sharp)

    return apply_per_channel(source, _one)


def unsharp_mask(
    img: np.ndarray,
    size: int,
    sigma: float,
    amount: float = 1.0,
) -> np.ndarray:
    """Unsharp masking: ``g = f + amount · (f - f_blur)`` (Chương 2).

    Mask = hiệu giữa ảnh gốc và ảnh làm mờ Gaussian; ``amount=1`` khớp slide
    ``g = f + mask``.

    Parameters
    ----------
    img:
        Ảnh ``uint8``.
    size, sigma:
        Tham số lọc Gaussian làm ``f'``.
    amount:
        Hệ số nhân mask (mặc định 1).

    Returns
    -------
    np.ndarray
        Ảnh làm sắc nét.

    Examples
    --------
    >>> import numpy as np
    >>> u = unsharp_mask(np.full((3, 3), 128, np.uint8), 3, 1.0)
    >>> np.all(u == 128)
    True
    """
    source = require_uint8_image(img)
    _odd_size(size, name="size")
    _check_sigma(sigma)
    am = float(amount)
    if not np.isfinite(am):
        raise ValueError("amount must be finite")
    return highboost(source, 1.0 * am, size, sigma)


def highboost(
    img: np.ndarray,
    k: float,
    size: int,
    sigma: float,
) -> np.ndarray:
    """High-boost / unsharp: ``g = f + k · (f - f_blur)``; ``k=1`` là unsharp (Chương 2).

    Parameters
    ----------
    img:
        Ảnh ``uint8``.
    k:
        Hệ số boost; ``k=1`` unsharp masking, ``k>1`` high-boost.
    size, sigma:
        Gaussian blur tạo ``f'``.

    Returns
    -------
    np.ndarray
        Ảnh đầu ra ``uint8``.

    Examples
    --------
    >>> import numpy as np
    >>> h = highboost(np.full((2, 2), 200, np.uint8), 2.0, 3, 1.0)
    >>> h.shape == (2, 2)
    True
    """
    source = require_uint8_image(img)
    _odd_size(size, name="size")
    _check_sigma(sigma)
    kb = float(k)
    if not np.isfinite(kb) or kb < 0.0:
        raise ValueError("k must be a finite non-negative number")
    ker = gaussian_kernel(size, sigma)

    def _one(ch: np.ndarray) -> np.ndarray:
        blur = convolve2d(ch, ker, _PADDING)
        f = ch.astype(np.float64)
        # Mask high-pass = chi tiết sắc (gốc trừ mờ).
        mask = f - blur.astype(np.float64)
        out = f + kb * mask
        return _uint8_from_float(out)

    return apply_per_channel(source, _one)
