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
    """Kiểm tra kích thước cửa sổ lẻ, dương (theo slide bộ lọc).

    Cửa sổ lẻ (3, 5, …) giữ tâm kernel trùng pixel đầu ra — chủ đề spatial filtering.

    Ví dụ: ``size=3`` → hợp lệ (tâm ở vị trí 1); ``size=4`` → ``ValueError``.

    Returns
    -------
    int
        ``size`` đã ép kiểu, đảm bảo lẻ và ≥ 1.
    """
    # bool là subclass int trong Python — loại trừ để không nhầm True/False với 1/0.
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    # Ép về int thuần (numpy scalar → Python int) cho các phép chia lấy nguyên sau này.
    s = int(size)
    # Lẻ: có pixel trung tâm; dương: kernel không rỗng.
    if s < 1 or s % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")
    return s


def _check_sigma(sigma: float) -> float:
    """Kiểm tra σ Gaussian: hữu hạn và dương (Chương 2 — Gaussian filter).

    σ lớn → kernel rộng hơn (làm mịn mạnh); σ→0⁺ hẹp lý thuyết nhưng mẫu rời vẫn cần size lẻ.

    Returns
    -------
    float
        σ đã ép ``float``.
    """
    # Đưa mọi kiểu số (numpy scalar) về float Python để so sánh thống nhất.
    s = float(sigma)
    # σ≤0 hoặc nan/inf làm công thức exp không còn ý nghĩa lowpass.
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("sigma must be a finite positive number")
    return s


def _check_sharpen_c(c: float) -> float:
    """Xác nhận hệ số ``c`` trong ``g = f + c·(L*f)`` khớp quy ước dấu Laplace slide.

    Kernel tâm **âm** (hệ số -4 ở giữa) → ``c = -1``; đổi kernel tâm dương thì ``c = +1``.

    Returns
    -------
    float
        ``c`` hợp lệ (-1 hoặc +1).
    """
    cf = float(c)
    # Chỉ hai dấu slide cho phép — tránh làm sắc nét ngược pha nếu nhân sai dấu.
    if cf not in (-1.0, 1.0):
        raise ValueError("c must be -1 or +1 per Ch.2 Laplacian sign convention")
    return cf


def _uint8_from_float(out: np.ndarray) -> np.ndarray:
    """Đưa kết quả lọc float về ``uint8`` hiển thị được (spatial filter pipeline).

    Ví dụ: ``out=127.6`` → ``128``; ``out=300`` → ``255`` (clip).

    Returns
    -------
    np.ndarray
        Cùng shape, dtype ``uint8``.
    """
    # làm tròn gần nhất rồi kẹp trong dải động ảnh 8-bit trước khi đổi dtype.
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
    Với ``size=3``, mọi ô ≈ 1/9 (trung bình 9 láng giềng):

    ::
        [[1/9, 1/9, 1/9],
         [1/9, 1/9, 1/9],
         [1/9, 1/9, 1/9]]

    >>> k = box_kernel(3)
    >>> k.shape
    (3, 3)
    >>> np.isclose(k.sum(), 1.0)
    True
    """
    # Kernel hộp luôn vuông m×m — chủ đề lowpass / averaging filter.
    s = _odd_size(size, name="size")
    # Ma trận toàn 1: mọi láng giềng trong cửa sổ có trọng số như nhau.
    ker = np.ones((s, s), dtype=np.float64)
    # Chia đều để tổng trọng số = 1 (trung bình cộng trong cửa sổ, không làm sáng ảnh).
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
    # Bán kính tâm: với size=3 thì c=1 → tọa độ {-1,0,1}.
    c = s // 2
    # Tọa độ lưới tâm tại (0,0) để công thức khớp slide w∝exp(-(x²+y²)/(2σ²)).
    coords = np.arange(s, dtype=np.float64) - float(c)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    # Hàm Gaussian đẳng hướng: trọng số giảm theo khoảng cách bình phương tới tâm.
    ker = np.exp(-(xx * xx + yy * yy) / (2.0 * sig * sig))
    total = float(ker.sum())
    # Phòng trường hợp toàn số 0 (lỗi tham số cực đoan) — tránh chia cho 0.
    if total <= 0.0:
        raise ValueError("degenerate Gaussian kernel")
    # Chuẩn hóa tổng = 1: giữ năng lượng trung bình như lowpass tuyến tính.
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
    # Đầu vào phải uint8 — format ảnh lý thuyết sau lấy mẫu / lượng tử hóa.
    source = require_uint8_image(img)
    # Kernel hộp chuẩn hóa — hiện thực “trung bình cửa sổ” qua tích chập.
    ker = box_kernel(size)

    def _one(ch: np.ndarray) -> np.ndarray:
        # Tích chập tuyến tính (tương đương correlation với kernel đối xứng).
        filtered = convolve2d(ch, ker, _PADDING)
        # Kết quả conv là float — đổi lại uint8 để khớp ảnh gốc.
        return _uint8_from_float(filtered)

    # RGB: lọc độc lập từng kênh (không trộn màu trong không gian xám).
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
    # σ và size quyết định độ “mờ” — weighted average thay vì hộp đều.
    ker = gaussian_kernel(size, sigma)

    def _one(ch: np.ndarray) -> np.ndarray:
        # Gaussian lowpass: láng giềng gần tâm đóng góp nhiều hơn.
        filtered = convolve2d(ch, ker, _PADDING)
        return _uint8_from_float(filtered)

    return apply_per_channel(source, _one)


def _order_statistic_filter(
    img: np.ndarray,
    size: int,
    reducer,
) -> np.ndarray:
    """Lõi lọc phi tuyến: cửa sổ trượt trên ảnh đã pad viền replicate.

    Khác tích chập tuyến tính: ``reducer`` là median/min/max/… trên tập giá trị
    trong cửa sổ (order-statistic filtering — Chương 2).
    """
    s = _odd_size(size, name="size")
    # Bán kề rơi để pad đối xứng quanh ảnh gốc.
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

    # ``img`` đã được validate ở hàm gọi (median/min/max/…).
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

    # Dùng lõi cửa sổ trượt chung — thay reducer bằng median.
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
    # Min trong cửa sổ — tương tự erosion grayscale trong morphology (cục bộ).
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
    # Max trong cửa sổ — dual với min filter (dilation grayscale cục bộ).
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
        # Hai thống kê bậc: đáy và đỉnh cửa sổ.
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
    # Khớp dấu c với ma trận Laplace cố định tâm âm trong module.
    coeff = _check_sharpen_c(c)

    def _one(ch: np.ndarray) -> np.ndarray:
        # L*f: đạo hàm bậc hai xấp xỉ — phản ứng mạnh tại biên.
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
    # Dùng chung kiểm tra kernel với các hàm lọc khác.
    _odd_size(size, name="size")
    _check_sigma(sigma)
    am = float(amount)
    # amount vô hạn sẽ phá công thức g = f + amount·(f - f_blur).
    if not np.isfinite(am):
        raise ValueError("amount must be finite")
    # unsharp là high-boost với hệ số k = amount (mặc định amount=1).
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
    # k âm sẽ làm mờ thay vì sharp; nan/inf không có ý nghĩa vật lý.
    if not np.isfinite(kb) or kb < 0.0:
        raise ValueError("k must be a finite non-negative number")
    # f_blur từ cùng Gaussian như mean_filter weighted variant.
    ker = gaussian_kernel(size, sigma)

    def _one(ch: np.ndarray) -> np.ndarray:
        blur = convolve2d(ch, ker, _PADDING)
        # Cộng trong float tránh tràn uint8 khi cộng mask.
        f = ch.astype(np.float64)
        # Mask high-pass = chi tiết sắc (gốc trừ mờ).
        mask = f - blur.astype(np.float64)
        # g = f + k·mask: k>1 là high-boost, k=1 là unsharp masking slide.
        out = f + kb * mask
        return _uint8_from_float(out)

    return apply_per_channel(source, _one)
