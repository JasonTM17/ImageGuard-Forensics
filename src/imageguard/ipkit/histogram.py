"""Xử lý histogram — Chương 2, mục 3 "Histogram processing"
(bài giảng Image Processing, giảng viên dunghv).

Histogram đếm số pixel theo từng mức xám; từ đó ta thực hiện cân bằng
(equalization), đặc tả (specification / matching) và xử lý cục bộ (local) để
tăng tương phản. Toàn bộ cài đặt thuần NumPy, chỉ dùng ảnh xám uint8.
"""

from __future__ import annotations

import numpy as np

from imageguard.ipkit._util import require_gray_uint8


def histogram(gray: np.ndarray) -> np.ndarray:
    """Đếm số pixel theo 256 mức xám: ``h(r_k) = n_k``.

    Theo định nghĩa ở slide "Histogram Processing" (Chương 2, mục 3): ``r_k`` là
    mức xám thứ k, ``n_k`` là số pixel có mức xám đó.

    Args:
        gray: Ảnh xám uint8 2D.

    Returns:
        Mảng ``int64`` 256 phần tử; phần tử ``k`` là số pixel có mức xám ``k``.

    Example:
        >>> histogram(np.array([[0, 0, 255]], dtype=np.uint8))[0]
        2
    """
    img = require_gray_uint8(gray)
    # bincount đếm số lần xuất hiện của từng giá trị; minlength đảm bảo đủ 256 bin.
    return np.bincount(img.ravel(), minlength=256).astype(np.int64)


def normalized_histogram(gray: np.ndarray) -> np.ndarray:
    """Histogram chuẩn hóa: ``p(r_k) = n_k / tổng số pixel``.

    Là xác suất ước lượng một pixel có mức xám ``r_k`` (slide "Histogram
    Processing", Chương 2, mục 3); tổng các phần tử bằng 1.

    Args:
        gray: Ảnh xám uint8 2D.

    Returns:
        Mảng ``float64`` 256 phần tử nằm trong [0, 1].
    """
    hist = histogram(gray)
    return hist / float(hist.sum())


def cdf(gray: np.ndarray) -> np.ndarray:
    """Hàm phân bố tích lũy chuẩn hóa: ``cdf_k = Σ_{j ≤ k} p(r_j)``.

    Tổng chạy của histogram chuẩn hóa, luôn đơn điệu tăng và kết thúc tại 1.0;
    đây là cơ sở cho phép cân bằng histogram ở slide "Histogram Equalization"
    (Chương 2, mục 3).

    Args:
        gray: Ảnh xám uint8 2D.

    Returns:
        Mảng ``float64`` 256 phần tử trong [0, 1], phần tử cuối bằng 1.0.
    """
    return np.cumsum(normalized_histogram(gray), dtype=np.float64)


def _equalize_lut(hist: np.ndarray) -> np.ndarray:
    """Tính bảng tra cân bằng từ một histogram (số đếm) 256 bin."""
    total = float(hist.sum())
    if total <= 0:
        raise ValueError("histogram must contain at least one pixel")
    # CDF chuẩn hóa về [0, 255] rồi làm tròn (nửa về số chẵn) như yêu cầu của bài giảng.
    cdf = np.cumsum(hist.astype(np.float64))
    return np.rint(cdf * (255.0 / total)).astype(np.uint8)


def equalize(gray: np.ndarray) -> np.ndarray:
    """Cân bằng histogram: ``s_k = round((L - 1) * cdf_k)`` với ``L = 256``.

    Theo slide "Histogram Equalization" (Chương 2, mục 3): ánh xạ mỗi mức xám về
    vị trí của CDF chuẩn hóa trên thang [0, L-1], giúp trải đều histogram và tăng
    tương phản toàn cục. Ảnh hằng số (chỉ một mức xám) được giữ nguyên vì không
    có tương phản để cân bằng.

    Args:
        gray: Ảnh xám uint8 2D.

    Returns:
        Ảnh xám uint8 cùng kích thước sau cân bằng.

    Example:
        >>> equalize(np.array([[0, 255]], dtype=np.uint8))
        array([[128, 255]], dtype=uint8)
    """
    img = require_gray_uint8(gray)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.int64)
    # Ảnh hằng số chỉ có một bin khác 0 -> giữ nguyên, tránh ánh xạ mọi pixel về 255.
    if int(np.count_nonzero(hist)) <= 1:
        return img.copy()
    lut = _equalize_lut(hist)
    return lut[img]


def _as_histogram(reference: object) -> np.ndarray:
    """Nhận ảnh xám uint8 hoặc mảng 256 bin số đếm; trả về histogram int64."""
    ref = np.asarray(reference)
    if ref.ndim == 2:
        # Tham chiếu là một ảnh xám: đếm histogram của chính nó.
        ref_img = require_gray_uint8(reference)
        return np.bincount(ref_img.ravel(), minlength=256).astype(np.int64)
    if ref.ndim == 1 and ref.shape[0] == 256:
        if ref.dtype.kind not in "iuf":
            raise ValueError("reference histogram must be numeric")
        vals = ref.astype(np.float64)
        if not np.all(np.isfinite(vals)) or np.any(vals < 0.0):
            raise ValueError("reference histogram must be non-negative finite counts")
        hist = np.rint(vals).astype(np.int64)
        if hist.sum() <= 0:
            raise ValueError("reference histogram must contain at least one pixel")
        return hist
    raise ValueError(
        "reference must be a 2D uint8 gray image or a length-256 histogram"
    )


def match_histogram(gray: np.ndarray, reference: object) -> np.ndarray:
    """Đặc tả histogram (matching/specification) bằng cách khớp CDF.

    Theo 4 bước ở slide "Histogram Matching (Specification)" (Chương 2, mục 3):
    (1) cân bằng ảnh vào được ``s_k = T(r_k)``; (2) cân bằng histogram đích được
    ``G(z_q)``; (3) với mỗi ``s_k`` tìm ``z_q`` sao cho ``G(z_q)`` gần ``s_k``
    nhất (bằng nhau thì lấy ``z`` nhỏ nhất); (4) ánh xạ mỗi pixel ``r -> s -> z``.

    Args:
        gray: Ảnh xám uint8 2D cần biến đổi.
        reference: Ảnh xám uint8 2D (lấy histogram của nó làm đích) hoặc mảng
            256 phần tử số đếm biểu diễn histogram đích.

    Returns:
        Ảnh xám uint8 cùng kích thước, có histogram xấp xỉ histogram đích.

    Example:
        >>> match_histogram(np.array([[0, 128, 255]], dtype=np.uint8),
        ...                 np.array([[100, 100, 150]], dtype=np.uint8))
        array([[  0, 100, 150]], dtype=uint8)
    """
    img = require_gray_uint8(gray)
    ref_hist = _as_histogram(reference)

    hist = np.bincount(img.ravel(), minlength=256).astype(np.int64)
    # Bước 1 & 2: biến đổi cân bằng T(r) cho ảnh vào và G(z) cho histogram đích.
    t_in = _equalize_lut(hist).astype(np.int64)
    g_ref = _equalize_lut(ref_hist).astype(np.int64)

    # Bước 3 & 4: với mỗi mức r, chọn z có G(z) gần T(r) nhất (hòa thì lấy z nhỏ).
    # np.argmin trả về chỉ số đầu tiên khi có nhiều cực tiểu bằng nhau -> khớp yêu cầu.
    lut = np.array([int(np.argmin(np.abs(g_ref - s))) for s in t_in], dtype=np.uint8)
    return lut[img]


def _check_tile(tile: int) -> int:
    """Kiểm tra kích thước ô (tile) là số nguyên dương."""
    if isinstance(tile, bool) or not isinstance(tile, (int, np.integer)):
        raise TypeError("tile must be an integer")
    t = int(tile)
    if t < 1:
        raise ValueError("tile must be a positive integer")
    return t


def _check_clip_limit(clip_limit: float | None) -> float | None:
    """Kiểm tra giới hạn cắt: None hoặc số trong (0, 1]."""
    if clip_limit is None:
        return None
    cl = float(clip_limit)
    if not np.isfinite(cl) or not (0.0 < cl <= 1.0):
        raise ValueError("clip_limit must be in (0, 1]")
    return cl


def _clip_histogram(hist: np.ndarray, clip_limit: float) -> np.ndarray:
    """Giới hạn độ cao mỗi bin rồi phân phối lại phần dư (giới hạn tương phản).

    Ý tưởng của giới hạn clip (dùng trong CLAHE): không cho một bin nào chiếm quá
    ``clip_limit`` phần của tổng pixel trong ô, để tránh khuếch đại nhiễu ở vùng
    phẳng. Phần dư bị cắt được chia đều lại cho 256 bin nên tổng số pixel không đổi.
    """
    total = float(hist.sum())
    # Ngưỡng cắt tính theo tỉ lệ của tổng số pixel trong ô (ít nhất 1 pixel).
    clip = max(1.0, clip_limit * total)
    out = hist.astype(np.float64).copy()
    excess = 0.0
    for k in range(256):
        if out[k] > clip:
            excess += out[k] - clip
            out[k] = clip
    out += excess / 256.0
    return out


def local_equalize(
    gray: np.ndarray, tile: int, clip_limit: float | None = None
) -> np.ndarray:
    """Cân bằng histogram cục bộ theo từng ô vuông (tile) không chồng lấn.

    Theo slide "Local Histogram Processing" (Chương 2, mục 3): thay vì dùng CDF
    của cả ảnh, ta chia ảnh thành các ô và cân bằng riêng từng ô để làm nổi chi
    tiết trong vùng nhỏ. Cách này dùng ô không chồng lấn nên nhanh nhưng tạo
    "hiệu ứng khối" (blocky) ở ranh giới các ô — bài giảng cũng ghi nhận điều này.

    Nếu ``clip_limit`` được cho, mỗi ô sẽ được giới hạn tương phản (xem
    :func:`_clip_histogram`) trước khi cân bằng. Không nội suy giữa các ô.

    Args:
        gray: Ảnh xám uint8 2D.
        tile: Kích thước cạnh ô vuông (số nguyên dương). Ô cuối ở biên có thể
            nhỏ hơn nếu kích thước ảnh không chia hết cho ``tile``.
        clip_limit: Giới hạn cắt tùy chọn trong (0, 1], hoặc ``None`` (mặc định).

    Returns:
        Ảnh xám uint8 cùng kích thước.

    Example:
        >>> local_equalize(np.zeros((4, 4), dtype=np.uint8), 2)
        array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]], dtype=uint8)
    """
    img = require_gray_uint8(gray)
    t = _check_tile(tile)
    cl = _check_clip_limit(clip_limit)
    h, w = img.shape
    out = np.empty_like(img)
    # Lướt từng ô không chồng lấn; ô cuối ở biên có thể nhỏ hơn nếu không chia hết.
    for y in range(0, h, t):
        for x in range(0, w, t):
            block = img[y : y + t, x : x + t]
            hist = np.bincount(block.ravel(), minlength=256).astype(np.int64)
            # Ô hằng số thì giữ nguyên, tránh khuếch đại nhiễu ở vùng phẳng.
            if int(np.count_nonzero(hist)) <= 1:
                out[y : y + t, x : x + t] = block
                continue
            if cl is not None:
                hist = _clip_histogram(hist, cl)
            lut = _equalize_lut(hist)
            out[y : y + t, x : x + t] = lut[block]
    return out
