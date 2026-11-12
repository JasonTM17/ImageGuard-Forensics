"""Ngưỡng hóa toàn cục và cục bộ (chỉ NumPy).

Chủ đề C5 — phân đoạn: Otsu (phương sai liên lớp) và ngưỡng thích nghi theo mean
cửa sổ qua bảng tổng tích (summed-area table).
"""

from __future__ import annotations

import numpy as np

from imageguard.ipkit._util import require_gray_uint8


def _odd_block(block: int) -> int:
    """Kích thước cửa sổ adaptive phải lẻ và ≥ 3 để có tâm rõ ràng."""
    if isinstance(block, bool) or not isinstance(block, (int, np.integer)):
        raise TypeError("block must be an integer")
    b = int(block)
    if b < 3 or b % 2 == 0:
        raise ValueError("block must be an odd integer >= 3")
    return b


def otsu_threshold(gray: np.ndarray) -> int:
    """Ngưỡng Otsu tối ưu cho ảnh xám 8 bit.

    Ý tưởng: với ngưỡng t, chia pixel thành lớp nền (≤t) và đối tượng (>t).
    Tối đa hóa phương sai *giữa hai lớp* σ²_b (công thức dùng xác suất ω, moment
    tích lũy μ; mẫu ω(1−ω)). ``argmax`` trên 256 mức gray cho ngưỡng nguyên [0,255].

    Tham số: ``gray`` xám uint8.

    Trả về: ngưỡng t (int). Ảnh rỗng hoặc histogram suy biến có xử lý đặc biệt trong mã.

    Liên quan C5 — phân đoạn histogram / ngưỡng tự động.
    """
    img = require_gray_uint8(gray)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = float(img.size)
    if total == 0:
        raise ValueError("empty image")
    if hist.sum() <= 0:
        return 0

    probs = hist / total
    bins = np.arange(256, dtype=np.float64)
    # ω(t): tích lũy xác suất lớp nền; μ(t): moment cường độ tích lũy
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
    """Ngưỡng cục bộ: pixel > (mean_cửa_sổ − c) → 255, ngược lại 0.

    Mean trong cửa sổ block×block tính bằng bảng tổng tích 2D (SAT): tổng
    hình chữ nhật = tổng bốn góc, tránh lặp O(block²) từng pixel.

    ``c`` hạ ngưỡng để tránh chia quá sáng (tương tự OpenCV adaptiveMean).

    Trả về: ảnh nhị phân uint8.

    Liên quan C5 — phân đoạn khi độ sáng nền thay đổi; C2 liên quan cửa sổ cục bộ.
    """
    img = require_gray_uint8(gray)
    b = _odd_block(block)
    offset = float(c)
    if not np.isfinite(offset):
        raise ValueError("c must be finite")

    h, w = img.shape
    r = b // 2
    src = img.astype(np.float64)
    padded = np.pad(src, r, mode="edge")
    area = float(b * b)

    # Tích phân 2D: sat(y,x) = tổng vùng từ gốc đến (y-1,x-1) trên ảnh đệm
    sat = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    sat = np.pad(sat, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)

    yy = np.arange(h, dtype=np.intp)[:, np.newaxis]
    xx = np.arange(w, dtype=np.intp)[np.newaxis, :]
    y2 = yy + b
    x2 = xx + b
    win_sum = sat[y2, x2] - sat[yy, x2] - sat[y2, xx] + sat[yy, xx]
    mean = win_sum / area
    return np.where(src > mean - offset, 255, 0).astype(np.uint8)
