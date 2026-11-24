"""Lấy mẫu ảnh theo tọa độ liên tục — triển khai thuần NumPy.

Hai bộ lấy mẫu trong module này là nền tảng cho mọi phép biến đổi hình học của
``imageguard.ipkit.geometry``.  Tọa độ đầu vào gồm ``rows`` (hàng/y) và ``cols``
(cột/x), là các mảng float cùng shape với lưới đầu ra.  Quy ước tọa độ: **x là
cột, y là hàng**; ngoài biên ảnh trả về 0 (viền hằng), và cùng một cặp tọa độ
được áp dụng cho mọi kênh của ảnh màu H×W×3.
"""

from __future__ import annotations

import numpy as np


def _require_image(image: np.ndarray) -> np.ndarray:
    """Kiểm tra ảnh đầu vào: uint8 xám H×W hoặc màu H×W×3, không rỗng.

    Tham số
    -------
    image : ảnh cần kiểm tra.

    Trả về
    ------
    Mảng đã ép ``asarray``, dtype uint8, đúng 2 hoặc 3 chiều.
    """
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError("image must be 2D gray or HxWx3 color")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError("image must have exactly 3 channels when 3D")
    if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.size == 0:
        raise ValueError("image must be non-empty")
    if arr.dtype != np.uint8:
        raise ValueError("image must be uint8")
    return arr


def _require_coords(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chuẩn hóa tọa độ lấy mẫu về float64, cùng shape.

    Tham số
    -------
    rows, cols : tọa độ hàng/cột (có thể là scalar hoặc mảng).

    Trả về
    ------
    Bộ ``(rows, cols)`` đã ép ``float64``; tọa độ không hữu hạn bị từ chối.
    """
    r = np.asarray(rows, dtype=np.float64)
    c = np.asarray(cols, dtype=np.float64)
    if r.shape != c.shape:
        raise ValueError("rows and cols must have the same shape")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(c)):
        raise ValueError("coordinates must be finite")
    return r, c


def _gather(src: np.ndarray, row: np.ndarray, col: np.ndarray):
    """Trích mẫu tại ``(row, col)``; ô ngoài biên lấy giá trị 0.

    Chỉ số âm/vượt biên được kẹp để indexing an toàn rồi triệt tiêu bằng mask,
    nên biên có giá trị hằng 0 (không lặp mép, không quay vòng).
    """
    h, w = src.shape[:2]
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    sampled = src[np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)]
    if src.ndim == 3:
        # Mỗi ô ảnh có 3 kênh — mask cần thêm trục kênh để broadcast.
        inside = inside[..., np.newaxis]
    return np.where(inside, sampled, 0)


def sample_nearest(image: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Lấy mẫu gần nhất: chỉ số = ``floor(coord + 0.5)``, ngoài biên trả 0.

    Tham số
    -------
    image : ảnh uint8 H×W hoặc H×W×3.
    rows, cols : tọa độ hàng/cột dạng float, cùng shape với lưới đầu ra.

    Trả về
    ------
    Ảnh uint8 có shape ``rows.shape`` (hoặc ``rows.shape + (3,)`` với ảnh màu).

    Ví dụ
    -----
    Với ảnh 2×2 ``[[0, 100], [50, 200]]``, tại ``row=0.2, col=0.6`` trả 100.
    """
    src = _require_image(image)
    r, c = _require_coords(rows, cols)
    h, w = src.shape[:2]
    # Kẹp tọa độ về [-1, h]×[-1, w] trước khi ép int: tọa độ khổng lồ (vd 1e300)
    # không làm tràn phép ép kiểu; ``_gather`` vẫn dùng mask để trả 0 khi ngoài biên.
    row = np.floor(np.clip(r + 0.5, -1.0, h)).astype(np.intp)
    col = np.floor(np.clip(c + 0.5, -1.0, w)).astype(np.intp)
    return _gather(src, row, col).astype(np.uint8, copy=False)


def sample_bilinear(image: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Lấy mẫu song tuyến từ bốn láng giềng; ô ngoài biên đóng góp 0.

    Trọng số ``(1-fx)(1-fy)``, ``fx(1-fy)``, ``(1-fx)fy``, ``fx·fy`` được áp
    dụng cho bốn ô ``(y0,x0)``, ``(y0,x0+1)``, ``(y0+1,x0)``, ``(y0+1,x0+1)``.
    Một láng giềng nằm ngoài ảnh có giá trị 0 nhưng vẫn giữ nguyên trọng số,
    nên kết quả tự động mờ dần về biên.  Cuối cùng làm tròn bằng ``rint`` rồi
    kẹp về đoạn 0..255 trước khi ép uint8.

    Tham số
    -------
    image : ảnh uint8 H×W hoặc H×W×3.
    rows, cols : tọa độ hàng/cột dạng float, cùng shape với lưới đầu ra.

    Trả về
    ------
    Ảnh uint8 có shape ``rows.shape`` (hoặc ``rows.shape + (3,)`` với ảnh màu).

    Ví dụ
    -----
    Với ảnh 2×2 ``[[0, 100], [50, 200]]``, tại ``row=0.5, col=0.5`` trả 88.
    """
    src = _require_image(image)
    r, c = _require_coords(rows, cols)
    h, w = src.shape[:2]
    # Kẹp về [-1, h]×[-1, w] trước khi ép int để tọa độ khổng lồ không tràn;
    # giá trị trong biên và lân cận biên không đổi (mask trong ``_gather`` lo phần ngoài).
    r = np.clip(r, -1.0, h)
    c = np.clip(c, -1.0, w)

    y0 = np.floor(r).astype(np.intp)
    x0 = np.floor(c).astype(np.intp)
    fy = r - y0
    fx = c - x0

    if src.ndim == 3:
        # Thêm trục kênh để trọng số broadcast đúng trên H×W×3.
        fy = fy[..., np.newaxis]
        fx = fx[..., np.newaxis]

    v00 = _gather(src, y0, x0).astype(np.float64)
    v01 = _gather(src, y0, x0 + 1).astype(np.float64)
    v10 = _gather(src, y0 + 1, x0).astype(np.float64)
    v11 = _gather(src, y0 + 1, x0 + 1).astype(np.float64)

    # Tính trong float64 để không tràn uint8 khi cộng các thành phần trọng số.
    out = (
        (1.0 - fx) * (1.0 - fy) * v00
        + fx * (1.0 - fy) * v01
        + (1.0 - fx) * fy * v10
        + fx * fy * v11
    )
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
