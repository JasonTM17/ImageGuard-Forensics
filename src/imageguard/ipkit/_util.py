"""Tiện ích kiểm tra đầu vào cho ipkit (toán tử xem/demo trong môn Xử lý ảnh).

Các hàm ở đây không làm toán trên ảnh mà đảm bảo dữ liệu đúng định dạng uint8
trước khi gọi lọc không gian, nhiễu, ngưỡng hoặc hình thái học.
"""

from __future__ import annotations

import numpy as np

_UINT8 = (np.uint8,)


def require_uint8_image(img: np.ndarray, *, name: str = "img") -> np.ndarray:
    """Ép kiểm tra ảnh uint8: xám 2D hoặc màu H×W×3.

    Tham số
    -------
    img : mảng NumPy đầu vào.
    name : tên tham số để hiển thị trong thông báo lỗi.

    Trả về
    ------
    Cùng tham chiếu sau ``np.asarray`` (không ép kiểu nếu đã đúng uint8).

    Liên quan bài giảng: tiền đề cho mọi chương (C2–C5) — ảnh số 8 bit/kênh.
    """
    arr = np.asarray(img)
    if arr.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2D gray or H×W×3")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError(f"{name} must have 3 channels when 3D")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if arr.dtype not in _UINT8:
        raise ValueError(f"{name} must be uint8")
    return arr


def require_gray_uint8(gray: np.ndarray, *, name: str = "gray") -> np.ndarray:
    """Ảnh xám 2D, không rỗng, dtype uint8 (dùng cho ngưỡng Otsu/adaptive).

    Tham số
    -------
    gray : ảnh một kênh.
    name : nhãn trong lỗi.

    Trả về
    ------
    Mảng đã ``asarray``.

    Liên quan: C5 phân đoạn — ngưỡng hóa cần một ma trận cường độ.
    """
    arr = np.asarray(gray)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if arr.dtype not in _UINT8:
        raise ValueError(f"{name} must be uint8")
    return arr


def apply_per_channel(img: np.ndarray, fn) -> np.ndarray:
    """Áp ``fn`` lên từng kênh RGB; ảnh xám thì gọi ``fn`` một lần.

    Tham số
    -------
    img : ảnh 2D hoặc 3D (3 kênh).
    fn : hàm nhận lát 2D, trả lát 2D cùng kích thước.

    Trả về
    ------
    Ảnh đầu ra cùng ``shape`` với đầu vào.

    Liên quan C2: cùng bộ lọc không gian áp độc lập R, G, B (không chuyển không gian màu).
    """
    if img.ndim == 2:
        return fn(img)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = fn(img[:, :, c])
    return out
