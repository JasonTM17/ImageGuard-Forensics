"""Toán tử gradient và biên qua tích chập 2D (chỉ NumPy).

Chủ đề C2 — lọc không gian: Sobel, Prewitt, Laplacian, biên độ gradient.
Giải thích convolution vs correlation ngay tại ``convolve2d``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from imageguard.ipkit._util import apply_per_channel, require_uint8_image

PaddingMode = Literal["zero", "replicate"]

# Ma trận Sobel: đạo hàm bậc 1 có làm mịn theo hướng vuông góc (trọng số 2 ở tâm)
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
# Prewitt: tương tự Sobel nhưng mọi hàng/cột đều trọng số 1 (nhạy nhiễu hơn một chút)
_PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
_PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
# Laplace rời rạc 4-láng giềng: ΔI ≈ I_trên + I_dưới + I_trái + I_phải − 4·I_trung tâm
_LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def _check_padding(padding: str) -> PaddingMode:
    """Chỉ cho phép viền zero hoặc lặp mép ảnh (replicate)."""
    if padding not in ("zero", "replicate"):
        raise ValueError("padding must be 'zero' or 'replicate'")
    return padding  # type: ignore[return-value]


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    padding: PaddingMode = "replicate",
) -> np.ndarray:
    """Tương quan 2D (correlation) giữa ảnh và ``kernel`` (kích thước lẻ).

    Convolution đúng nghĩa lật kernel 180° trước khi trượt; ở đây kernel được
    dùng *nguyên hướng* như trong OpenCV ``filter2D`` — tức là correlation:
    tại mỗi pixel, tổng có trọng số của các mẫu lân cận khớp *vị trí* ô ma trận.

    Viền: ``zero`` = đệm 0 (mode constant); ``replicate`` = kéo giá trị mép (edge).

    Tham số: ``image`` 2D float; ``kernel`` 2D lẻ; ``padding`` như trên.

    Trả về: float32 cùng kích thước ảnh.

    Liên quan C2 — lọc không gian tuyến tính / tích chập.
    """
    _check_padding(padding)
    ker = np.asarray(kernel, dtype=np.float64)
    if ker.ndim != 2 or ker.shape[0] % 2 == 0 or ker.shape[1] % 2 == 0:
        raise ValueError("kernel must be 2D with odd height and width")
    if ker.size == 0:
        raise ValueError("kernel must be non-empty")

    src = np.asarray(image, dtype=np.float64)
    if src.ndim != 2 or src.size == 0:
        raise ValueError("image must be a non-empty 2D array")

    ph, pw = ker.shape[0] // 2, ker.shape[1] // 2
    mode = "constant" if padding == "zero" else "edge"
    pad_kw = {"constant_values": 0.0} if padding == "zero" else {}
    padded = np.pad(src, ((ph, ph), (pw, pw)), mode=mode, **pad_kw)

    h, w = src.shape
    out = np.zeros((h, w), dtype=np.float64)
    kh, kw = ker.shape
    # Mỗi (yi,xi) gom toàn ảnh dịch — tương đương tổng các mặt nạ dịch, không lật kernel
    for yi in range(kh):
        for xi in range(kw):
            out += ker[yi, xi] * padded[yi : yi + h, xi : xi + w]
    return out.astype(np.float32)


def sobel(img: np.ndarray, padding: PaddingMode = "replicate") -> np.ndarray:
    """Gradient Sobel: ``(gx, gy)`` float32; ảnh RGB thì từng kênh.

    gx đạo hàm theo x, gy theo y — độ lớn biên thường lấy sqrt(gx²+gy²).

    Liên quan C2 — phát hiện biên bằng lọc đạo hàm bậc một.
    """
    source = require_uint8_image(img)
    pad = _check_padding(padding)

    def _one(ch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = convolve2d(ch, _SOBEL_X, pad)
        gy = convolve2d(ch, _SOBEL_Y, pad)
        return gx, gy

    if source.ndim == 2:
        return _one(source)
    gxs, gys = [], []
    for c in range(3):
        gx, gy = _one(source[:, :, c])
        gxs.append(gx)
        gys.append(gy)
    return np.stack(gxs, axis=2), np.stack(gys, axis=2)


def prewitt(img: np.ndarray, padding: PaddingMode = "replicate") -> np.ndarray:
    """Gradient Prewitt: cặp ``(gx, gy)`` như Sobel với kernel đều trọng số.

    Liên quan C2 — biên; so sánh với Sobel khi giảng viên hỏi độ mịn vs nhiễu.
    """
    source = require_uint8_image(img)
    pad = _check_padding(padding)

    def _one(ch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return convolve2d(ch, _PREWITT_X, pad), convolve2d(ch, _PREWITT_Y, pad)

    if source.ndim == 2:
        return _one(source)
    gxs, gys = [], []
    for c in range(3):
        gx, gy = _one(source[:, :, c])
        gxs.append(gx)
        gys.append(gy)
    return np.stack(gxs, axis=2), np.stack(gys, axis=2)


def laplacian(img: np.ndarray, padding: PaddingMode = "replicate") -> np.ndarray:
    """Laplacian rời rạc 4-láng giềng (đạo hàm bậc hai), float32.

    Điểm cực trị đạo hàm bậc một có thể là không, hoặc dùng kết hợp với ảnh gốc
    (unsharp / zero-crossing) — mã này chỉ trả đáp ứng Laplace.

    Liên quan C2 — biên và làm sắc nét.
    """
    source = require_uint8_image(img)
    pad = _check_padding(padding)
    return apply_per_channel(
        source, lambda ch: convolve2d(ch, _LAPLACIAN, pad)
    )


def gradient_magnitude(img: np.ndarray, padding: PaddingMode = "replicate") -> np.ndarray:
    """Biên độ gradient từ Sobel: sqrt(gx² + gy²).

    Liên quan C2; thường dùng làm bước tiền xử lý C5 (ngưỡng trên ảnh gradient).
    """
    gx, gy = sobel(img, padding)
    return np.sqrt(gx * gx + gy * gy, dtype=np.float32)
