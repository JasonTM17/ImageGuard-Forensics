"""Toán tử hình thái học ảnh xám/nhị phân (chỉ NumPy).

Chủ đề C4: phần tử cấu trúc (structuring element), co giãn (erosion/dilation),
mở/đóng, gradient và top-hat — mỗi pixel là min/max trong cửa sổ se.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from imageguard.ipkit._util import apply_per_channel, require_uint8_image

ShapeKind = Literal["rect", "cross", "ellipse"]


def structuring_element(kind: ShapeKind, ksize: int) -> np.ndarray:
    """Tạo phần tử cấu trúc nhị phân (0/1) hình chữ nhật, chữ thập hoặc elip.

    Tham số: ``kind`` loại hình; ``ksize`` lẻ dương — kích thước ma trận.

    Trả về: uint8, ô = 1 là vị trí tham gia phép hình thái.

    Liên quan C4 — định hình cửa sổ cho erosion/dilation.
    """
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
    """Co ảnh xám: tại mỗi pixel = min các mẫu dưới ``se`` (chỉ ô se=1).

    Đệm 255 ở viền để erosion không bị kéo về 0 ở mép (giả lập nền sáng).
    """
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
    """Giãn ảnh xám: mỗi pixel = max trong cửa sổ ``se``; viền đệm 0."""
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
    """Kiểm tra phần tử cấu trúc 2D lẻ, giá trị đúng 0 hoặc 1 (không làm tròn 0.3→active).

    Chỉ giá trị hữu hạn và đúng 0/1 được chấp nhận; NaN hay phân số bị từ chối.

    Trả về: ``se`` dạng uint8 0/1.

    Liên quan C4 — định nghĩa miền cấu trúc trước min/max.
    """
    elem = np.asarray(se)
    if (
        elem.ndim != 2
        or elem.size == 0
        or elem.shape[0] % 2 == 0
        or elem.shape[1] % 2 == 0
    ):
        raise ValueError("se must be a non-empty 2D structuring element with odd sizes")
    if elem.dtype.kind not in "biuf":
        raise ValueError("se must be numeric or boolean")
    if elem.dtype.kind == "b":
        return elem.astype(np.uint8)
    values = elem.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("se must be binary (0/1); found non-finite values")
    if not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError("se must be binary (0/1); values must be exactly 0 or 1")
    return values.astype(np.uint8)


def erode(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Co (erosion): min trong cửa sổ ``se`` cho từng kênh uint8.

    Liên quan C4 — làm co vật thể sáng, xóa nhiễu sáng nhỏ.
    """
    source = require_uint8_image(img)
    elem = _check_se(se)
    return apply_per_channel(source, lambda ch: _erode_gray(ch, elem))


def dilate(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Giãn (dilation): max trong cửa sổ ``se``.

    Liên quan C4 — làm phình vật thể, lấp lỗ tối nhỏ.
    """
    source = require_uint8_image(img)
    elem = _check_se(se)
    return apply_per_channel(source, lambda ch: _dilate_gray(ch, elem))


def open_(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Mở (opening): dilate(erode) — xóa chi tiết sáng hẹp hơn ``se``.

    Liên quan C4 — khử nhiễu sáng mà không đổi kích thước vật thể lớn nhiều.
    """
    return dilate(erode(img, se), se)


def close_(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Đóng (closing): erode(dilate) — lấp khe tối hẹp.

    Liên quan C4 — nối đường viền bị đứt, giữ khối sáng lớn.
    """
    return erode(dilate(img, se), se)


def gradient(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Gradient hình thái: dilate − erode (kẹp uint8).

    Nhấn mạnh biên vật thể theo độ dày ~ ``se``.

    Liên quan C4; có thể liên hệ C2 khi so sánh biên gradient vs Sobel.
    """
    d = dilate(img, se).astype(np.int16)
    e = erode(img, se).astype(np.int16)
    return np.clip(d - e, 0, 255).astype(np.uint8)


def tophat(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """White top-hat: ảnh − open(ảnh) — trích chi tiết sáng nhỏ hơn ``se``.

    Liên quan C4 — tăng cường đối tượng nhỏ trên nền tương đối đồng đều.
    """
    base = require_uint8_image(img)
    opened = open_(base, se).astype(np.int16)
    return np.clip(base.astype(np.int16) - opened, 0, 255).astype(np.uint8)
