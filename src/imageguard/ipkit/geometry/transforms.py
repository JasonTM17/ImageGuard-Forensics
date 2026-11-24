"""Phép biến đổi hình học 2D trên ảnh uint8 — triển khai thuần NumPy.

Quy ước tọa độ: **x là cột, y là hàng**.  Ma trận 2×3 truyền tiếp (forward)
ánh xạ pixel nguồn sang pixel đầu ra theo ``[x', y'] = A·[x, y] + t``.  Khi lấy
mẫu ảnh đầu ra, dùng ma trận nghịch: ``nguồn = A_inv·[x, y] - A_inv·t`` với
``[x, y]`` là tọa độ trên lưới đầu ra.  Viền là hằng 0 và cùng một cặp tọa độ
được áp dụng cho từng kênh của ảnh màu.
"""

from __future__ import annotations

import math

import numpy as np

from imageguard.ipkit.geometry.sampling import (
    _require_image,
    sample_bilinear,
    sample_nearest,
)


def _check_interpolation(interpolation: str) -> str:
    """Chỉ chấp nhận ``"nearest"`` hoặc ``"bilinear"``."""
    if interpolation not in ("nearest", "bilinear"):
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")
    return interpolation


def _finite_number(value, *, name: str) -> float:
    """Ép số vô hướng về float và từ chối giá trị không hữu hạn.

    ``bool`` bị loại vì là subclass của ``int`` trong Python: ``True/False``
    không phải là hệ số biến đổi hợp lệ.
    """
    if isinstance(value, (bool, str, bytes)):
        raise TypeError(f"{name} must be a real number")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


# Trần giảng dạy: tránh cấp phát ``np.indices`` trên kích thước không dùng cho demo.
_MAX_SIDE = 4096
_MAX_PIXELS = 4_000_000


def _check_out_shape(out_shape) -> tuple[int, int]:
    """Kiểm tra kích thước đầu ra ``(height, width)`` gồm hai số nguyên dương.

    ``bool`` không được tính là kích thước nguyên dù kế thừa ``int``.
    Mỗi cạnh tối đa 4096 và tổng số pixel tối đa 4_000_000.
    """
    if isinstance(out_shape, (str, bytes)):
        raise TypeError("out_shape must be a (height, width) pair")
    try:
        values = tuple(out_shape)
    except TypeError as exc:
        raise TypeError("out_shape must be a (height, width) pair") from exc
    if len(values) != 2:
        raise ValueError("out_shape must have exactly two dimensions")
    dims: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("out_shape dimensions must be integers")
        dim = int(value)
        if dim <= 0:
            raise ValueError("out_shape dimensions must be positive")
        if dim > _MAX_SIDE:
            raise ValueError(f"out_shape dimensions must be <= {_MAX_SIDE}")
        dims.append(dim)
    if dims[0] * dims[1] > _MAX_PIXELS:
        raise ValueError(f"out_shape must contain at most {_MAX_PIXELS} pixels")
    return dims[0], dims[1]


def _reject_bool_matrix(matrix) -> None:
    """Từ chối phần tử ``bool`` trong ma trận (``bool`` là subclass của ``int``)."""
    if isinstance(matrix, np.ndarray):
        if matrix.dtype == np.bool_:
            raise TypeError("matrix entries must be real numbers")
        return
    stack = [matrix]
    while stack:
        item = stack.pop()
        if isinstance(item, (bool, np.bool_)):
            raise TypeError("matrix entries must be real numbers")
        if isinstance(item, (list, tuple)):
            stack.extend(item)


def _sampler(interpolation: str):
    """Chọn hàm lấy mẫu tương ứng với tên nội suy đã kiểm tra."""
    return sample_nearest if interpolation == "nearest" else sample_bilinear


def _apply(image, matrix: np.ndarray, out_shape: tuple[int, int], mode: str) -> np.ndarray:
    """Ánh xạ ngược lưới đầu ra về ảnh nguồn rồi lấy mẫu.

    Tham số
    -------
    image : ảnh uint8 H×W hoặc H×W×3.
    matrix : ma trận forward 2×3 (đã kiểm tra khả nghịch).
    out_shape : ``(height, width)`` của ảnh kết quả.
    mode : ``"nearest"`` hoặc ``"bilinear"`` (đã kiểm tra).
    """
    src = _require_image(image)
    out_h, out_w = _check_out_shape(out_shape)
    # np.indices trả (yy, xx): yy là chỉ số hàng/y, xx là chỉ số cột/x.
    yy, xx = np.indices((out_h, out_w), dtype=np.float64)

    inv_a = np.linalg.inv(matrix[:, :2])
    inv_t = inv_a @ matrix[:, 2]
    # source = A_inv·[x, y] - A_inv·t ; trục hoành dùng cho cột, tung cho hàng.
    src_x = inv_a[0, 0] * xx + inv_a[0, 1] * yy - inv_t[0]
    src_y = inv_a[1, 0] * xx + inv_a[1, 1] * yy - inv_t[1]
    return _sampler(mode)(src, src_y, src_x)


def translate(
    image: np.ndarray,
    tx: float,
    ty: float,
    *,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Dịch ảnh, giữ nguyên shape; ``tx>0`` sang phải, ``ty>0`` xuống dưới.

    Ma trận forward ``[[1, 0, tx], [0, 1, ty]]``; lấy mẫu nghịch nên đầu ra
    tại ``(x, y)`` lấy nguồn tại ``(x - tx, y - ty)``.  Vùng trống ở biên có
    giá trị 0 (không quay vòng ảnh).
    """
    mode = _check_interpolation(interpolation)
    dx = _finite_number(tx, name="tx")
    dy = _finite_number(ty, name="ty")
    src = _require_image(image)
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    return _apply(src, matrix, src.shape[:2], mode)


def scale(
    image: np.ndarray,
    sx: float,
    sy: float,
    *,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Co ảnh theo trục hoành ``sx`` / trục tung ``sy`` (đều phải > 0).

    Shape đầu ra là ``(max(1, round(H*sy)), max(1, round(W*sx)))``; ma trận
    forward ``[[sx, 0, 0], [0, sy, 0]]``.
    """
    mode = _check_interpolation(interpolation)
    fx = _finite_number(sx, name="sx")
    fy = _finite_number(sy, name="sy")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("sx and sy must be positive")
    src = _require_image(image)
    h, w = src.shape[:2]
    out_h = max(1, round(h * fy))
    out_w = max(1, round(w * fx))
    matrix = np.array([[fx, 0.0, 0.0], [0.0, fy, 0.0]], dtype=np.float64)
    return _apply(src, matrix, (out_h, out_w), mode)


def rotate(
    image: np.ndarray,
    angle_deg: float,
    *,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Xoay ảnh quanh tâm ``((W-1)/2, (H-1)/2)``, giữ nguyên shape.

    Góc dương là xoay ngược chiều kim đồng hồ; tỉ lệ luôn bằng 1.  Ma trận
    forward dùng hệ số ``alpha = cos(theta)``, ``beta = sin(theta)``:

    ``[[alpha, beta, (1-alpha)*cx - beta*cy],``
    `` [-beta, alpha, beta*cx + (1-alpha)*cy]]``
    """
    mode = _check_interpolation(interpolation)
    theta = math.radians(_finite_number(angle_deg, name="angle_deg"))
    src = _require_image(image)
    h, w = src.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    alpha, beta = math.cos(theta), math.sin(theta)
    matrix = np.array(
        [
            [alpha, beta, (1.0 - alpha) * cx - beta * cy],
            [-beta, alpha, beta * cx + (1.0 - alpha) * cy],
        ],
        dtype=np.float64,
    )
    return _apply(src, matrix, (h, w), mode)


def affine(
    image: np.ndarray,
    matrix: np.ndarray,
    out_shape,
    *,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Áp ma trận affine 2×3 tùy ý (phải hữu hạn và khả nghịch).

    Tham số
    -------
    image : ảnh uint8 H×W hoặc H×W×3.
    matrix : ma trận forward 2×3; khối 2×2 trên phải không sinh.
    out_shape : cặp số nguyên dương ``(height, width)`` của ảnh đầu ra.
    interpolation : ``"nearest"`` hoặc ``"bilinear"``.
    """
    mode = _check_interpolation(interpolation)
    _reject_bool_matrix(matrix)
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (2, 3):
        raise ValueError("matrix must have shape (2, 3)")
    if not np.all(np.isfinite(m)):
        raise ValueError("matrix must be finite")
    try:
        np.linalg.inv(m[:, :2])
    except np.linalg.LinAlgError as exc:
        raise ValueError("matrix must be invertible") from exc
    shape = _check_out_shape(out_shape)
    return _apply(image, m, shape, mode)
