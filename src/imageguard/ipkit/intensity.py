"""Các phép biến đổi cường độ (point processing) — Chương 2, mục 2
"Basic intensity transformation functions" (bài giảng Image Processing,
giảng viên dunghv).

Mọi hàm ở đây đều là *phép toán điểm* (point operator): giá trị pixel đầu ra
chỉ phụ thuộc vào chính giá trị pixel đầu vào, không nhìn sang lân cận. Chúng
được cài bằng NumPy thuần thông qua **bảng tra LUT (Look-Up Table)**: ta tính
trước kết quả cho toàn bộ 256 mức xám có thể, rồi dùng mảng đầu vào làm chỉ số
để tra bảng. Làm vậy chỉ tính hàm 256 lần thay vì hàng triệu pixel.

Các hàm đều nhận ảnh uint8 2D (xám) hoặc H×W×3 (màu); với ảnh màu ta áp dụng
cùng một phép biến đổi lên từng kênh một nhờ :func:`apply_per_channel`.
"""

from __future__ import annotations

import numpy as np

from imageguard.ipkit._util import apply_per_channel, require_uint8_image


def _build_lut(values: np.ndarray) -> np.ndarray:
    """Làm tròn về số nguyên gần nhất rồi cắt vào [0, 255], đổi sang uint8."""
    # rint làm tròn nửa về số chẵn, khớp cách OpenCV làm tròn khi lượng tử hóa.
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _check_positive_float(value: float, name: str) -> float:
    """Ép về float và yêu cầu hữu hạn, lớn hơn 0."""
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return v


def _check_int_in_range(value: int, name: str, lo: int, hi: int) -> int:
    """Kiểm tra tham số nguyên trong [lo, hi]; sai kiểu -> TypeError, sai miền -> ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    v = int(value)
    if v < lo or v > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}]")
    return v


def negative(img: np.ndarray) -> np.ndarray:
    """Ảnh âm bản: ``s = L - 1 - r`` với ``L = 256``.

    Công thức ở slide "Negative Image" (Chương 2, mục 2): lấy mức xám lớn nhất
    trừ đi giá trị vào, làm nổi chi tiết trắng/xám nằm trong vùng tối — hữu ích
    khi vùng đen chiếm phần lớn ảnh (ví dụ ảnh chụp nhũ ảnh trong bài giảng).

    Args:
        img: Ảnh uint8 2D (xám) hoặc H×W×3 (màu).

    Returns:
        Ảnh uint8 cùng kích thước, từng kênh bị đảo mức xám.

    Example:
        >>> negative(np.array([[0, 255]], dtype=np.uint8))
        array([[255,   0]], dtype=uint8)
    """
    source = require_uint8_image(img)
    # Bảng tra đảo mức xám: mức r ánh xạ sang 255 - r.
    lut = (255 - np.arange(256)).astype(np.uint8)
    return apply_per_channel(source, lambda ch: lut[ch])


def log_transform(img: np.ndarray, c: float | None = None) -> np.ndarray:
    """Biến đổi log: ``s = c * log(1 + r)``.

    Theo slide "Log Transformations" (Chương 2, mục 2): ánh xạ một dải hẹp các
    giá trị tối thành một dải rộng, nén phần sáng — dùng để khuếch đại chi tiết
    trong vùng tối của ảnh có dải động lớn.

    Nếu ``c`` là ``None``, ta tự chọn ``c = 255 / log(1 + r_max)`` để mức xám
    lớn nhất của ảnh ánh xạ đúng về 255, còn mức 0 (log(1)=0) giữ ở 0.

    Args:
        img: Ảnh uint8 2D (xám) hoặc H×W×3 (màu).
        c: Hằng số dương tùy chọn; mặc định tự tính để đầu ra trải [0, 255].

    Returns:
        Ảnh uint8 cùng kích thước sau biến đổi log.

    Example:
        >>> log_transform(np.array([[0, 255]], dtype=np.uint8))
        array([[  0, 255]], dtype=uint8)
    """
    source = require_uint8_image(img)
    if c is None:
        # log(1 + 0) = 0 nên mức 0 luôn giữ 0; chọn c để mức lớn nhất về đúng 255.
        rmax = float(source.max())
        cc = 255.0 / np.log1p(rmax) if rmax > 0 else 1.0
    else:
        cc = _check_positive_float(c, "c")
    r = np.arange(256, dtype=np.float64)
    # log1p(r) = log(1 + r): chính xác hơn cho r nhỏ, không cần nhánh riêng cho 0.
    lut = _build_lut(cc * np.log1p(r))
    return apply_per_channel(source, lambda ch: lut[ch])


def gamma_transform(img: np.ndarray, gamma: float, c: float = 1.0) -> np.ndarray:
    """Biến đổi lũy thừa (power-law / gamma): ``s = c * r^gamma`` với ``r`` chuẩn hóa [0, 1].

    Theo slide "Power Law (Gamma) Transformations" (Chương 2, mục 2): với
    ``0 < gamma < 1`` ta mở rộng dải tối, với ``gamma > 1`` ta nén dải tối và mở
    rộng dải sáng; ``gamma = 1`` là phép đồng nhất. Dùng để hiệu chỉnh gamma cho
    màn hình CRT, máy in, máy quét.

    Args:
        img: Ảnh uint8 2D (xám) hoặc H×W×3 (màu).
        gamma: Số mũ dương (``gamma > 0``).
        c: Hằng số dương (mặc định 1.0).

    Returns:
        Ảnh uint8 cùng kích thước sau biến đổi gamma.

    Example:
        >>> gamma_transform(np.array([[0, 128, 255]], dtype=np.uint8), 1.0)
        array([[  0, 128, 255]], dtype=uint8)
    """
    source = require_uint8_image(img)
    g = _check_positive_float(gamma, "gamma")
    cc = _check_positive_float(c, "c")
    # Chuẩn hóa mức xám về [0, 1] trước khi nâng lũy thừa, rồi nhân lại 255.
    r = np.arange(256, dtype=np.float64) / 255.0
    lut = _build_lut(255.0 * cc * np.power(r, g))
    return apply_per_channel(source, lambda ch: lut[ch])


def _check_percent(value: float, name: str) -> float:
    """Ép về float và yêu cầu hữu hạn, nằm trong [0, 100]."""
    v = float(value)
    if not np.isfinite(v) or not (0.0 <= v <= 100.0):
        raise ValueError(f"{name} must be a finite number in [0, 100]")
    return v


def contrast_stretch(
    img: np.ndarray, low: float = 0.0, high: float = 100.0
) -> np.ndarray:
    """Kéo giãn tương phản tuyến tính (min-max), có thể cắt phân vị.

    Theo slide "Contrast Stretching" (Chương 2, mục 2): ánh xạ tuyến tính
    ``s = (r - r_min) * 255 / (r_max - r_min)`` để dải cường độ trải đủ [0, 255].
    Đặt ``(r1, s1) = (r_min, 0)`` và ``(r2, s2) = (r_max, L-1)`` đúng như bài giảng.

    Nếu ``low > 0`` hoặc ``high < 100``, ta lấy ``r_min``, ``r_max`` là các phân vị
    tương ứng thay cho min/max, nhờ đó loại bỏ ảnh hưởng của vài pixel nhiễu ở
    hai đuôi histogram.

    Args:
        img: Ảnh uint8 2D (xám) hoặc H×W×3 (màu).
        low: Phân vị dưới (0..100), mặc định 0 (dùng min).
        high: Phân vị trên (0..100), mặc định 100 (dùng max).

    Returns:
        Ảnh uint8 cùng kích thước sau khi kéo giãn.

    Example:
        >>> contrast_stretch(np.array([[20, 200]], dtype=np.uint8))
        array([[  0, 255]], dtype=uint8)
    """
    source = require_uint8_image(img)
    lo = _check_percent(low, "low")
    hi = _check_percent(high, "high")
    if lo >= hi:
        raise ValueError("low must be less than high")

    def _one(ch: np.ndarray) -> np.ndarray:
        # Mặc định dùng min/max; nếu cho phân vị thì cắt phần đuôi của histogram.
        if lo > 0.0 or hi < 100.0:
            rmin = float(np.percentile(ch, lo))
            rmax = float(np.percentile(ch, hi))
        else:
            rmin = float(ch.min())
            rmax = float(ch.max())
        # Ảnh hằng số (không có dải động) thì giữ nguyên, vì không thể kéo giãn.
        if rmax <= rmin:
            return ch.copy()
        r = np.arange(256, dtype=np.float64)
        # Nội suy tuyến tính từ [rmin, rmax] sang [0, 255]; ngoài khoảng bị cắt về biên.
        lut = _build_lut((r - rmin) * (255.0 / (rmax - rmin)))
        return lut[ch]

    return apply_per_channel(source, _one)


def _check_points(points: object) -> np.ndarray:
    """Kiểm tra danh sách điểm điều khiển (r, s): trả về mảng float64 (n, 2)."""
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        raise ValueError("points must be an (n, 2) array with at least 2 rows")
    if pts.dtype.kind not in "iuf":
        raise ValueError("points must be numeric")
    pts = pts.astype(np.float64)
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must be finite")
    r = pts[:, 0]
    s = pts[:, 1]
    if not np.all(np.diff(r) > 0):
        raise ValueError("r coordinates must be strictly increasing")
    if r.min() < 0.0 or r.max() > 255.0:
        raise ValueError("r coordinates must be in [0, 255]")
    if s.min() < 0.0 or s.max() > 255.0:
        raise ValueError("s coordinates must be in [0, 255]")
    return pts


def piecewise_linear(img: np.ndarray, points: object) -> np.ndarray:
    """Biến đổi tuyến tính từng đoạn qua các điểm điều khiển ``(r, s)``.

    Theo slide "Piecewise-Linear Transformation" (Chương 2, mục 2): nối các điểm
    điều khiển thành một đường gấp khúc; giữa hai điểm liên tiếp là nội suy tuyến
    tính, ngoài đoạn đầu/cuối giữ nguyên giá trị biên (hằng số). Trường hợp đặc
    biệt hai điểm ``(r_min, 0), (r_max, 255)`` chính là kéo giãn tương phản.

    Args:
        img: Ảnh uint8 2D (xám) hoặc H×W×3 (màu).
        points: Mảng ``(n, 2)`` các điểm ``(r, s)`` với ``r`` tăng nghiêm ngặt,
            ``r`` và ``s`` đều trong [0, 255].

    Returns:
        Ảnh uint8 cùng kích thước sau biến đổi.

    Example:
        >>> piecewise_linear(np.array([[0, 128, 255]], dtype=np.uint8),
        ...                  [(0, 0), (255, 255)])
        array([[  0, 128, 255]], dtype=uint8)
    """
    source = require_uint8_image(img)
    pts = _check_points(points)
    r = pts[:, 0]
    s = pts[:, 1]
    xs = np.arange(256, dtype=np.float64)
    # Với mỗi mức xám x, tìm đoạn chứa nó: chỉ số chèn bên phải trừ 1.
    seg = np.clip(np.searchsorted(r, xs, side="right") - 1, 0, len(r) - 2)
    x0 = r[seg]
    x1 = r[seg + 1]
    y0 = s[seg]
    y1 = s[seg + 1]
    # Nội suy tuyến tính trong đoạn; chặn t về [0, 1] để hai đầu là hằng số.
    t = np.clip((xs - x0) / (x1 - x0), 0.0, 1.0)
    lut = _build_lut(y0 + t * (y1 - y0))
    return apply_per_channel(source, lambda ch: lut[ch])


def threshold_binary(img: np.ndarray, thresh: int) -> np.ndarray:
    """Ngưỡng nhị phân: ``s = 255`` nếu ``r > thresh``, ngược lại ``s = 0``.

    Phép toán điểm "thresholding" được nhắc trong slide "Contrast Stretching"
    (Chương 2, mục 2) và "Intensity Level Slicing": tách ảnh thành hai mức để làm
    rõ vùng cần quan tâm. Khớp hành vi ``cv2.threshold(..., cv2.THRESH_BINARY)``.

    Args:
        img: Ảnh uint8 2D (xám) hoặc H×W×3 (màu).
        thresh: Ngưỡng nguyên trong [0, 255].

    Returns:
        Ảnh nhị phân uint8 (giá trị 0 hoặc 255), cùng kích thước.

    Example:
        >>> threshold_binary(np.array([[10, 200]], dtype=np.uint8), 128)
        array([[  0, 255]], dtype=uint8)
    """
    source = require_uint8_image(img)
    t = _check_int_in_range(thresh, "thresh", 0, 255)
    # Bảng tra: mức lớn hơn ngưỡng -> 255, còn lại -> 0.
    lut = np.where(np.arange(256) > t, 255, 0).astype(np.uint8)
    return apply_per_channel(source, lambda ch: lut[ch])


def bit_plane_slice(img: np.ndarray, bit: int) -> np.ndarray:
    """Tách mặt phẳng bit: trả về ảnh nhị phân của bit thứ ``bit``.

    Theo slide "Bit Plane Slicing" (Chương 2, mục 2): ảnh 8-bit gồm 8 mặt phẳng
    bit; bit 0 là bit thấp nhất (LSB, ít ý nghĩa nhất) còn bit 7 là bit cao nhất
    (MSB, mang phần lớn thông tin hình dạng). Kết quả là ảnh nhị phân hiển thị
    được: 255 nơi bit bằng 1, 0 nơi bit bằng 0.

    Args:
        img: Ảnh uint8 2D (xám) hoặc H×W×3 (màu).
        bit: Số bit nguyên trong [0, 7] (0 = LSB, 7 = MSB).

    Returns:
        Ảnh nhị phân uint8 (giá trị 0 hoặc 255), cùng kích thước.

    Example:
        >>> bit_plane_slice(np.array([[1, 128]], dtype=np.uint8), 7)
        array([[  0, 255]], dtype=uint8)
    """
    source = require_uint8_image(img)
    b = _check_int_in_range(bit, "bit", 0, 7)
    # Dịch phải b bit rồi lấy bit thấp nhất: bằng 1 -> trắng (255), bằng 0 -> đen.
    lut = np.where((np.arange(256) >> b) & 1, 255, 0).astype(np.uint8)
    return apply_per_channel(source, lambda ch: lut[ch])
