"""Moment, Haar-like và LBPH — C6 Feature Extraction (HCMUTE, dunghv).

Chương trình môn Xử lý ảnh: shape moments & Hu invariants, summed-area table (Haar),
Local Binary Pattern Histogram (LBPH). Chỉ NumPy; không dùng OpenCV trong module.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from imageguard.ipkit._util import require_gray_uint8

HaarKind = Literal["two_h", "two_v", "three_h", "three_v", "four"]

# Bộ 8 láng giềng theo chiều kim đồng hồ từ góc trên-trái (C6 — LBPH).
_LBP_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
)


def _check_moment_order(p: int, q: int) -> tuple[int, int]:
    """Kiểm tra bậc moment ``(p,q)`` trước khi tính (C6 — shape moments)."""
    # bool là subclass int trong Python — tách ra để không nhận True/False như bậc.
    if isinstance(p, bool) or isinstance(q, bool):
        raise TypeError("p and q must be integers")
    if not isinstance(p, (int, np.integer)) or not isinstance(q, (int, np.integer)):
        raise TypeError("p and q must be integers")
    # Chuẩn hóa về int Python (từ np.int64 nếu cần) cho phép toán mũ ổn định.
    pi, qi = int(p), int(q)
    if pi < 0 or qi < 0:
        raise ValueError("p and q must be non-negative")
    return pi, qi


def raw_moment(gray: np.ndarray, p: int, q: int) -> float:
    """Moment thô \\(m_{p,q} = \\sum_{x,y} x^p y^q f(x,y)\\) (C6 — Shape moments).

    Tham số ``p`` là bậc theo cột ``x``, ``q`` theo hàng ``y``; ảnh uint8.

    Ví dụ::

        m00 = raw_moment(gray, 0, 0)  # "khối lượng" / tổng cường độ

    Ví dụ 2×2 (cường độ f, x cột y hàng)::

        [[1, 3],     m10 = 0*1+1*3+0*4+1*6 = 9  (trọng số x^1 y^0)
         [4, 6]]
    """
    # Ảnh grayscale uint8 — đầu vào bắt buộc cho mọi hàm moment trong ipkit.
    img = require_gray_uint8(gray)
    p_i, q_i = _check_moment_order(p, q)
    h, w = img.shape
    # m00: không cần lưới tọa độ — tổng cường độ = “khối lượng” vùng (segmentation/shape).
    if p_i == 0 and q_i == 0:
        return float(img.sum(dtype=np.float64))
    # Lưới tọa độ ảnh: x = cột, y = hàng (khớp cv2.moments).
    yy, xx = np.indices((h, w), dtype=np.float64)
    # Trọng số không gian x^p y^q — moment bậc cao nhấn vùng xa gốc tọa độ.
    weights = (xx**p_i) * (yy**q_i)
    # Tổng có trọng số = định nghĩa m_{p,q} (feature extraction / mô tả hình dạng).
    return float(np.sum(weights * img.astype(np.float64)))


def centroid(gray: np.ndarray) -> tuple[float, float]:
    """Tâm khối \\((\\bar x, \\bar y) = (m_{10}/m_{00}, m_{01}/m_{00})\\) (C6).

    Trả về ``(x_bar, y_bar)`` với ``x`` là cột, ``y`` là hàng.

    Ví dụ::

        cx, cy = centroid(blob)
    """
    img = require_gray_uint8(gray)
    m00 = raw_moment(img, 0, 0)
    # Không có khối lượng → tâm không xác định (ROI toàn 0 sau segmentation).
    if m00 == 0:
        raise ValueError("image has zero mass (m00 == 0)")
    # Trung bình có trọng số theo cường độ: vị trí “trọng tâm” blob.
    return raw_moment(img, 1, 0) / m00, raw_moment(img, 0, 1) / m00


def central_moment(gray: np.ndarray, p: int, q: int) -> float:
    """Moment trung tâm \\(\\mu_{p,q} = \\sum (x-\\bar x)^p (y-\\bar y)^q f(x,y)\\) (C6).

    Ví dụ::

        mu20 = central_moment(gray, 2, 0)
    """
    img = require_gray_uint8(gray)
    p_i, q_i = _check_moment_order(p, q)
    x_bar, y_bar = centroid(img)
    h, w = img.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    # Đưa gốc moment về tâm khối để bất biến translation.
    dx = xx - x_bar
    dy = yy - y_bar
    # Moment trung tâm: trọng số trên độ lệch (x-x̄, y-ȳ) — nền tảng Hu invariants.
    weights = (dx**p_i) * (dy**q_i)
    return float(np.sum(weights * img.astype(np.float64)))


def normalized_central_moment(gray: np.ndarray, p: int, q: int) -> float:
    r"""Moment trung tâm chuẩn hóa \(\eta_{p,q} = \mu_{p,q} / m_{00}^{\gamma}\), \(\gamma=(p+q)/2+1\) (C6).

    Dùng cho Hu invariants và so khớp cv2 ``nu*`` trong ``cv2.moments``.

    Ví dụ::

        eta11 = normalized_central_moment(gray, 1, 1)
    """
    img = require_gray_uint8(gray)
    p_i, q_i = _check_moment_order(p, q)
    m00 = raw_moment(img, 0, 0)
    if m00 == 0:
        raise ValueError("image has zero mass (m00 == 0)")
    mu = central_moment(img, p_i, q_i)
    # γ = (p+q)/2+1: chuẩn hóa bậc để η bất biến scale (đồng bộ cv2 nu*).
    gamma = (p_i + q_i) / 2.0 + 1.0
    return mu / (m00**gamma)


def hu_moments(gray: np.ndarray) -> np.ndarray:
    """Bảy bất biến Hu \\(\\phi_1\\ldots\\phi_7\\) (Hu 1962) — C6 moment invariants / TRS.

    Dùng moment trung tâm chuẩn hóa \\(\\eta_{p,q}\\): \\(\\phi_1=\\eta_{20}+\\eta_{02}\\),
    \\(\\phi_2=(\\eta_{20}-\\eta_{02})^2+4\\eta_{11}^2\\), … \\(\\phi_7\\) đầy đủ trong code.
    Đầu ra khớp ``cv2.HuMoments`` trên OpenCV 4.x (giá trị \\(\\phi_k\\) thô, không log).

    Ví dụ::

        hu = hu_moments(shape_roi)
    """
    img = require_gray_uint8(gray)

    def eta(pi: int, qi: int) -> float:
        return normalized_central_moment(img, pi, qi)

    # η bậc 2–3 đủ để ghép các công thức Hu cổ điển (7 đại lượng bất biến).
    n20, n11, n02 = eta(2, 0), eta(1, 1), eta(0, 2)
    n30, n21, n12, n03 = eta(3, 0), eta(2, 1), eta(1, 2), eta(0, 3)

    # Bất biến Hu cổ điển trên η (chuẩn hóa theo m00) — feature extraction / nhận dạng hình.
    # φ1–φ2: tổ hợp bậc 2 (độ “tròn”, hướng trục chính).
    phi1 = n20 + n02
    phi2 = (n20 - n02) ** 2 + 4.0 * n11**2
    # φ3–φ4: tổ hợp bậc 3 (asymmetry, “đầu mũi”).
    phi3 = (n30 - 3.0 * n12) ** 2 + (3.0 * n21 - n03) ** 2
    phi4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    # φ5–φ7: tích η bậc cao — bất biến xoay/đối xứng phức tạp hơn (vẫn không đổi TRS lý tưởng).
    phi5 = (n30 - 3.0 * n12) * (n30 + n12) * (phi4 - 3.0 * (n21 + n03) ** 2) + (
        3.0 * n21 - n03
    ) * (n21 + n03) * (3.0 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    phi6 = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4.0 * n11 * (
        n30 + n12
    ) * (n21 + n03)
    phi7 = (3.0 * n21 - n03) * (n30 + n12) * (phi4 - 3.0 * (n21 + n03) ** 2) - (
        n30 - 3.0 * n12
    ) * (n21 + n03) * (3.0 * (n30 + n12) ** 2 - (n21 + n03) ** 2)

    # Trả về φ_k trực tiếp (OpenCV 4 HuMoments trên môi trường đồ án).
    return np.array((phi1, phi2, phi3, phi4, phi5, phi6, phi7), dtype=np.float64)


def integral_image(gray: np.ndarray) -> np.ndarray:
    """Ảnh tích phân / SAT: tổng các pixel phía trên và bên trái (C6 — integral image).

    Kích thước ``(H+1, W+1)``, hàng/cột 0 bằng 0; khớp ``cv2.integral``.

    Công thức: ``I[i,j] = sum gray[0:i, 0:j]`` (half-open trên chỉ số ảnh gốc).

    Ví dụ::

        sat = integral_image(gray)
        s = rect_sum(sat, top, left, h, w)

    Ví dụ 3×3::

        gray=[[1,2,3],[4,5,6],[7,8,9]]  → sat hàng/col 0 = 0, ô (i,j)=tổng góc trên-trái.
    """
    img = require_gray_uint8(gray).astype(np.float64)
    # SAT 2D: tích lũy theo hàng rồi cột, thêm viền 0 để tra cứu 4 điểm.
    row_cum = np.cumsum(img, axis=1)
    total = np.cumsum(row_cum, axis=0)
    # Ma trận (H+1)×(W+1): hàng/cột 0 = 0 để chỉ số rect_sum trùng cv2.integral.
    sat = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    sat[1:, 1:] = total
    return sat


def rect_sum(
    sat: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
) -> float:
    """Tổng cường độ trong H×W chữ nhật qua 4 điểm SAT (C6: Sum = S4 − S2 − S3 + S1).

    ``(top, left)`` góc trên-trái; ``height``/``width`` kích thước (pixel, không âm).
    Chỉ số SAT: ``S1=sat[top,left]``, ``S2=sat[top,left+width]``,
    ``S3=sat[top+height,left]``, ``S4=sat[top+height,left+width]``.

    Ví dụ::

        total = rect_sum(sat, 0, 0, gray.shape[0], gray.shape[1])
    """
    sat_arr = np.asarray(sat, dtype=np.float64)
    if sat_arr.ndim != 2 or sat_arr.shape[0] < 2 or sat_arr.shape[1] < 2:
        raise ValueError("sat must be a 2D summed-area table with size at least 2x2")
    # Kiểm tra kiểu từng tham số hình học — tránh bool lọt vào chỉ số SAT.
    for name, val in (
        ("top", top),
        ("left", left),
        ("height", height),
        ("width", width),
    ):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    t, l, hh, ww = int(top), int(left), int(height), int(width)
    if hh < 0 or ww < 0:
        raise ValueError("height and width must be non-negative")
    # Kích thước ảnh gốc = sat−1 (hàng/cột 0 là padding).
    h_max, w_max = sat_arr.shape[0] - 1, sat_arr.shape[1] - 1
    if t < 0 or l < 0 or t + hh > h_max or l + ww > w_max:
        raise ValueError("rectangle exceeds image bounds")
    # Công thức bài giảng: D = S4 + S1 − S2 − S3 (tổng trong ABC D).
    return float(
        sat_arr[t + hh, l + ww]
        - sat_arr[t, l + ww]
        - sat_arr[t + hh, l]
        + sat_arr[t, l]
    )


def haar_feature(
    sat: np.ndarray,
    kind: HaarKind,
    top: int,
    left: int,
    height: int,
    width: int,
) -> float:
    """Giá trị Haar-like trong cửa sổ (C6 — difference of adjacent rectangles).

    Quy ước dấu (vùng trắng ``+1``, vùng đen ``−1``, Viola–Jones kiểu dọc/ngang):
    ``two_h``: nửa trái ``+``, nửa phải ``−``;
    ``two_v``: nửa trên ``+``, nửa dưới ``−``;
    ``three_h`` / ``three_v``: giữa ``+2``, hai bên ``−1`` (chia 3 bằng phần);
    ``four``: ``(TL+BR) − (TR+BL)``.

    ``two_h`` với bên trái tối / phải sáng thường **âm** (tổng trái < tổng phải).

    Ví dụ::

        v = haar_feature(sat, "two_h", 0, 0, 24, 24)
    """
    if kind not in ("two_h", "two_v", "three_h", "three_v", "four"):
        raise ValueError("kind must be one of two_h, two_v, three_h, three_v, four")
    hh, ww = int(height), int(width)
    if hh <= 0 or ww <= 0:
        raise ValueError("height and width must be positive")
    t, l = int(top), int(left)

    # Gói rect_sum trong cửa sổ (top,left) — mọi Haar là tổ hợp các hình chữ nhật con.
    def rs(dy: int, dx: int, h: int, w: int) -> float:
        return rect_sum(sat, t + dy, l + dx, h, w)

    if kind == "two_h":
        # Chia đôi chiều ngang (nguyên //2, phần lẻ ở nửa phải).
        half = ww // 2
        if half == 0 or ww - half <= 0:
            raise ValueError("width too small for two_h")
        # Chênh lệch trái−phải qua SAT.
        return rs(0, 0, hh, half) - rs(0, half, hh, ww - half)
    if kind == "two_v":
        half = hh // 2
        if half == 0 or hh - half <= 0:
            raise ValueError("height too small for two_v")
        # Chênh lệch trên−dưới (Viola–Jones: cạnh ngang nổi bật → dấu dương/âm).
        return rs(0, 0, half, ww) - rs(half, 0, hh - half, ww)
    if kind == "three_h":
        # Ba cột: hai lần //3 + phần dư → cột phải có thể rộng hơn 1 px.
        w_left = ww // 3
        w_mid = ww // 3
        w_right = ww - w_left - w_mid
        if w_left == 0 or w_mid == 0 or w_right == 0:
            raise ValueError("width too small for three_h")
        # Chia ngang 3 phần (//3 + phần dư vào cạnh phải).
        left_sum = rs(0, 0, hh, w_left)
        mid_sum = rs(0, w_left, hh, w_mid)
        right_sum = rs(0, w_left + w_mid, hh, w_right)
        # Giữa +2, hai bên −1 (feature line 3 ô ngang).
        return 2.0 * mid_sum - left_sum - right_sum
    if kind == "three_v":
        h_top = hh // 3
        h_mid = hh // 3
        h_bot = hh - h_top - h_mid
        if h_top == 0 or h_mid == 0 or h_bot == 0:
            raise ValueError("height too small for three_v")
        top_sum = rs(0, 0, h_top, ww)
        mid_sum = rs(h_top, 0, h_mid, ww)
        bot_sum = rs(h_top + h_mid, 0, h_bot, ww)
        # Băng giữa +2, hai băng ngoài −1 (feature line dọc).
        return 2.0 * mid_sum - top_sum - bot_sum
    # four: đường chéo đối xứng trong cửa sổ 2×2.
    half_h, half_w = hh // 2, ww // 2
    if half_h == 0 or half_w == 0:
        raise ValueError("window too small for four")
    # Cửa sổ 2×2 ô: đối chéo (TL+BR) trừ (TR+BL) — cạnh chéo / mắt trong cascade Haar.
    tl = rs(0, 0, half_h, half_w)
    tr = rs(0, half_w, half_h, ww - half_w)
    bl = rs(half_h, 0, hh - half_h, half_w)
    br = rs(half_h, half_w, hh - half_h, ww - half_w)
    return (tl + br) - (tr + bl)


def haar_feature_map(
    gray: np.ndarray,
    kind: HaarKind,
    win_h: int,
    win_w: int,
    stride: int,
) -> np.ndarray:
    """Bản đồ Haar-like: quét cửa sổ ``win_h×win_w`` bước ``stride`` (C6).

    Ví dụ::

        fm = haar_feature_map(gray, "two_v", 8, 8, 4)
    """
    img = require_gray_uint8(gray)
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)):
        raise TypeError("stride must be an integer")
    st = int(stride)
    if st <= 0:
        raise ValueError("stride must be positive")
    wh, ww = int(win_h), int(win_w)
    if wh <= 0 or ww <= 0:
        raise ValueError("win_h and win_w must be positive")
    h, w = img.shape
    if wh > h or ww > w:
        raise ValueError("window larger than image")
    # Một SAT cho cả ảnh — O(1) mỗi vị trí cửa sổ (khác brute-force cộng pixel).
    sat = integral_image(img)
    # Kích thước bản đồ đầu ra theo sliding window stride cố định.
    out_h = (h - wh) // st + 1
    out_w = (w - ww) // st + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("no window positions for given size and stride")
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for iy, top in enumerate(range(0, h - wh + 1, st)):
        for ix, left in enumerate(range(0, w - ww + 1, st)):
            out[iy, ix] = haar_feature(sat, kind, top, left, wh, ww)
    return out


def _lbp_code_at(img: np.ndarray, row: int, col: int) -> int:
    """Mã LBP 8-bit tại một pixel nội (C6 — feature extraction / texture)."""
    center = int(img[row, col])
    code = 0
    for bit, (dr, dc) in enumerate(_LBP_OFFSETS):
        nr, nc = row + dr, col + dc
        neighbor = int(img[nr, nc])
        # Quy tắc bài giảng (theo yêu cầu đồ án): bit=1 khi láng giềng >= tâm.
        if neighbor >= center:
            # Gán bit thứ ``bit`` (chiều kim đồng hồ) trong mã nhị phân 8-bit.
            code |= 1 << bit
    return code


def lbp_image(gray: np.ndarray, radius: int = 1, neighbors: int = 8) -> np.ndarray:
    """Ảnh mã LBP 8-bit (C6 — LBPH): láng giềng quay chiều kim đồng hồ, bit=1 nếu **>=** tâm.

    Chỉ hỗ trợ ``radius=1``, ``neighbors=8`` (ô 3×3). Biên không đủ 8 điểm → 0.

    Thứ tự bit (LSB = láng giềng đầu tiên): trên-trái → trên → trên-phải → phải →
    dưới-phải → dưới → dưới-trái → trái.

    Ví dụ::

        codes = lbp_image(face_gray, radius=1, neighbors=8)

    Ví dụ 3×3 quanh tâm 5 (láng giềng ≥5 → bit 1)::

        7 6 8        Tâm=5 → mã LBP phụ thuộc thứ tự 8 hướng (_LBP_OFFSETS).
        4 5 9
        1 3 2
    """
    # Bản dạy chỉ cửa sổ 3×3 cố định; radius≠1 cần nội suy vòng tròn (ngoài phạm vi).
    if radius != 1 or neighbors != 8:
        raise ValueError(
            "only radius=1 and neighbors=8 are supported in this teaching build"
        )
    img = require_gray_uint8(gray)
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.uint8)
    if h < 3 or w < 3:
        return out
    # Bỏ viền 1 pixel vì cần đủ 8 láng giềng.
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            out[row, col] = _lbp_code_at(img, row, col)
    return out


def _transition_count(code: int) -> int:
    """Đếm số lần đổi 0↔1 trên vòng 8 bit LBP (uniform pattern — rotation)."""
    # Tách 8 bit LSB→MSB theo thứ tự láng giềng đã mã hóa.
    bits = [(code >> i) & 1 for i in range(8)]
    # Nối vòng để bắt transition giữa bit 7 và bit 0.
    doubled = bits + bits
    return sum(1 for i in range(8) if doubled[i] != doubled[i + 1])


def uniform_pattern_lookup() -> np.ndarray:
    """Ánh xạ mã LBP 0…255 → nhãn uniform 0…58 (59 bin, Ojala uniform patterns).

    Pattern **uniform** khi số lần đổi 0↔1 trên vòng ≤ 2 (nhạy rotation trong lớp mẫu).
    Mọi mã không uniform gộp vào bin 58.

    Ví dụ::

        lut = uniform_pattern_lookup()
        assert lut.shape == (256,)
    """
    # Mặc định mọi mã vào bin 58 (nhóm non-uniform gộp chung).
    lut = np.full(256, 58, dtype=np.int32)
    uniform_codes: list[int] = []
    for code in range(256):
        if _transition_count(code) <= 2:
            uniform_codes.append(code)
    # 58 pattern uniform + 1 ô “non-uniform” = 59 bin LBPH chuẩn.
    for bin_id, code in enumerate(uniform_codes):
        lut[code] = bin_id
    return lut


def lbph(
    gray: np.ndarray, grid: tuple[int, int] = (8, 8), uniform: bool = True
) -> np.ndarray:
    """Descriptor LBPH: histogram LBP theo lưới ``grid`` (hàng, cột), L1-norm (C6).

    ``uniform=True`` dùng ``uniform_pattern_lookup`` (59 bin/cell). Chiều dài =
    ``grid[0] * grid[1] * n_bins``.

    Ví dụ::

        hist = lbph(gray, grid=(8, 8), uniform=True)
    """
    img = require_gray_uint8(gray)
    if (
        not isinstance(grid, tuple)
        or len(grid) != 2
        or not isinstance(grid[0], (int, np.integer))
        or not isinstance(grid[1], (int, np.integer))
    ):
        raise TypeError("grid must be a pair of integers (rows, cols)")
    n_rows, n_cols = int(grid[0]), int(grid[1])
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("grid dimensions must be positive")
    # Mã LBP toàn ảnh trước — mỗi ô lưới chỉ cần histogram cục bộ.
    lbp = lbp_image(img)
    lut = uniform_pattern_lookup() if uniform else None
    n_bins = 59 if uniform else 256
    h, w = img.shape
    # Chia ô lưới gần bằng nhau (cạnh cuối có thể rộng hơn một pixel).
    row_edges = np.linspace(0, h, n_rows + 1, dtype=int)
    col_edges = np.linspace(0, w, n_cols + 1, dtype=int)
    parts: list[np.ndarray] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            patch = lbp[
                row_edges[ri] : row_edges[ri + 1], col_edges[ci] : col_edges[ci + 1]
            ]
            # Ánh xạ uniform → 59 bin; đếm tần suất pattern trong từng ô (spatial grid).
            hist = np.bincount(
                lut[patch.ravel()] if uniform else patch.ravel(), minlength=n_bins
            ).astype(np.float64)
            parts.append(hist)
    # Ghép histogram các ô theo thứ tự hàng-cột thành vector đặc trưng một chiều.
    vec = np.concatenate(parts)
    total = vec.sum()
    # L1 chuẩn hóa — so sánh khuôn mặt / texture độc lập độ sáng tổng thể.
    if total > 0:
        vec /= total
    return vec.astype(np.float64)


def chi_square_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    r"""Khoảng \(\chi^2\) giữa hai histogram LBPH: \(\sum (h_1-h_2)^2/(h_1+h_2+\epsilon)\) (C6).

    Đối xứng; hai histogram giống nhau → 0. ``epsilon`` nhỏ tránh chia cho 0.

    Ví dụ::

        d = chi_square_distance(lbph(a), lbph(b))
    """
    a = np.asarray(h1, dtype=np.float64).ravel()
    b = np.asarray(h2, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("histograms must have the same length")
    if a.size == 0:
        raise ValueError("empty histogram")
    denom = a + b
    # Bỏ qua bin cả hai bằng 0 — không đóng góp vào χ² (tránh 0/0).
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    diff = a - b
    # Tổng (h1−h2)²/(h1+h2) trên các bin có tín hiệu — đo khác biệt LBPH sau chuẩn hóa.
    return float(np.sum((diff[mask] ** 2) / denom[mask]))
