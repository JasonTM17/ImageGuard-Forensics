"""Các toán tử phân đoạn ảnh (Segmentation) — Chương 5 môn Xử lý ảnh.

Triển khai thuần NumPy (không `cv2`/`scipy`/`skimage`) các kỹ thuật mà bài
giảng Chương 5 (Digital Image Processing, PGS.TS. Hoàng Văn Dũng — dunghv)
giới thiệu:

* Mục 5.2 — Phát hiện điểm, đường và biên: ``detect_points``, ``detect_lines``,
  ``canny_edges``, ``hough_lines``.
* Mục 5.3 — Ngưỡng (thresholding): ``iterative_global_threshold``.
* Mục 5.4 — Phân đoạn theo vùng: ``region_growing``.
* Mục 5.5 — Phân đoạn bằng các tiếp cận mới (watershed): ``simple_watershed``,
  và ``kmeans_segmentation`` (phân cụm theo mức xám/màu).

Đây là mã học tập/demo: mỗi toán tử viết lại "từ gốc" để sinh viên đọc và giải
thích được từng dòng trước giảng viên, không phải bản sao chính xác từng pixel
của OpenCV.
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Sequence

import numpy as np

from imageguard.ipkit._util import require_gray_uint8, require_uint8_image
from imageguard.ipkit.edges import convolve2d, sobel

# ---------------------------------------------------------------------------
# Mục 5.2 — Phát hiện điểm, đường và biên (Point, Line, and Edge Detection)
# ---------------------------------------------------------------------------

# Kernel "Laplacian" 3×3 dùng cho phát hiện điểm (điểm đơn độc). Các trọng số
# bao quanh dương và tâm âm nên một điểm sáng cô lập cho phản hồi âm mạnh.
_POINT_KERNEL = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)

# Bốn kernel phát hiện đường một pixel (Line Detection, hình "Line detection
# kernels" trong slide). Mỗi kernel dò một hướng: hàng giữa/đường chéo mang giá
# trị dương (+2), phần còn lại âm (-1), tổng trọng số bằng 0 nên vùng phẳng
# cho phản hồi xấp xỉ 0 còn đường thẳng đúng hướng cho phản hồi dương lớn.
_LINE_KERNELS: dict[str, np.ndarray] = {
    # Đường nằm ngang: sọc ngang ở hàng giữa.
    "horizontal": np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float64),
    # Đường dọc: sọc dọc ở cột giữa.
    "vertical": np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float64),
    # Đường +45° (từ trên-phải xuống dưới-trái): đường chéo phụ.
    "+45": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float64),
    # Đường -45° (từ trên-trái xuống dưới-phải): đường chéo chính.
    "-45": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float64),
}


def _check_threshold(t: float, name: str = "T") -> float:
    """Kiểm tra ngưỡng: phải là số hữu hạn và không âm."""
    if isinstance(t, bool) or not isinstance(t, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number")
    val = float(t)
    if not np.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return val


def detect_points(gray: np.ndarray, T: float) -> np.ndarray:
    """Phát hiện điểm đơn độc bằng kernel Laplacian 3×3 ``[1 1 1; 1 -8 1; 1 1 1]``.

    Công thức (Mục 5.2 — Point Detection): quét kernel trên ảnh được phản hồi
    ``R(x,y)``, rồi đánh dấu điểm nếu trị tuyệt đối của phản hồi vượt ngưỡng:

        g(x, y) = 1  nếu  |R(x, y)| > T,   ngược lại g(x, y) = 0.

    Một điểm sáng (hoặc tối) cô lập làm ``|R|`` lớn tại đúng vị trí của nó, còn
    vùng phẳng cho ``|R|`` gần 0.

    Args:
        gray: Ảnh xám uint8 2D (H, W).
        T: Ngưỡng không âm (cùng thang với độ lớn phản hồi).

    Returns:
        Mặt nạ uint8 cùng kích thước, giá trị 255 tại điểm được phát hiện, 0
        còn lại.

    Example:
        >>> img = np.zeros((5, 5), dtype=np.uint8)
        >>> img[2, 2] = 255
        >>> mask = detect_points(img, T=1000)
        >>> mask[2, 2]
        255
    """
    img = require_gray_uint8(gray)
    t = _check_threshold(T)
    # Nhân chập kernel điểm rồi lấy trị tuyệt đối: điểm cô lập cho |R| lớn.
    response = convolve2d(img, _POINT_KERNEL, "replicate")
    magnitude = np.abs(response)
    # Ngưỡng hóa: vượt T mới được coi là điểm (Mục 5.2).
    return np.where(magnitude > t, 255, 0).astype(np.uint8)


def detect_lines(gray: np.ndarray, direction: str, T: float) -> np.ndarray:
    """Phát hiện đường một pixel theo bốn hướng bằng kernel 3×3 tương ứng.

    Mỗi hướng dùng đúng một kernel trong bộ "Line detection kernels" (Mục 5.2).
    Kernel khớp hướng cho phản hồi dương lớn tại các pixel thuộc đường, vùng
    phẳng và hướng khác cho phản hồi gần 0. Kết quả là phản hồi được ngưỡng:

        g(x, y) = 1  nếu  R(x, y) > T,   ngược lại g(x, y) = 0.

    Args:
        gray: Ảnh xám uint8 2D (H, W).
        direction: Một trong ``"horizontal"``, ``"vertical"``, ``"+45"``,
            ``"-45"``. ``"+45"`` dò đường từ trên-phải xuống dưới-trái,
            ``"-45"`` dò đường từ trên-trái xuống dưới-phải.
        T: Ngưỡng không âm.

    Returns:
        Mặt nạ uint8 cùng kích thước, 255 tại pixel thuộc đường được phát
        hiện, 0 còn lại.

    Raises:
        ValueError: nếu ``direction`` không hợp lệ.

    Example:
        >>> img = np.zeros((7, 7), dtype=np.uint8)
        >>> img[3, :] = 255                       # đường ngang ở hàng 3
        >>> mask = detect_lines(img, "horizontal", T=1000)
        >>> mask[3, 3]
        255
    """
    img = require_gray_uint8(gray)
    t = _check_threshold(T)
    if direction not in _LINE_KERNELS:
        raise ValueError(
            "direction must be one of 'horizontal', 'vertical', '+45', '-45'"
        )
    kernel = _LINE_KERNELS[direction]
    # Nhân chập kernel hướng đã chọn; đường đúng hướng cho phản hồi dương lớn.
    response = convolve2d(img, kernel, "replicate")
    return np.where(response > t, 255, 0).astype(np.uint8)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    """Kernel Gauss 2D chuẩn hóa (tích ngoài của hai Gauss 1D)."""
    radius = max(1, int(np.ceil(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    g1 = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    g1 /= g1.sum()
    # Tích ngoài tạo kernel 2D; vì g1 đã chuẩn hóa nên tổng kernel 2D = 1.
    return np.outer(g1, g1)


def canny_edges(gray: np.ndarray, low: float, high: float, sigma: float) -> np.ndarray:
    """Dò biên Canny viết lại từ gốc gồm 4 giai đoạn (Mục 5.2 — Edge Detection).

    Bốn giai đoạn, đều được chú thích tại từng khối trong thân hàm:

    1. **Làm trơn Gauss** với ``sigma`` để giảm nhiễu trước khi lấy đạo hàm.
    2. **Gradient Sobel**: ``mag = sqrt(gx^2 + gy^2)`` và hướng
       ``angle = atan2(gy, gx)``.
    3. **Non-maximum suppression (NMS)** trên 4 hướng đã lượng tử hóa
       (0°, 45°, 90°, 135°): chỉ giữ pixel là cực đại dọc theo hướng gradient.
    4. **Ngưỡng kép + hysteresis**: pixel ``>= high`` là biên chắc chắn, pixel
       ``>= low`` là biên yếu; biên yếu nào nối được tới biên chắc chắn qua
       đường biên yếu (lần theo 8-lân cận) mới được giữ.

    Args:
        gray: Ảnh xám uint8 2D (H, W).
        low: Ngưỡng thấp (đơn vị độ lớn gradient).
        high: Ngưỡng cao; phải ``high >= low``.
        sigma: Độ lệch chuẩn của bộ lọc Gauss, ``sigma > 0``.

    Returns:
        Ảnh biên uint8 cùng kích thước, 255 tại pixel biên, 0 còn lại.

    Raises:
        ValueError: nếu ``low > high``, ``sigma <= 0`` hoặc ngưỡng không hữu hạn.

    Example:
        >>> step = np.zeros((32, 32), dtype=np.uint8)
        >>> step[:, 16:] = 255
        >>> edges = canny_edges(step, low=50, high=150, sigma=1.0)
        >>> edges.dtype
        dtype('uint8')
    """
    img = require_gray_uint8(gray)
    lo = _check_threshold(low, name="low")
    hi = _check_threshold(high, name="high")
    if hi < lo:
        raise ValueError("high must be >= low")
    if isinstance(sigma, bool) or not isinstance(
        sigma, (int, float, np.integer, np.floating)
    ):
        raise TypeError("sigma must be a number")
    s = float(sigma)
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("sigma must be a finite positive number")

    # Giai đoạn 1 — Làm trơn Gauss: giảm nhiễu để gradient không nhầm nhiễu
    # thành biên. Nhân chập trên ảnh float rồi lượng tử về uint8 cho `sobel`.
    smoothed = convolve2d(img.astype(np.float64), _gaussian_kernel(s), "replicate")
    smoothed_u8 = np.clip(np.rint(smoothed), 0, 255).astype(np.uint8)

    # Giai đoạn 2 — Gradient Sobel: tái dùng `sobel` có sẵn trong edges.py.
    gx, gy = sobel(smoothed_u8, "replicate")
    mag = np.hypot(gx.astype(np.float64), gy.astype(np.float64))
    angle = np.rad2deg(np.arctan2(gy.astype(np.float64), gx.astype(np.float64)))

    # Giai đoạn 3 — NMS: lượng tử hóa góc về 4 hướng [0°, 180°).
    angle = np.where(angle < 0, angle + 180.0, angle)
    direction = np.zeros_like(angle, dtype=np.int8)
    direction[(angle < 22.5) | (angle >= 157.5)] = 0  # ngang (đông-tây)
    direction[(angle >= 22.5) & (angle < 67.5)] = 1  # chéo 45°
    direction[(angle >= 67.5) & (angle < 112.5)] = 2  # dọc (bắc-nam)
    direction[(angle >= 112.5) & (angle < 157.5)] = 3  # chéo 135°

    # Đệm -inf để pixel ở biên không bao giờ thắng phép so sánh cực đại.
    pad = np.pad(mag, 1, mode="constant", constant_values=-np.inf)
    up = pad[0:-2, 1:-1]
    down = pad[2:, 1:-1]
    left = pad[1:-1, 0:-2]
    right = pad[1:-1, 2:]
    ul = pad[0:-2, 0:-2]
    ur = pad[0:-2, 2:]
    dl = pad[2:, 0:-2]
    dr = pad[2:, 2:]

    nms = np.zeros_like(mag)
    # Hướng 0: so với hai láng giềng trái/phải (trục ngang của gradient).
    m = (direction == 0) & (mag >= left) & (mag >= right)
    nms[m] = mag[m]
    # Hướng 1: so với láng giềng chéo ur/dl.
    m = (direction == 1) & (mag >= ur) & (mag >= dl)
    nms[m] = mag[m]
    # Hướng 2: so với láng giềng trên/dưới.
    m = (direction == 2) & (mag >= up) & (mag >= down)
    nms[m] = mag[m]
    # Hướng 3: so với láng giềng chéo ul/dr.
    m = (direction == 3) & (mag >= ul) & (mag >= dr)
    nms[m] = mag[m]

    # Giai đoạn 4 — Ngưỡng kép + hysteresis: biên mạnh là hạt nhân, biên yếu
    # chỉ được giữ nếu nối được tới biên mạnh qua chuỗi biên yếu.
    strong = nms >= hi
    weak = (nms >= lo) & ~strong
    edges = strong.copy()

    # Lần vết biên nối (connected edge tracing): xuất phát từ mọi biên mạnh,
    # lan theo 8-lân cận qua các biên yếu để "nối" chúng thành biên.
    stack: list[tuple[int, int]] = list(zip(*np.nonzero(strong)))
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < img.shape[0]
                    and 0 <= nx < img.shape[1]
                    and weak[ny, nx]
                    and not edges[ny, nx]
                ):
                    edges[ny, nx] = True
                    stack.append((ny, nx))

    return np.where(edges, 255, 0).astype(np.uint8)


def hough_lines(
    binary_edges: np.ndarray, theta_step_deg: float, rho_step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Biến đổi Hough tìm đường thẳng trong ảnh biên (Mục 5.2 — Hough transform).

    Dùng tham số hóa ``(rho, theta)``:

        rho = x·cos(theta) + y·sin(theta)

    Mỗi pixel biên bỏ phiếu cho mọi cặp ``(theta, rho)`` mà đường thẳng tương
    ứng đi qua nó; các phiếu dồn vào một accumulator 2D (lược đồ). Đỉnh cục bộ
    của accumulator tương ứng với một đường thẳng được nhiều pixel biên ủng hộ.

    Args:
        binary_edges: Ảnh biên nhị phân 2D (``uint8`` khác 0, hoặc ``bool``);
            pixel khác 0 được coi là điểm biên.
        theta_step_deg: Bước góc ``theta`` (độ); ``theta`` quét trong
            ``[0°, 180°)``.
        rho_step: Bước khoảng cách ``rho`` (pixel); ``rho`` quét trong
            ``[-D, D]`` với ``D = sqrt(H^2 + W^2)``.

    Returns:
        Bộ ba mảng ``(rhos, thetas, votes)``: các đỉnh cục bộ của accumulator,
        sắp xếp theo số phiếu giảm dần. ``rhos`` là khoảng cách (pixel),
        ``thetas`` là góc (radian), ``votes`` là số phiếu. Phần tử đầu tiên là
        đường thẳng được ủng hộ mạnh nhất.

    Raises:
        ValueError: nếu ``theta_step_deg``/``rho_step`` không dương hữu hạn.

    Example:
        >>> edges = np.zeros((32, 32), dtype=np.uint8)
        >>> edges[10, :] = 255
        >>> rho, theta, votes = hough_lines(edges, theta_step_deg=1.0, rho_step=1.0)
        >>> abs(float(rho[0]) - 10.0) < 1.5 and abs(float(np.rad2deg(theta[0])) - 90.0) < 1.0
        True
    """
    arr = np.asarray(binary_edges)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("binary_edges must be a non-empty 2D array")
    if arr.dtype.kind not in "bu":
        raise ValueError("binary_edges must be boolean or uint8")
    if isinstance(theta_step_deg, bool) or not isinstance(
        theta_step_deg, (int, float, np.integer, np.floating)
    ):
        raise TypeError("theta_step_deg must be a number")
    if isinstance(rho_step, bool) or not isinstance(
        rho_step, (int, float, np.integer, np.floating)
    ):
        raise TypeError("rho_step must be a number")
    dtheta = float(theta_step_deg)
    drho = float(rho_step)
    if not np.isfinite(dtheta) or dtheta <= 0.0:
        raise ValueError("theta_step_deg must be a finite positive number")
    if not np.isfinite(drho) or drho <= 0.0:
        raise ValueError("rho_step must be a finite positive number")

    h, w = arr.shape
    ys, xs = np.nonzero(arr)

    # Thang rời rạc cho theta (radian) và rho (pixel).
    theta_rad = np.deg2rad(dtheta)
    n_theta = max(1, round(np.pi / theta_rad))
    thetas = np.arange(n_theta, dtype=np.float64) * theta_rad

    diag = float(np.hypot(h, w))
    n_rho = max(1, int(np.ceil(2.0 * diag / drho)) + 1)
    rhos = -diag + np.arange(n_rho, dtype=np.float64) * drho

    # Accumulator: đếm phiếu cho từng ô (rho, theta).
    acc = np.zeros((n_rho, n_theta), dtype=np.int64)
    for y, x in zip(ys.tolist(), xs.tolist()):
        rho_vals = x * np.cos(thetas) + y * np.sin(thetas)
        rho_idx = np.rint((rho_vals + diag) / drho).astype(np.intp)
        rho_idx = np.clip(rho_idx, 0, n_rho - 1)
        # Cộng 1 phiếu cho mỗi ô (rho, theta) mà đường đi qua điểm (x, y).
        np.add.at(acc, (rho_idx, np.arange(n_theta)), 1)

    # Trích đỉnh: một ô là đỉnh nếu lớn hơn hoặc bằng 8 láng giềng và có phiếu.
    acc_pad = np.pad(acc, 1, mode="constant", constant_values=0)
    center = acc_pad[1:-1, 1:-1]
    is_peak = center > 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            is_peak &= (
                center >= acc_pad[1 + dy : 1 + dy + n_rho, 1 + dx : 1 + dx + n_theta]
            )

    r_idx, t_idx = np.nonzero(is_peak)
    votes = acc[r_idx, t_idx]
    # Sắp theo số phiếu giảm dần để đường mạnh nhất nằm ở đầu mảng.
    order = np.argsort(votes)[::-1]
    return (
        rhos[r_idx[order]],
        thetas[t_idx[order]],
        votes[order],
    )


# ---------------------------------------------------------------------------
# Mục 5.3 — Ngưỡng (Thresholding)
# ---------------------------------------------------------------------------


def iterative_global_threshold(
    gray: np.ndarray, eps: float = 0.5
) -> tuple[float, np.ndarray]:
    """Ngưỡng toàn cục cơ bản bằng vòng lặp (Mục 5.3 — Basic Global Thresholding).

    Thuật toán (theo slide "Global Thresholding — Basic"):

    1. Khởi tạo ``T0`` = trung bình cường độ toàn ảnh.
    2. Tách pixel thành hai nhóm: nhóm vật thể ``G1 = {pixel > T}`` và nhóm nền
       ``G2 = {pixel <= T}``.
    3. Tính trung bình ``m1 = mean(G1)``, ``m2 = mean(G2)``.
    4. Cập nhật ``T = (m1 + m2) / 2``.
    5. Lặp lại bước 2–4 đến khi ``|T_mới - T_cũ| < eps``.

    Args:
        gray: Ảnh xám uint8 2D (H, W).
        eps: Sai số dừng (dương, hữu hạn). Mặc định 0.5.

    Returns:
        Bộ ``(T, binary)``: ``T`` là ngưỡng hội tụ (float), ``binary`` là ảnh
        nhị phân uint8 (255 nếu ``pixel > T``, ngược lại 0).

    Raises:
        ValueError: nếu ``eps`` không dương hữu hạn.

    Example:
        >>> img = np.zeros((8, 8), dtype=np.uint8)
        >>> img[:, :4] = 60
        >>> img[:, 4:] = 180
        >>> T, binary = iterative_global_threshold(img, eps=0.5)
        >>> 60.0 < T < 180.0
        True
    """
    img = require_gray_uint8(gray)
    if isinstance(eps, bool) or not isinstance(
        eps, (int, float, np.integer, np.floating)
    ):
        raise TypeError("eps must be a number")
    e = float(eps)
    if not np.isfinite(e) or e <= 0.0:
        raise ValueError("eps must be a finite positive number")

    data = img.astype(np.float64).ravel()
    # Bước 1: T0 = trung bình toàn ảnh.
    t = float(data.mean())
    # Giới hạn vòng lặp để tránh dao động bệnh lý trên dữ liệu rời rạc.
    for _ in range(1000):
        g1 = data[data > t]  # nhóm vật thể (sáng hơn ngưỡng)
        g2 = data[data <= t]  # nhóm nền (tối hơn hoặc bằng ngưỡng)
        # Nếu một nhóm rỗng (ảnh hằng) thì không tách được nữa → dừng.
        if g1.size == 0 or g2.size == 0:
            break
        new_t = 0.5 * (float(g1.mean()) + float(g2.mean()))
        # Bước 5: dừng khi thay đổi nhỏ hơn eps.
        if abs(new_t - t) < e:
            t = new_t
            break
        t = new_t

    # Bước cuối: ngưỡng hóa "pixel > T -> trắng, ngược lại đen".
    binary = np.where(img > t, 255, 0).astype(np.uint8)
    return t, binary


# ---------------------------------------------------------------------------
# Mục 5.4 — Phân đoạn theo vùng (Region-Based Segmentation)
# ---------------------------------------------------------------------------


def region_growing(
    gray: np.ndarray,
    seeds: Sequence[Sequence[int]],
    tol: float,
    connectivity: int = 8,
) -> np.ndarray:
    """Tăng trưởng vùng (Region Growing) theo độ tương đồng với hạt giống.

    Mỗi hạt giống ``(row, col)`` mở một vùng; vùng mở rộng dần theo BFS tới các
    lân cận chưa gán nhãn nếu chúng "đủ giống" hạt giống, tức là:

        |I(ứng viên) - I(hạt giống)| <= tol

    Đây là dạng đơn giản nhất của region growing dùng vị từ tương đồng với hạt
    giống (seed-similarity predicate) như công thức vị từ ``P`` trong phần
    "Basic Formulation" (Mục 5.4). Nhiều hạt giống xử lý lần lượt: pixel đã được
    vùng trước chiếm thì vùng sau bỏ qua (đúng tính chất "các vùng rời nhau").

    Args:
        gray: Ảnh xám uint8 2D (H, W).
        seeds: Dãy các hạt giống, mỗi phần tử là ``(row, col)`` trong ảnh.
        tol: Ngưỡng chênh lệch cường độ (không âm, hữu hạn).
        connectivity: ``4`` hoặc ``8`` lân cận. Mặc định 8.

    Returns:
        Bản đồ nhãn int32 cùng kích thước: ``0`` là nền chưa gán, ``1..N`` là
        nhãn của vùng tương ứng với hạt giống thứ 1..N.

    Raises:
        ValueError: nếu ``connectivity`` không phải 4/8, ``tol`` âm, hoặc
            hạt giống nằm ngoài ảnh.
        TypeError: nếu tọa độ hạt giống không phải số nguyên.

    Example:
        >>> img = np.zeros((9, 9), dtype=np.uint8)
        >>> img[2:7, 2:7] = 100                      # hình vuông đồng nhất
        >>> labels = region_growing(img, [(4, 4)], tol=10, connectivity=8)
        >>> labels[4, 4] == 1 and labels[0, 0] == 0
        True
    """
    img = require_gray_uint8(gray)
    if isinstance(tol, bool) or not isinstance(
        tol, (int, float, np.integer, np.floating)
    ):
        raise TypeError("tol must be a number")
    t = float(tol)
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("tol must be a finite non-negative number")
    if isinstance(connectivity, bool) or not isinstance(
        connectivity, (int, np.integer)
    ):
        raise TypeError("connectivity must be an integer")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    h, w = img.shape
    # Chuẩn hóa và kiểm tra tọa độ hạt giống ngay từ đầu.
    seed_list: list[tuple[int, int]] = []
    for s in seeds:
        if len(s) != 2:
            raise ValueError("each seed must be a (row, col) pair")
        sy, sx = s
        if (
            isinstance(sy, bool)
            or isinstance(sx, bool)
            or not isinstance(sy, (int, np.integer))
            or not isinstance(sx, (int, np.integer))
        ):
            raise TypeError("seed coordinates must be integers")
        sy, sx = int(sy), int(sx)
        if not (0 <= sy < h and 0 <= sx < w):
            raise ValueError("seed coordinates must lie inside the image")
        seed_list.append((sy, sx))
    if not seed_list:
        raise ValueError("at least one seed is required")

    if connectivity == 4:
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        offsets = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

    labels = np.zeros((h, w), dtype=np.int32)
    for label, (sy, sx) in enumerate(seed_list, start=1):
        if labels[sy, sx] != 0:
            continue  # hạt giống nằm trong vùng đã mọc trước đó → bỏ qua
        seed_val = int(img[sy, sx])
        queue: deque[tuple[int, int]] = deque([(sy, sx)])
        labels[sy, sx] = label
        while queue:
            y, x = queue.popleft()
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if labels[ny, nx] != 0:
                    continue
                # Vị từ tương đồng: chênh lệch với hạt giống nằm trong tol.
                if abs(int(img[ny, nx]) - seed_val) <= t:
                    labels[ny, nx] = label
                    queue.append((ny, nx))
    return labels


# ---------------------------------------------------------------------------
# Mục 5.5 — Phân đoạn bằng các tiếp cận mới (K-means, Morphological Watersheds)
# ---------------------------------------------------------------------------


def kmeans_segmentation(
    img: np.ndarray, k: int, seed: int, max_iter: int
) -> tuple[np.ndarray, np.ndarray]:
    """Phân cụm K-means (Lloyd) cho pixel theo mức xám hoặc màu.

    Thuộc nhóm "Segmentation by some new approaches" (Mục 5.5). Mỗi pixel là một
    vector đặc trưng (1 chiều với ảnh xám, 3 chiều với ảnh màu). Vòng lặp Lloyd:

    1. Khởi tạo ``k`` tâm bằng cách chọn ngẫu nhiên (không lặp) ``k`` pixel từ
       bộ sinh ``numpy.random.Generator`` được seed — đảm bảo tất định.
    2. Gán mỗi pixel về tâm gần nhất (khoảng cách Euclid).
    3. Cập nhật mỗi tâm = trung bình các pixel thuộc cụm đó.
    4. Lặp lại 2–3 cho tới khi nhãn không đổi hoặc hết ``max_iter``.

    Args:
        img: Ảnh uint8 2D (xám) hoặc H×W×3 (màu).
        k: Số cụm (``k >= 2`` và không vượt số pixel).
        seed: Seed nguyên cho bộ sinh khởi tạo (tất định).
        max_iter: Số vòng lặp Lloyd tối đa (``>= 1``).

    Returns:
        Bộ ``(labels, centres)``: ``labels`` là bản đồ nhãn int32 cùng kích
        thước không gian của ảnh (nhãn ``0..k-1``), ``centres`` là mảng
        ``(k, d)`` float64 chứa tọa độ các tâm (``d = 1`` cho xám, ``3`` cho màu).

    Raises:
        ValueError: nếu ``k``/``max_iter`` không hợp lệ.
        TypeError: nếu ``k``/``max_iter``/``seed`` không phải số nguyên.

    Example:
        >>> img = np.zeros((16, 16), dtype=np.uint8)
        >>> img[:, :8] = 50
        >>> img[:, 8:] = 200
        >>> labels, centres = kmeans_segmentation(img, k=2, seed=1, max_iter=20)
        >>> sorted(np.rint(centres[:, 0]).astype(int))
        [50, 200]
    """
    source = require_uint8_image(img)
    for name, val in (("k", k), ("max_iter", max_iter), ("seed", seed)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    kk = int(k)
    iters = int(max_iter)
    sd = int(seed)
    if kk < 2:
        raise ValueError("k must be >= 2")
    if iters < 1:
        raise ValueError("max_iter must be >= 1")

    # Trải phẳng pixel thành ma trận (N, d): xám d=1, màu d=3.
    if source.ndim == 3:
        data = source.reshape(-1, 3).astype(np.float64)
    else:
        data = source.reshape(-1, 1).astype(np.float64)
    n = data.shape[0]
    if kk > n:
        raise ValueError("k must not exceed the number of pixels")

    rng = np.random.default_rng(sd)
    # Bước 1: khởi tạo tâm từ k GIÁ TRỊ pixel phân biệt (forgy). Lấy theo giá trị
    # thay vì theo chỉ số để tránh hai tâm khởi tạo trùng giá trị (vd ảnh chỉ có
    # hai mức xám 50/200), gây ra cụm rỗng và nhãn dính về một cụm duy nhất.
    unique_vals = np.unique(data, axis=0)
    if unique_vals.shape[0] < kk:
        raise ValueError("image has fewer than k distinct pixel values")
    init_idx = rng.choice(unique_vals.shape[0], size=kk, replace=False)
    centres = unique_vals[init_idx].copy()

    labels_flat = np.zeros(n, dtype=np.intp)
    for _ in range(iters):
        # Bước 2: gán về tâm gần nhất theo khoảng cách Euclid bình phương.
        dists = np.stack([((data - c) ** 2).sum(axis=1) for c in centres], axis=1)
        new_labels = np.argmin(dists, axis=1)
        # Bước 3: cập nhật tâm = trung bình từng cụm; cụm rỗng thì gán lại một
        # pixel ngẫu nhiên (seed vẫn tất định) để không kẹt.
        new_centres = np.empty_like(centres)
        for c in range(kk):
            pts = data[new_labels == c]
            if pts.shape[0] == 0:
                # Cụm rỗng: gán lại một giá trị pixel phân biệt ngẫu nhiên
                # (vẫn tất định theo seed) để không kẹt ở cụm trống.
                new_centres[c] = unique_vals[int(rng.integers(unique_vals.shape[0]))]
            else:
                new_centres[c] = pts.mean(axis=0)
        # Bước 4: dừng sớm nếu nhãn không thay đổi.
        if np.array_equal(new_labels, labels_flat):
            centres = new_centres
            labels_flat = new_labels
            break
        centres = new_centres
        labels_flat = new_labels

    labels = labels_flat.reshape(source.shape[:2]).astype(np.int32)
    return labels, centres


def simple_watershed(gray_or_gradient: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Watershed điều khiển bằng marker, làm ngập bằng hàng đợi ưu tiên (heapq).

    Mục 5.5 — Segmentation by Morphological Watersheds. Ảnh được xem như một bề
    mặt địa hình: cường độ thấp là "lòng chảo" (regional minima) sẽ được "ngập
    nước". Đây là bản marker-controlled cổ điển (kiểu Meyer), KHÔNG phải bản sao
    chính xác từng pixel của ``cv2.watershed``:

    * Pixel mang nhãn dương trong ``markers`` là mầm (hạt) của từng lưu vực.
    * Mầm được đẩy vào một min-heap theo cường độ; mỗi lần lấy pixel thấp nhất
      ra và lan nhãn của nó tới các lân cận chưa gán (nước dâng dần từ thấp lên).
    * Khi hai lưu vực chạm nhau, pixel tiếp giáp được đánh dấu ``-1`` — đó là
      đường phân thủy (watershed line).

    Args:
        gray_or_gradient: Ảnh 2D số (uint8 hoặc float) đóng vai trò bề mặt để
            ngập; thường dùng ảnh gradient để đường phân thủy bám vào biên.
        markers: Bản đồ marker int32 cùng kích thước: ``0`` = chưa gán nhãn
            (có thể bị ngập), nhãn dương = mầm của lưu vực.

    Returns:
        Bản đồ nhãn int32 cùng kích thước: nhãn dương = lưu vực, ``0`` = pixel
        chưa được ngập, ``-1`` = đường phân thủy.

    Raises:
        ValueError: nếu ảnh/marker sai chiều, marker có giá trị âm, hoặc không
            có mầm dương nào.

    Example:
        >>> surf = np.array([[3, 2, 3], [2, 1, 2], [3, 2, 3]], dtype=np.float64)
        >>> mk = np.zeros((3, 3), dtype=np.int32)
        >>> mk[1, 0] = 1
        >>> mk[1, 2] = 2
        >>> labels = simple_watershed(surf, mk)
        >>> labels[1, 0] == 1 and labels[1, 2] == 2
        True
    """
    surface = np.asarray(gray_or_gradient)
    if surface.ndim != 2 or surface.size == 0:
        raise ValueError("gray_or_gradient must be a non-empty 2D array")
    if surface.dtype.kind not in "biuf":
        raise ValueError("gray_or_gradient must be numeric")
    surf = surface.astype(np.float64)
    if not np.all(np.isfinite(surf)):
        raise ValueError("gray_or_gradient must be finite")

    mk = np.asarray(markers)
    if mk.ndim != 2 or mk.shape != surface.shape:
        raise ValueError("markers must be a 2D array with the same shape as the image")
    if mk.dtype.kind not in "iu":
        raise ValueError("markers must be an integer label map")
    if np.any(mk < 0):
        raise ValueError("markers must contain only non-negative labels")

    h, w = surface.shape
    labels = mk.astype(np.int32).copy()
    seed_mask = labels > 0
    if not seed_mask.any():
        raise ValueError("at least one positive marker (seed) is required")

    # Hàng đợi ưu tiên theo cường độ: nước dâng từ pixel thấp nhất trước.
    heap: list[tuple[float, int, int]] = []
    for y, x in zip(*np.nonzero(seed_mask)):
        heapq.heappush(heap, (surf[y, x], int(y), int(x)))

    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    while heap:
        _, y, x = heapq.heappop(heap)
        lab = int(labels[y, x])
        if lab <= 0:
            continue  # pixel đã thành đường phân thủy → bỏ qua
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if labels[ny, nx] == 0:
                # Lân cận chưa gán: nước của lưu vực hiện tại tràn sang.
                labels[ny, nx] = lab
                heapq.heappush(heap, (surf[ny, nx], ny, nx))
            elif labels[ny, nx] > 0 and labels[ny, nx] != lab and not seed_mask[ny, nx]:
                # Hai lưu vực khác nhãn gặp nhau → đắp đập (đường phân thủy).
                labels[ny, nx] = -1
    return labels
