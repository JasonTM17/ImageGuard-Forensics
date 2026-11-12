"""Lọc ảnh trong miền tần số (Frequency-domain filtering) — NumPy only.

Module này hiện thực **Chương 3 — Image Enhancement in the Frequency Domain**
(tài liệu *Digital Image Processing*, PGS.TS. Hoàng Văn Dũng, HCMUTE): biến đổi
Fourier rời rạc 2D (DFT), phổ biên độ để hiển thị, các bộ lọc thông thấp
(lowpass) lý tưởng / Butterworth / Gaussian, các bộ lọc thông cao (highpass)
tương ứng, bộ lọc Laplacian trong miền tần số, và các bước triển khai lọc
(padding, dịch tâm, nhân, biến đổi ngược, cắt, kẹp về uint8).

Chỉ dùng NumPy cho phép toán; ``np.fft`` được phép dùng riêng cho chính phép
biến đổi (``fft2`` / ``ifft2`` / ``fftshift`` / ``ifftshift``). Định nghĩa DFT
được nêu trong docstring của ``dft2_centered``.

Module này chỉ phục vụ demo/học tập — **không** được import vào detector.
"""

from __future__ import annotations

import numpy as np

from imageguard.ipkit._util import require_gray_uint8


def _check_shape(shape: tuple[int, int] | list[int]) -> tuple[int, int]:
    """Kiểm tra ``shape`` là cặp (rows, cols) gồm hai số nguyên dương."""
    if isinstance(shape, (bool, np.bool_)) or not isinstance(shape, (tuple, list)):
        raise TypeError("shape must be a (rows, cols) tuple or list")
    if len(shape) != 2:
        raise ValueError("shape must be a (rows, cols) pair")
    m, n = shape
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)):
        raise TypeError("shape rows must be an integer")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise TypeError("shape cols must be an integer")
    if m <= 0 or n <= 0:
        raise ValueError("shape entries must be positive")
    return int(m), int(n)


def _check_d0(d0: float) -> float:
    """Kiểm tra ``d0`` (tần số cắt) là số thực hữu hạn và dương."""
    if isinstance(d0, (bool, np.bool_)) or not isinstance(
        d0, (int, float, np.integer, np.floating)
    ):
        raise TypeError("d0 must be a number")
    d = float(d0)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("d0 must be a finite positive number")
    return d


def _check_order(order: int) -> int:
    """Kiểm tra ``order`` (bậc Butterworth) là số nguyên >= 1."""
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise TypeError("order must be an integer")
    n = int(order)
    if n < 1:
        raise ValueError("order must be >= 1")
    return n


def _check_gain(k: float) -> float:
    """Kiểm tra ``k`` (hệ số khuếch đại tần số cao) là số thực hữu hạn >= 0."""
    if isinstance(k, (bool, np.bool_)) or not isinstance(
        k, (int, float, np.integer, np.floating)
    ):
        raise TypeError("k must be a number")
    g = float(k)
    if not np.isfinite(g) or g < 0.0:
        raise ValueError("k must be a finite non-negative number")
    return g


def dft2_centered(gray: np.ndarray) -> np.ndarray:
    """Biến đổi Fourier rời rạc 2D đã dịch tâm (phổ phức, tâm là DC).

    Áp dụng định nghĩa DFT 2D của bài giảng (mục "2D DFT and IDFT"):

        F(u, v) = sum_x sum_y f(x, y) * exp(-j * 2*pi * (u*x/M + v*y/N))

    với ``u, v`` là biến tần số, ``x, y`` là biến không gian. ``np.fft.fft2``
    thực hiện chính công thức trên (không chuẩn hoá), sau đó ``fftshift`` đưa
    thành phần một chiều (DC, tần số 0) về tâm khung ảnh để dễ quan sát và lọc.

    Tham số:
        gray: ảnh xám 2D ``uint8`` (M x N).

    Trả về:
        ``np.ndarray`` phức ``complex128`` (M x N): phổ đã dịch tâm, DC nằm tại
        ``(M//2, N//2)``.

    Ví dụ::

        F = dft2_centered(gray)
        F[F.shape[0] // 2, F.shape[1] // 2]  # ~ tổng mức xám (thành phần DC)
    """
    img = require_gray_uint8(gray)
    # fft2 tính DFT 2D không chuẩn hoá; chuyển sang float64 tránh tràn kiểu uint8
    F = np.fft.fft2(img.astype(np.float64))
    # fftshift đưa DC (tần số 0) về tâm để phổ đối xứng và dễ đọc
    return np.fft.fftshift(F)


def magnitude_spectrum(gray: np.ndarray, log: bool = True) -> np.ndarray:
    """Phổ biên độ ``|F(u,v)|`` đã dịch tâm, chuẩn hoá về ``uint8`` để hiển thị.

    Theo đoạn mã OpenCV của bài giảng (mục "OpenCV in use"), phổ biên độ được
    nén bằng ``20 * log(|F|)`` (logarit tự nhiên) rồi co giãn về ``[0, 255]``.
    Ở đây dùng ``np.log1p(|F|) = log(1 + |F|)`` để tránh ``log(0) = -inf`` ở
    các thành phần tần số có biên độ bằng 0.

    Tham số:
        gray: ảnh xám 2D ``uint8``.
        log: nếu ``True`` (mặc định) dùng thang logarit (dễ thấy chi tiết yếu);
            nếu ``False`` dùng thang tuyến tính của ``|F|``.

    Trả về:
        ``np.ndarray`` ``uint8`` (M x N): ảnh phổ để hiển thị, ``0..255``.

    Ví dụ::

        spec = magnitude_spectrum(gray)          # thang logarit
        spec_lin = magnitude_spectrum(gray, False)
    """
    if not isinstance(log, (bool, np.bool_)):
        raise TypeError("log must be a bool")
    mag = np.abs(dft2_centered(gray))
    if log:
        # log1p nén dải động rất lớn của |F| (DC có thể lớn gấp hàng nghìn lần)
        mag = 20.0 * np.log1p(mag)
    lo = float(mag.min())
    hi = float(mag.max())
    if hi <= lo:
        # Ảnh hằng/ảnh rỗng: mọi biên độ bằng nhau -> phổ trả về toàn 0
        return np.zeros(mag.shape, dtype=np.uint8)
    # Chuẩn hoá min-max về [0, 255] để hiển thị; kẹp + làm tròn về uint8
    scaled = (mag - lo) / (hi - lo) * 255.0
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def distance_grid(shape: tuple[int, int] | list[int]) -> np.ndarray:
    """Khoảng cách ``D(u, v)`` từ mỗi điểm ``(u, v)`` tới tâm khung tần số.

    Theo bài giảng (mục "Ideal Lowpass Filters"):

        D(u, v) = sqrt((u - M/2)^2 + (v - N/2)^2)

    với ``(M, N)`` là kích thước khung, tâm đặt tại ``(M/2, N/2)``. Dùng phép
    chia thực ``M/2`` (không phải ``M//2``) để đúng với công thức slide; với
    kích thước chẵn (thường gặp sau padding) thì ``M/2 == M//2`` nên trùng với
    vị trí DC của ``fftshift``.

    Tham số:
        shape: cặp ``(rows, cols)`` kích thước khung tần số.

    Trả về:
        ``np.ndarray`` ``float64`` (rows x cols): giá trị ``D(u, v)``.

    Ví dụ::

        D = distance_grid((8, 8))
        D[4, 4]   # 0.0  (tâm)
        D[4, 7]   # 3.0  (cách tâm 3 cột)
    """
    rows, cols = _check_shape(shape)
    # Toạ độ u, v dịch về gốc tâm (M/2, N/2); dùng float để tâm có thể lệch nửa pixel
    u = np.arange(rows, dtype=np.float64)[:, None] - rows / 2.0
    v = np.arange(cols, dtype=np.float64)[None, :] - cols / 2.0
    # Khoảng cách Euclid tới tâm theo định nghĩa của bài giảng
    return np.sqrt(u * u + v * v)


def ideal_lowpass(shape: tuple[int, int] | list[int], d0: float) -> np.ndarray:
    """Bộ lọc thông thấp lý tưởng (ILPF): giữ nguyên tần số có ``D <= d0``.

    Hàm truyền đạt theo bài giảng (mục "Ideal Lowpass Filters"):

        H(u, v) = 1  nếu  D(u, v) <= D0
        H(u, v) = 0  nếu  D(u, v) >  D0

    Bộ lọc này "cắt" toàn bộ tần số cao ngoài bán kính ``d0`` (tạo hiện tượng
    ringing khi lọc ngược vì hàm truyền đạt gián đoạn).

    Tham số:
        shape: cặp ``(rows, cols)`` kích thước khung tần số.
        d0: tần số cắt ``D0`` (dương, đơn vị khoảng cách pixel).

    Trả về:
        ``np.ndarray`` ``float64`` (rows x cols) gồm giá trị ``0`` hoặc ``1``.

    Ví dụ::

        H = ideal_lowpass((64, 64), 10.0)
    """
    d = _check_d0(d0)
    D = distance_grid(shape)
    # Giữ (bằng 1) các điểm nằm trong đĩa bán kính D0, cắt (bằng 0) phần ngoài
    return (D <= d).astype(np.float64)


def butterworth_lowpass(
    shape: tuple[int, int] | list[int], d0: float, order: int
) -> np.ndarray:
    """Bộ lọc thông thấp Butterworth (BLPF) bậc ``n``.

    Hàm truyền đạt theo bài giảng (mục "Butterworth Lowpass Filter"):

        H(u, v) = 1 / (1 + (D(u, v) / D0)^(2n))

    Tại ``D = D0`` thì ``H = 1/2`` với mọi bậc ``n``; ``n`` càng lớn càng tiến
    gần bộ lọc lý tưởng (dốc hơn) nhưng cũng dễ ringing hơn.

    Tham số:
        shape: cặp ``(rows, cols)`` kích thước khung tần số.
        d0: tần số cắt ``D0`` (dương).
        order: bậc ``n`` của bộ lọc (nguyên, ``>= 1``).

    Trả về:
        ``np.ndarray`` ``float64`` (rows x cols), giá trị trong ``(0, 1]``.

    Ví dụ::

        H = butterworth_lowpass((64, 64), 15.0, order=2)
    """
    d = _check_d0(d0)
    n = _check_order(order)
    D = distance_grid(shape)
    # Chuẩn hoá khoảng cách theo D0 rồi luỹ thừa 2n làm mềm vùng cắt
    return 1.0 / (1.0 + (D / d) ** (2 * n))


def gaussian_lowpass(shape: tuple[int, int] | list[int], d0: float) -> np.ndarray:
    """Bộ lọc thông thấp Gaussian (GLPF).

    Hàm truyền đạt theo bài giảng (mục "Gaussian Lowpass Filter"):

        H(u, v) = exp(-D(u, v)^2 / (2 * D0^2))

    Hàm Gaussian trơn đều, không gây ringing (khác với bộ lọc lý tưởng).

    Tham số:
        shape: cặp ``(rows, cols)`` kích thước khung tần số.
        d0: tần số cắt ``D0`` (dương).

    Trả về:
        ``np.ndarray`` ``float64`` (rows x cols), giá trị trong ``(0, 1]``.

    Ví dụ::

        H = gaussian_lowpass((64, 64), 20.0)
    """
    d = _check_d0(d0)
    D = distance_grid(shape)
    # Hàm mũ giảm dần theo bình phương khoảng cách -> triệt tiêu tần số cao mượt mà
    return np.exp(-(D * D) / (2.0 * d * d))


def ideal_highpass(shape: tuple[int, int] | list[int], d0: float) -> np.ndarray:
    """Bộ lọc thông cao lý tưởng (IHPF): phần bù của ILPF.

    Theo bài giảng (mục "Sharpening Filter"), bộ lọc thông cao là phần bù của
    thông thấp: ``H_hp = 1 - H_lp``, nên:

        H(u, v) = 0  nếu  D(u, v) <= D0
        H(u, v) = 1  nếu  D(u, v) >  D0

    Tham số:
        shape: cặp ``(rows, cols)``.
        d0: tần số cắt ``D0`` (dương).

    Trả về:
        ``np.ndarray`` ``float64`` (rows x cols) gồm giá trị ``0`` hoặc ``1``.

    Ví dụ::

        H = ideal_highpass((64, 64), 10.0)
    """
    return 1.0 - ideal_lowpass(shape, d0)


def butterworth_highpass(
    shape: tuple[int, int] | list[int], d0: float, order: int
) -> np.ndarray:
    """Bộ lọc thông cao Butterworth (BHPF) bậc ``n``: phần bù của BLPF.

    Dùng quan hệ ``H_hp = 1 - H_lp`` (bài giảng, mục "Sharpening Filter"), tương
    đương công thức tường minh của slide:

        H(u, v) = 1 / (1 + (D0 / D(u, v))^(2n))

    Tại ``D = D0`` thì ``H = 1/2`` với mọi bậc ``n``.

    Tham số:
        shape: cặp ``(rows, cols)``.
        d0: tần số cắt ``D0`` (dương).
        order: bậc ``n`` (nguyên, ``>= 1``).

    Trả về:
        ``np.ndarray`` ``float64`` (rows x cols), giá trị trong ``[0, 1)``.

    Ví dụ::

        H = butterworth_highpass((64, 64), 15.0, order=2)
    """
    return 1.0 - butterworth_lowpass(shape, d0, order)


def gaussian_highpass(shape: tuple[int, int] | list[int], d0: float) -> np.ndarray:
    """Bộ lọc thông cao Gaussian (GHPF): phần bù của GLPF.

    Dùng quan hệ ``H_hp = 1 - H_lp`` (bài giảng, mục "Sharpening Filter"), tương
    đương công thức tường minh của slide:

        H(u, v) = 1 - exp(-D(u, v)^2 / (2 * D0^2))

    Tham số:
        shape: cặp ``(rows, cols)``.
        d0: tần số cắt ``D0`` (dương).

    Trả về:
        ``np.ndarray`` ``float64`` (rows x cols), giá trị trong ``[0, 1)``.

    Ví dụ::

        H = gaussian_highpass((64, 64), 20.0)
    """
    return 1.0 - gaussian_lowpass(shape, d0)


def laplacian_frequency(shape: tuple[int, int] | list[int]) -> np.ndarray:
    """Bộ lọc Laplacian trong miền tần số.

    Đạo hàm bậc hai (Laplacian) ứng với phép nhân tần số ``-4*pi^2*(u^2 + v^2)``
    trong miền Fourier. Sau khi dịch tâm (bài giảng, mục "Laplacian in Frequency
    Domain"):

        H(u, v) = -4 * pi^2 * D(u, v)^2

    (Slide viết dạng rút gọn ``-(u - M/2)^2 - (v - N/2)^2``, bỏ hằng số ``4*pi^2``
    vì nó chỉ co giãn kết quả và bị triệt tiêu khi chuẩn hoá để hiển thị.)

    Tham số:
        shape: cặp ``(rows, cols)``.

    Trả về:
        ``np.ndarray`` ``float64`` (rows x cols): giá trị ``<= 0``, bằng 0 tại tâm.

    Ví dụ::

        H = laplacian_frequency((64, 64))
    """
    D = distance_grid(shape)
    # Nhân -4*pi^2 biến khoảng cách tần số thành đáp ứng Laplacian liên tục
    return -4.0 * np.pi * np.pi * D * D


def apply_frequency_filter(
    gray: np.ndarray, H: np.ndarray, pad: bool = True
) -> np.ndarray:
    """Lọc ảnh trong miền tần số theo các bước của bài giảng (mục "Implementation").

    Các bước: (1) đệm ảnh lên ``P x Q = 2M x 2N`` bằng 0 để tránh nhiễu "wrap
    around" do tính tuần hoàn ngầm của DFT; (2) tính ``F = fft2`` rồi dịch tâm
    bằng ``fftshift`` (chọn cách dịch tâm ``fftshift`` thay vì nhân
    ``(-1)^(x+y)`` vì rõ ràng và ít lỗi dấu hơn; hai cách cho kết quả tương
    đương); (3) nhân ``G = F_shift * H`` với ``H`` được tính trên lưới đã dịch
    tâm; (4) biến đổi ngược ``ifft2`` sau khi ``ifftshift``; (5) cắt về ``M x N``
    rồi kẹp và làm tròn về ``uint8``.

    Tham số:
        gray: ảnh xám 2D ``uint8`` (M x N).
        H: hàm truyền đạt 2D (số thực, hữu hạn). Phải có kích thước
            ``(2M, 2N)`` khi ``pad=True``, hoặc ``(M, N)`` khi ``pad=False``.
        pad: nếu ``True`` (mặc định) đệm ảnh lên ``2M x 2N`` trước khi biến đổi.

    Trả về:
        ``np.ndarray`` ``uint8`` (M x N): ảnh sau khi lọc.

    Ví dụ::

        M, N = gray.shape
        H = gaussian_lowpass((2 * M, 2 * N), 20.0)  # hàm truyền trên lưới đã đệm
        smooth = apply_frequency_filter(gray, H)
    """
    img = require_gray_uint8(gray)
    M, N = img.shape
    if not isinstance(pad, (bool, np.bool_)):
        raise TypeError("pad must be a bool")
    Harr = np.asarray(H, dtype=np.float64)
    if Harr.ndim != 2 or Harr.size == 0:
        raise ValueError("H must be a non-empty 2D array")
    if not np.all(np.isfinite(Harr)):
        raise ValueError("H must be finite")
    P, Q = (2 * M, 2 * N) if pad else (M, N)
    if Harr.shape != (P, Q):
        raise ValueError(f"H must have shape {(P, Q)} for pad={pad}, got {Harr.shape}")

    # Đệm 0 về P x Q: chống wrap-around của DFT (tuần hoàn ngầm)
    padded = np.zeros((P, Q), dtype=np.float64)
    padded[:M, :N] = img.astype(np.float64)
    # DFT 2D rồi fftshift đưa DC về tâm để khớp lưới của H
    F = np.fft.fftshift(np.fft.fft2(padded))
    # Lọc: nhân phổ với hàm truyền đạt H (miền tần số)
    G = F * Harr
    # ifftshift trả DC về góc, rồi biến đổi ngược để về miền không gian
    g = np.fft.ifft2(np.fft.ifftshift(G))
    # Lấy phần thực (phần ảo ~0 do đối xứng), cắt về M x N, kẹp về [0, 255]
    out = np.real(g)[:M, :N]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def sharpen_frequency(gray: np.ndarray, d0: float, k: float) -> np.ndarray:
    """Làm nét bằng nhấn mạnh tần số cao (high-frequency emphasis).

    Theo phần "Sharpening Filter" của bài giảng, dùng bộ lọc nhấn mạnh tần số
    cao với độ lệch (offset) 1 và hệ số khuếch đại ``k``:

        H_emph(u, v) = 1 + k * H_hp(u, v)

    trong đó ``H_hp`` là bộ lọc thông cao Gaussian ``gaussian_highpass`` (trơn,
    không ringing). Kết quả giữ nguyên phần tần số thấp (DC) và cộng thêm ``k``
    lần thành phần tần số cao, tương tự unsharp masking trong miền tần số.

    Tham số:
        gray: ảnh xám 2D ``uint8``.
        d0: tần số cắt ``D0`` (dương).
        k: hệ số khuếch đại tần số cao (``>= 0``; ``k = 0`` trả về ảnh gốc).

    Trả về:
        ``np.ndarray`` ``uint8`` (M x N): ảnh đã làm nét.

    Ví dụ::

        sharp = sharpen_frequency(gray, d0=20.0, k=1.5)
    """
    img = require_gray_uint8(gray)
    d = _check_d0(d0)
    gain = _check_gain(k)
    M, N = img.shape
    # Hàm truyền đạt phải được tính trên lưới đã đệm 2M x 2N (khớp apply_frequency_filter)
    H_hp = gaussian_highpass((2 * M, 2 * N), d)
    # Nhấn mạnh: giữ tần số thấp (hệ số 1), khuếch đại tần số cao thêm k lần
    H_emph = 1.0 + gain * H_hp
    return apply_frequency_filter(img, H_emph, pad=True)
