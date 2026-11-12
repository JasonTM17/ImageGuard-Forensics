"""ipkit.frequency — DFT, phổ biên độ, các bộ lọc miền tần số (vs NumPy/cv2)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.ipkit.frequency import (
    apply_frequency_filter,
    butterworth_highpass,
    butterworth_lowpass,
    dft2_centered,
    distance_grid,
    gaussian_highpass,
    gaussian_lowpass,
    ideal_highpass,
    ideal_lowpass,
    laplacian_frequency,
    magnitude_spectrum,
    sharpen_frequency,
)


# ---------------------------------------------------------------------------
# DFT và phổ biên độ
# ---------------------------------------------------------------------------
def test_dft2_centered_is_complex_and_dc_at_center() -> None:
    rng = np.random.default_rng(1)
    gray = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    F = dft2_centered(gray)
    assert F.dtype == np.complex128
    assert F.shape == gray.shape
    # Thành phần DC (tần số 0) sau fftshift nằm ở tâm và bằng tổng mức xám
    M, N = gray.shape
    np.testing.assert_allclose(F[M // 2, N // 2], float(gray.sum()), rtol=1e-9)


def test_magnitude_spectrum_range_and_shape() -> None:
    rng = np.random.default_rng(2)
    gray = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    spec = magnitude_spectrum(gray)
    assert spec.dtype == np.uint8
    assert spec.shape == gray.shape
    # Chuẩn hoá min-max: pixel sáng nhất là 255, tối nhất là 0
    assert int(spec.max()) == 255
    assert int(spec.min()) == 0
    # Thang tuyến tính cũng cho uint8 hợp lệ
    spec_lin = magnitude_spectrum(gray, log=False)
    assert spec_lin.dtype == np.uint8
    assert int(spec_lin.max()) == 255


def test_magnitude_spectrum_constant_zero() -> None:
    # Ảnh toàn 0: phổ toàn 0 -> log1p(0)=0 -> không co giãn được, trả về toàn 0
    spec = magnitude_spectrum(np.zeros((8, 8), dtype=np.uint8))
    np.testing.assert_array_equal(spec, np.zeros((8, 8), dtype=np.uint8))


def test_parseval_energy_conservation() -> None:
    rng = np.random.default_rng(3)
    gray = rng.integers(0, 256, (20, 24), dtype=np.uint8)
    F = dft2_centered(gray)
    # Parseval: sum |f|^2 == sum |F|^2 / (M*N) (fft2 không chuẩn hoá)
    energy_spatial = float(np.sum(gray.astype(np.float64) ** 2))
    energy_freq = float(np.sum(np.abs(F) ** 2)) / gray.size
    np.testing.assert_allclose(energy_spatial, energy_freq, rtol=1e-9)


# ---------------------------------------------------------------------------
# Lưới khoảng cách và hàm truyền đạt
# ---------------------------------------------------------------------------
def test_distance_grid_center_and_values() -> None:
    D = distance_grid((8, 8))
    assert D.shape == (8, 8)
    assert D[4, 4] == 0.0  # tâm có khoảng cách 0
    assert D[4, 7] == 3.0  # cách tâm 3 cột
    assert D[0, 4] == 4.0  # cách tâm 4 hàng
    # Đối xứng quanh tâm
    assert D[5, 4] == D[3, 4]


def test_lowpass_center_is_one_highpass_center_is_zero() -> None:
    shape = (8, 8)
    # Tâm (D=0): LP giữ lại (=1), HP chặn (=0)
    assert ideal_lowpass(shape, 5.0)[4, 4] == 1.0
    assert butterworth_lowpass(shape, 5.0, 2)[4, 4] == 1.0
    assert gaussian_lowpass(shape, 5.0)[4, 4] == 1.0
    assert ideal_highpass(shape, 5.0)[4, 4] == 0.0
    assert butterworth_highpass(shape, 5.0, 2)[4, 4] == 0.0
    assert gaussian_highpass(shape, 5.0)[4, 4] == 0.0


def test_ideal_lowpass_cutoff() -> None:
    H = ideal_lowpass((8, 8), 3.0)
    # Trong đĩa bán kính D0 (D<=3) -> 1; ngoài (D>3) -> 0
    assert H[4, 7] == 1.0  # D = 3
    assert H[4, 4] == 1.0  # D = 0
    assert H[0, 0] == 0.0  # D = sqrt(16+16) > 3


def test_butterworth_half_at_d0_any_order() -> None:
    # Pixel (4,7) có D = 3 = D0 -> H = 1/(1+1^2n) = 0.5 với mọi bậc n
    for order in (1, 2, 5):
        H = butterworth_lowpass((8, 8), 3.0, order)
        assert H[4, 7] == pytest.approx(0.5, abs=1e-12)


def test_gaussian_lowpass_decays_and_bounded() -> None:
    H = gaussian_lowpass((16, 16), 4.0)
    assert H[8, 8] == pytest.approx(1.0, abs=1e-12)  # tâm = 1
    assert np.all(H > 0.0) and np.all(H <= 1.0)  # nằm trong (0, 1]
    # Đơn điệu giảm theo khoảng cách: xa tâm thì nhỏ hơn
    assert H[8, 12] < H[8, 9] < H[8, 8]


def test_highpass_is_one_minus_lowpass() -> None:
    shape = (9, 9)
    np.testing.assert_allclose(
        ideal_highpass(shape, 4.0), 1.0 - ideal_lowpass(shape, 4.0)
    )
    np.testing.assert_allclose(
        butterworth_highpass(shape, 4.0, 3),
        1.0 - butterworth_lowpass(shape, 4.0, 3),
    )
    np.testing.assert_allclose(
        gaussian_highpass(shape, 4.0), 1.0 - gaussian_lowpass(shape, 4.0)
    )


def test_laplacian_frequency_formula() -> None:
    shape = (8, 8)
    H = laplacian_frequency(shape)
    D = distance_grid(shape)
    # H = -4*pi^2 * D^2
    np.testing.assert_allclose(H, -4.0 * np.pi**2 * D**2, atol=1e-12)
    assert H[4, 4] == pytest.approx(0.0, abs=1e-12)  # tâm = 0
    assert np.all(H <= 0.0)  # luôn không dương


# ---------------------------------------------------------------------------
# Lọc miền tần số
# ---------------------------------------------------------------------------
def test_ideal_lowpass_huge_d0_is_identity() -> None:
    rng = np.random.default_rng(4)
    gray = rng.integers(0, 256, (24, 32), dtype=np.uint8)
    # D0 rất lớn -> H = 1 toàn lưới (kể cả lưới đã đệm) -> lọc = đồng nhất
    H = ideal_lowpass((48, 64), 1e9)
    out = apply_frequency_filter(gray, H)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, gray)


def test_apply_frequency_filter_no_pad_identity() -> None:
    rng = np.random.default_rng(5)
    gray = rng.integers(0, 256, (9, 11), dtype=np.uint8)
    # pad=False: H phải có kích thước đúng bằng ảnh
    H = ideal_lowpass(gray.shape, 1e9)
    out = apply_frequency_filter(gray, H, pad=False)
    np.testing.assert_array_equal(out, gray)


def test_apply_frequency_filter_1x1_identity() -> None:
    gray = np.array([[128]], dtype=np.uint8)
    H = ideal_lowpass((2, 2), 1e9)  # lưới đệm 2x2 cho ảnh 1x1
    out = apply_frequency_filter(gray, H)
    np.testing.assert_array_equal(out, gray)


def test_highpass_constant_image_is_zero() -> None:
    gray = np.full((16, 16), 128, dtype=np.uint8)
    # Ảnh hằng chỉ có thành phần DC; HP chặn đúng DC (tâm) -> kết quả ~0.
    # Dùng pad=False vì đệm 0 tạo cạnh 128->0 quanh ảnh, chính HP sẽ giữ lại
    # cạnh đó; khung không đệm mới thực sự "chỉ có DC".
    H = ideal_highpass((16, 16), 5.0)
    out = apply_frequency_filter(gray, H, pad=False)
    assert int(np.max(np.abs(out.astype(np.float64)))) == 0


def test_gaussian_lowpass_reduces_variance() -> None:
    rng = np.random.default_rng(123)
    gray = rng.integers(0, 256, (48, 48), dtype=np.uint8)
    H = gaussian_lowpass((96, 96), 8.0)
    out = apply_frequency_filter(gray, H)
    var_in = float(gray.astype(np.float64).var())
    var_out = float(out.astype(np.float64).var())
    # Làm mượt làm giảm phương sai của nhiễu trắng một cách rõ rệt
    assert var_out < 0.5 * var_in


def test_apply_frequency_filter_deterministic() -> None:
    rng = np.random.default_rng(6)
    gray = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    H = gaussian_lowpass((32, 32), 10.0)
    np.testing.assert_array_equal(
        apply_frequency_filter(gray, H), apply_frequency_filter(gray, H)
    )


def test_gaussian_lowpass_vs_spatial_gaussian() -> None:
    # Bộ lọc Gaussian LP tần số H = exp(-D^2/(2 D0^2)) trên lưới P x P tương
    # ứng với Gaussian không gian có sigma = P / (2*pi*D0) (đạo hàm từ cặp
    # biến đổi Fourier của hàm Gaussian). So sánh lỏng vì: (a) DFT là tích chập
    # vòng còn cv2.GaussianBlur là tích chập tuyến tính, (b) cv2 cắt kernel ở
    # ~±3 sigma, (c) biên độ bị ảnh hưởng bởi biên — nên chỉ so sánh vùng lõi.
    M, d0 = 64, 5.0
    P = 2 * M
    sigma = P / (2.0 * np.pi * d0)  # ~4.07 px
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, (M, M), dtype=np.uint8)
    H = gaussian_lowpass((P, P), d0)
    out = apply_frequency_filter(gray, H)
    ref = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    b = int(np.ceil(3.0 * sigma)) + 1  # loại bỏ vùng biên chịu hiệu ứng biên
    sl = slice(b, M - b)
    mad = float(
        np.mean(np.abs(out[sl, sl].astype(np.float64) - ref[sl, sl].astype(np.float64)))
    )
    # Dung sai lỏng (mức xám/255): hai cách lọc là xấp xỉ chứ không đồng nhất
    assert mad < 6.0


def test_sharpen_frequency_identity_at_k0_and_shape() -> None:
    rng = np.random.default_rng(8)
    gray = rng.integers(0, 256, (24, 24), dtype=np.uint8)
    out = sharpen_frequency(gray, d0=15.0, k=0.0)
    assert out.dtype == np.uint8
    assert out.shape == gray.shape
    # k=0 -> H_emph = 1 -> kết quả đồng nhất với ảnh gốc
    np.testing.assert_array_equal(out, gray)


def test_sharpen_frequency_increases_high_frequency_content() -> None:
    # Ảnh có cạnh rõ: làm nét (k>0) phải khác k=0 và giữ nguyên mức xám trung bình
    y = np.linspace(0, 255, 32, dtype=np.float64)
    ramp = np.tile(y[:, None], (1, 32)).astype(np.uint8)
    base = sharpen_frequency(ramp, d0=10.0, k=0.0)
    sharp = sharpen_frequency(ramp, d0=10.0, k=2.0)
    assert np.any(sharp != base)  # thực sự thay đổi ảnh


# ---------------------------------------------------------------------------
# Kiểm tra đầu vào sai
# ---------------------------------------------------------------------------
def test_frequency_validation_errors() -> None:
    gray = np.zeros((8, 8), dtype=np.uint8)
    # Sai kiểu ảnh / shape
    with pytest.raises(ValueError, match="uint8"):
        dft2_centered(np.ones((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="2D"):
        dft2_centered(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="positive"):
        distance_grid((0, 5))
    with pytest.raises(TypeError, match="tuple"):
        distance_grid("x")
    # d0 không hợp lệ
    with pytest.raises(ValueError, match="positive"):
        ideal_lowpass((8, 8), 0.0)
    with pytest.raises(ValueError, match="positive"):
        gaussian_lowpass((8, 8), -3.0)
    with pytest.raises(TypeError, match="number"):
        ideal_lowpass((8, 8), "5")
    # order không hợp lệ
    with pytest.raises(ValueError, match=">= 1"):
        butterworth_lowpass((8, 8), 5.0, 0)
    with pytest.raises(TypeError, match="integer"):
        butterworth_lowpass((8, 8), 5.0, 1.5)
    # H sai kích thước / không hữu hạn (gray 8x8, pad=True cần H 16x16)
    with pytest.raises(ValueError, match="shape"):
        apply_frequency_filter(gray, np.ones((8, 8)))
    with pytest.raises(ValueError, match="finite"):
        apply_frequency_filter(gray, np.full((16, 16), np.inf))
    # sharpen: k âm
    with pytest.raises(ValueError, match="non-negative"):
        sharpen_frequency(gray, d0=10.0, k=-1.0)
