"""Toán tử tạo nhiễu tổng hợp trên ảnh (chỉ NumPy).

Mô phỏng nhiễu trong bài lab: muối-tiêu, speckle (nhân), Poisson (shot noise).
Chủ đề: mô hình nhiễu trước khi lọc không gian (C2) hoặc khử nhiễu.
"""

from __future__ import annotations

import numpy as np

from imageguard.ipkit._util import apply_per_channel, require_uint8_image


def _check_amount(amount: float) -> float:
    """Kiểm tra xác suất nhiễu muối-tiêu nằm trong [0, 1]."""
    a = float(amount)
    if not np.isfinite(a) or a < 0.0 or a > 1.0:
        raise ValueError("amount must be in [0, 1]")
    return a


def _check_sigma(sigma: float) -> float:
    """Kiểm tra độ lệch chuẩn speckle không âm và hữu hạn."""
    s = float(sigma)
    if not np.isfinite(s) or s < 0.0:
        raise ValueError("sigma must be a finite non-negative number")
    return s


def _check_seed(seed: int) -> int:
    """Đảm bảo seed là số nguyên (không nhận bool) để RNG tái lập được."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    return int(seed)


def salt_pepper(img: np.ndarray, amount: float, seed: int) -> np.ndarray:
    """Nhiễu muối-tiêu (salt-and-pepper).

    Ý tưởng: chọn ngẫu nhiên ``amount`` × số điểm ảnh; mỗi điểm bị đổi thành
    255 (muối) hoặc 0 (tiêu) với xác suất 50/50 — nhiễu xung đột cực trị.

    Tham số: ``amount`` ∈ [0,1] tỉ lệ pixel bị thay; ``seed`` cho ``numpy.random``.

    Trả về: ảnh uint8 cùng kích thước.

    Liên quan C2: ví dụ nhiễu impulsive cần lọc trung vị / không gian.
    """
    source = require_uint8_image(img)
    prob = _check_amount(amount)
    rng = np.random.default_rng(_check_seed(seed))
    out = source.copy()

    def _one(channel: np.ndarray) -> np.ndarray:
        ch = channel.copy()
        flat = ch.reshape(-1)
        n_flip = round(prob * flat.size)
        if n_flip == 0:
            return ch
        # Chọn không lặp vị trí bị hỏng để đúng tỉ lệ amount
        idx = rng.choice(flat.size, size=n_flip, replace=False)
        salt = rng.random(n_flip) >= 0.5
        flat[idx[salt]] = 255
        flat[idx[~salt]] = 0
        return ch

    return apply_per_channel(out, _one) if source.ndim == 3 else _one(out)


def speckle(img: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Nhiễu speckle nhân: I' = clip(I · (1 + σ·N(0,1))).

    Cường độ nhân với hệ số ngẫu nhiên quanh 1; σ càng lớn dao động càng mạnh.
    Kết quả làm tròn và kẹp [0,255] về uint8.

    Liên quan C2: nhiễu đaplicative thường gặp trong radar/SAR; khử bằng lọc thích nghi.
    """
    source = require_uint8_image(img)
    s = _check_sigma(sigma)
    rng = np.random.default_rng(_check_seed(seed))

    def _one(channel: np.ndarray) -> np.ndarray:
        base = channel.astype(np.float64)
        noise = 1.0 + s * rng.standard_normal(base.shape)
        return np.clip(np.rint(base * noise), 0, 255).astype(np.uint8)

    return apply_per_channel(source, _one) if source.ndim == 3 else _one(source)


def poisson(img: np.ndarray, seed: int) -> np.ndarray:
    """Nhiễu shot (Poisson): mỗi pixel ~ Poisson(λ = cường độ pixel).

    Phản ánh đếm photon: phương sai tỉ lệ với λ; vùng sáng biến thiên mạnh hơn
    vùng tối. Đầu ra kẹp 255.

    Liên quan C2 / mô hình cảm biến: nhiễu phụ thuộc tín hiệu (khác Gaussian cộng).
    """
    source = require_uint8_image(img)
    rng = np.random.default_rng(_check_seed(seed))

    def _one(channel: np.ndarray) -> np.ndarray:
        lam = channel.astype(np.float64)
        noisy = rng.poisson(lam)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    return apply_per_channel(source, _one) if source.ndim == 3 else _one(source)
