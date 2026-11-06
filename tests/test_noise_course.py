"""L3 — noise residual order course scenario (deterministic seed 20261106).

Một ảnh 128×128: nửa trái là nhiễu Gaussian mạnh, nửa phải là cùng nhiễu
sau khi Gaussian blur mạnh. NoiseMap không bao giờ trả SUSPICIOUS_INDICATION
(contract aux) và residual std trên nửa mịn phải thấp hơn nửa nhiễu —
ánh xạ trực tiếp vào `residual = I − medianBlur(I, k)` mà noise_map.py
tài liệu hóa. Analyzer chỉ lộ σ median/MAD/outlier_frac nên thứ tự residual
được đo trong test, không qua helper private.

Không trùng `tests/test_auxiliary_l2.py`: đó là blur patch trên ảnh thật;
đây là hai nửa của cùng một noise field, đo trực tiếp thứ tự residual.
"""

from __future__ import annotations

import cv2
import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.noise_map import NoiseMap


def _doc_from(bgr: np.ndarray) -> ImageDocument:
    h, w = bgr.shape[:2]
    return ImageDocument(
        array=bgr, width=w, height=h, channels=3, image_format="png",
        sha256="n" * 64, file_bytes=0, file_name="noise_halves.png",
    )


def test_noise_half_blur_residual_order(ts) -> None:
    rng = np.random.default_rng(20261106)
    h = w = 128
    half = w // 2
    noise = rng.normal(0.0, 40.0, (h, half)).astype(np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[:, :half] = noise[:, :, None]
    img[:, half:] = cv2.GaussianBlur(noise, (0, 0), 4.0)[:, :, None]
    img = np.clip(128.0 + img, 0, 255).astype(np.uint8)

    result = NoiseMap().analyze(_doc_from(img), ts)
    assert result.verdict != Verdict.SUSPICIOUS_INDICATION

    # Cùng công thức residual mà noise_map.py tài liệu hóa: I − medianBlur(I, k).
    k = int(ts.value("noise", "median_k"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = gray - cv2.medianBlur(gray, k).astype(np.float32)

    # Lùi một khoảng lề khỏi đường chia để medianBlur không trộn hai nửa.
    margin = 2 * k
    noisy_std = float(residual[:, margin:half - margin].std())
    blurred_std = float(residual[:, half + margin:w - margin].std())
    assert blurred_std < noisy_std
