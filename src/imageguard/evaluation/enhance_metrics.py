"""Chỉ số chất lượng ảnh và căn chỉnh mặt nạ cho tab Enhance.

PSNR (đỉnh tín hiệu trên nhiễu): MSE trên mọi pixel/kênh ở float64, rồi
``10 * log10(255² / MSE)`` dB — chủ đề đánh giá sau lọc/khôi phục (C2/C3).

SSIM (Wang et al. 2004): cửa sổ Gaussian 11×11, σ=1.5; trung bình cục bộ μ,
phương sai σ² và hiệp σ_xy; hằng ổn định C1, C2; lấy trung bình bản đồ SSIM cục bộ
— gần với cảm nhận cấu trúc hơn MSE thuần.

``align_mask`` / ``iou`` hỗ trợ so khớp vùng (segmentation / đánh giá mặt nạ).
PSNR khi MSE=0 và IoU khi hợp rỗng trả về ``None``.
"""

from __future__ import annotations

import cv2
import numpy as np


def psnr(pred: np.ndarray, ref: np.ndarray) -> float | None:
    """PSNR (dB): so sánh ảnh khôi phục ``pred`` với tham chiếu ``ref``.

    Công thức: 10·log₁₀(255²/MSE).     Khác kích thước hoặc MSE=0 → ``None``.
    MSE tính trên mọi pixel/kênh — chỉ số toàn cục sau lọc/khôi phục (C2/C3).
    """
    prediction = np.asarray(pred)
    reference = np.asarray(ref)
    if prediction.shape != reference.shape:
        return None
    difference = prediction.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(difference * difference, dtype=np.float64))
    if mse == 0.0:
        return None
    return float(10.0 * np.log10((255.0**2) / mse))


def ssim(pred: np.ndarray, ref: np.ndarray) -> float | None:
    """SSIM trung bình (Wang 2004): μ, σ² cục bộ qua lọc Gaussian 11×11 (C2).

    Bản đồ SSIM = [(2μ_xμ_y+C1)(2σ_xy+C2)] / [(μ_x²+μ_y²+C1)(σ_x²+σ_y²+C2)].
    Ảnh nhỏ hơn 11×11 hoặc shape lệch → ``None``.
    """
    prediction = np.asarray(pred)
    reference = np.asarray(ref)
    if prediction.shape != reference.shape or prediction.ndim not in (2, 3):
        return None
    if prediction.shape[0] < 11 or prediction.shape[1] < 11:
        return None

    x = prediction.astype(np.float64)
    y = reference.astype(np.float64)
    mu_x = cv2.GaussianBlur(
        x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT
    )
    mu_y = cv2.GaussianBlur(
        y, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT
    )
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    # E[X²] − μ² ≈ phương sai cục bộ; E[XY] − μ_xμ_y ≈ hiệp cục bộ
    sigma_x_sq = cv2.GaussianBlur(
        x * x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT
    ) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(
        y * y, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT
    ) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(
        x * y, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT
    ) - mu_xy

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (
        sigma_x_sq + sigma_y_sq + c2
    )
    score_map = numerator / denominator
    return float(np.mean(score_map, dtype=np.float64))


def align_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Đổi kích thước mặt nạ về (cao, rộng) đích — INTER_NEAREST giữ nhãn rời rạc.

    Chỉ căn tọa độ với ảnh khác cỡ; không dùng cho IoU nếu caller resize sai luồng.
    """
    if len(shape) < 2:
        raise ValueError("shape must include height and width")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("target height and width must be positive")
    return cv2.resize(
        np.asarray(mask), (width, height), interpolation=cv2.INTER_NEAREST
    )


def iou(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """IoU nhị phân: |pred ∩ gt| / |pred ∪ gt| (segmentation / đánh giá vùng).

    Pixel > 0 coi là foreground. Hợp rỗng hoặc shape khác → ``None``.
    """
    prediction = np.asarray(pred)
    ground_truth = np.asarray(gt)
    if prediction.shape != ground_truth.shape:
        return None
    predicted_pixels = prediction > 0
    true_pixels = ground_truth > 0
    union = int(np.count_nonzero(predicted_pixels | true_pixels))
    if union == 0:
        return None
    intersection = int(np.count_nonzero(predicted_pixels & true_pixels))
    return float(intersection / union)
