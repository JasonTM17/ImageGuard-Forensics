"""Enhancement metrics and mask alignment.

PSNR is ``10 * log10(255**2 / MSE)``, with MSE computed over every pixel and
channel in float64. SSIM follows Wang et al. (2004): an 11x11 Gaussian window
with sigma 1.5, ``C1 = (0.01*255)**2``, and ``C2 = (0.03*255)**2``; the result
is the mean of the local SSIM map across pixels and channels. Undefined PSNR
for identical inputs and IoU for two empty masks are represented by ``None``.
"""

from __future__ import annotations

import cv2
import numpy as np


def psnr(pred: np.ndarray, ref: np.ndarray) -> float | None:
    """Compute peak signal-to-noise ratio in dB, or None when undefined."""
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
    """Compute mean Wang 2004 local SSIM, or None for incompatible images."""
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
    """Resize a mask to ``(height, width)`` for coordinate alignment only."""
    if len(shape) < 2:
        raise ValueError("shape must include height and width")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("target height and width must be positive")
    return cv2.resize(
        np.asarray(mask), (width, height), interpolation=cv2.INTER_NEAREST
    )


def iou(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """Compute binary intersection over union without resizing or cropping."""
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
