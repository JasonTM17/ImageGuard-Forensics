"""Keypoint → mask/heatmap.

Keypoint là ĐIỂM, không phải vùng — phải nở thành vùng:
  mỗi verified keypoint (cả hai phía) đóng một Gaussian blob σ ∝ k.size
  (scale của feature — vùng lớn sinh keypoint lớn) → density map
  → normalize [0,1] = heatmap → threshold → morphology close = mask.

Convention: mask = union source-side + dest-side (không phân biệt được
hướng copy chỉ từ keypoint — ghi rõ trong limitations của result).
"""

from __future__ import annotations

import cv2
import numpy as np


def build_mask(
    kps: list,
    pairs: np.ndarray,
    inlier_idx: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Trả (mask uint8 0/255, heatmap float32 [0,1]) kích thước `shape` (h,w)."""
    h, w = shape
    heat = np.zeros((h, w), np.float32)
    if len(inlier_idx):
        ver = pairs[inlier_idx]
        for side in (0, 1):
            for kp_idx in ver[:, side]:
                kp = kps[int(kp_idx)]
                x, y = round(kp.pt[0]), round(kp.pt[1])
                sigma = max(float(kp.size) / 2.0, 8.0)
                r = int(sigma * 2)
                x0, x1 = max(0, x - r), min(w, x + r + 1)
                y0, y1 = max(0, y - r), min(h, y + r + 1)
                if x1 <= x0 or y1 <= y0:
                    continue
                yy, xx = np.mgrid[y0:y1, x0:x1]
                blob = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma)))
                heat[y0:y1, x0:x1] += blob.astype(np.float32)
    if heat.max() > 0:
        heat /= heat.max()
    mask = np.zeros((h, w), np.uint8)
    mask[heat > 0.25] = 255
    # Close nối các blob rời thành vùng liền (keypoint thưa → mask đứt gãy).
    k = max(5, min(h, w) // 40) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return mask, heat


def upscale_mask(mask: np.ndarray, scale: float, shape: tuple[int, int]) -> np.ndarray:
    """Map mask từ working copy về kích thước ảnh gốc khi scale<1."""
    if scale >= 1.0:
        return mask
    out = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(out > 127, 255, 0).astype(np.uint8)
