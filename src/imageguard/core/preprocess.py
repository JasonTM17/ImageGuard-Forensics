"""Preprocessing per-algorithm. Quy tắc vàng: KHÔNG áp chung một pipeline —
mỗi method cần tín hiệu khác nhau.

- keypoint/block: grayscale + cap WORK_MAX_SIDE (tốc độ, SIFT scale-invariant
  nên downscale không mất cặp copy-move lớn).
- ela: giữ NGUYÊN file bytes — ELA phân tích lỗi re-encode, decode rồi chỉnh
  pixel là vô nghĩa.
- noise: giữ nhiễu tuyệt đối — denoise xoá luôn dấu vết forensic cần đo.
"""

from __future__ import annotations

import cv2
import numpy as np

from imageguard.core import limits
from imageguard.core.config import ThresholdSet, load_thresholds
from imageguard.core.models import ImageDocument


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def working_copy(
    doc: ImageDocument, cfg: ThresholdSet | None = None
) -> tuple[np.ndarray, float]:
    """Downscale về WORK_MAX_SIDE nếu cần. Trả (gray, scale_factor);
    scale_factor < 1 nghĩa là mask phải upscale lại khi map về ảnh gốc."""
    ts = cfg or load_thresholds()
    max_side = ts.limit("work_max_side", limits.WORK_MAX_SIDE)
    gray = to_gray(doc.array)
    h, w = gray.shape
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        doc.resolution_reduced = True
        doc.scale_factor = scale
    return gray, scale


def preprocess_keypoint(doc: ImageDocument, cfg: ThresholdSet | None = None) -> np.ndarray:
    gray, _ = working_copy(doc, cfg)
    # CLAHE nhẹ — tăng keypoint ở vùng tương phản thấp mà không đẩy nhiễu.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_block(doc: ImageDocument, cfg: ThresholdSet | None = None) -> np.ndarray:
    gray, _ = working_copy(doc, cfg)
    # Median 3×3: block-matching nhạy với nhiễu chấm; kernel nhỏ giữ cạnh.
    return cv2.medianBlur(gray, 3)


def preprocess_noise_residual(doc: ImageDocument, cfg: ThresholdSet | None = None) -> np.ndarray:
    """Residual = ảnh − median(ảnh): giữ high-frequency nhiễu để so noise σ
    giữa các vùng. KHÔNG denoise ảnh gốc trước đó."""
    ts = cfg or load_thresholds()
    k = int(ts.value("noise", "median_k"))
    gray = to_gray(doc.array).astype(np.float32)
    smooth = cv2.medianBlur(gray, k | 1)  # k phải lẻ
    return gray - smooth
