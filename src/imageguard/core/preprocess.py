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


def scaled_kernel(k: int, scale: float) -> int:
    """Kernel (px, số lẻ) trên ảnh đã downscale — giữ cùng kích thước vật lý."""
    return max(3, round(k * scale) | 1)


def scaled_block(bs: int, scale: float) -> int:
    """Cạnh block (px) trên ảnh đã downscale — giữ cùng kích thước vật lý."""
    return max(4, round(bs * scale))


def aux_working_copy(
    doc: ImageDocument, cfg: ThresholdSet | None = None
) -> tuple[ImageDocument, float]:
    """Bản làm việc cho aux methods (noise/freq): downscale về aux_max_side.

    Trả (work_doc, scale). Khi scale < 1, caller PHẢI upscale mask/heatmap về
    (doc.height, doc.width) bằng upscale_to_doc và ghi resolution_reduced để
    report minh bạch — y hệt pattern WORK_MAX_SIDE của detector.
    """
    ts = cfg or load_thresholds()
    max_side = ts.limit("aux_max_side", limits.AUX_MAX_SIDE)
    h, w = doc.array.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    if scale >= 1.0:
        return doc, 1.0
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(doc.array, (nw, nh), interpolation=cv2.INTER_AREA)
    work = ImageDocument(
        array=small, width=nw, height=nh, channels=doc.channels,
        image_format=doc.image_format, sha256=doc.sha256,
        file_bytes=doc.file_bytes, file_name=doc.file_name, exif=doc.exif,
        resolution_reduced=True, scale_factor=scale)
    return work, scale


def upscale_to_doc(arr: np.ndarray, doc: ImageDocument,
                   interp: int = cv2.INTER_NEAREST) -> np.ndarray:
    """Upscale mask/heatmap về đúng kích thước ảnh gốc (no-op nếu đã khớp)."""
    if arr.shape[:2] == (doc.height, doc.width):
        return arr
    return cv2.resize(arr, (doc.width, doc.height), interpolation=interp)


def aux_limitation(scale: float) -> str | None:
    if scale >= 1.0:
        return None
    return (f"ảnh lớn — aux phân tích trên bản downscale ×{scale:.2f} "
            "(kernel đổi tỉ lệ thuận, mask/heatmap upscale về kích thước gốc)")
