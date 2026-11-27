"""YuNet face detection — ROI locator cho GenConViT.

YuNet (``cv2.FaceDetectorYN``) định vị vùng mặt để GenConViT chấm điểm,
độc lập với Haar cascade dùng trong các đặc trưng DIP hiện có. OpenCV của
dự án vẫn là 4.x và không yêu cầu OpenCV 5. Model ONNX ~232 KB từ
opencv_zoo, pin SHA-256, CPU-only, cho cả bbox lẫn landmarks. File model là
tùy chọn và KHÔNG commit vào repo (``models/`` đã gitignore); thiếu file thì
method trả NOT_APPLICABLE thay vì crash.

Detector cache theo (width, height) vì ``setInputSize`` phải khớp cỡ ảnh —
tạo mới mỗi lần tốn ~2 s trong khi detect chỉ ~vài chục ms.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from imageguard.core.paths import model_root

_LOGGER = logging.getLogger(__name__)

YUNET_MODEL_PATH = model_root() / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_BYTES = 232_589
YUNET_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

# Ảnh lớn hơn cạnh này được downscale riêng cho bước detection (dnn chậm theo
# diện tích); bbox sau đó scale ngược về toạ độ ảnh gốc để crop lấy đúng
# pixel nguyên bản — crop KHÔNG lấy từ bản downscale.
DETECT_MAX_SIDE = 1280

_MIN_FACE_SIDE = 32  # px — dưới này crop quá nhỏ cho classifier 224×224


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def yunet_integrity_error() -> str | None:
    """None nếu model YuNet hợp lệ; chuỗi mô tả ngắn nếu thiếu/sai integrity."""
    path = Path(YUNET_MODEL_PATH)
    if not path.is_file():
        return "missing"
    if path.stat().st_size != YUNET_MODEL_BYTES:
        return "size-mismatch"
    if sha256_file(path) != YUNET_MODEL_SHA256:
        return "sha256-mismatch"
    return None


def _quiet_cv2_logging() -> None:
    """OpenCV dnn có thể cảnh báo 'Targets are not supported' mỗi lần tạo Net —
    noise thuần, giảm xuống ERROR nếu API tồn tại (không bắt buộc)."""
    set_level = getattr(cv2, "setLogLevel", None)
    level = getattr(cv2, "LOG_LEVEL_ERROR", None)
    if level is None:
        utils_logging = getattr(getattr(cv2, "utils", None), "logging", None)
        level = getattr(utils_logging, "LOG_LEVEL_ERROR", None)
    if set_level is not None and level is not None:
        try:
            set_level(level)
        except Exception:  # logging noise không đáng làm hỏng detection
            _LOGGER.debug("cv2.setLogLevel unavailable", exc_info=True)


@lru_cache(maxsize=4)
def _detector(width: int, height: int) -> Any:
    _quiet_cv2_logging()
    return cv2.FaceDetectorYN.create(
        str(YUNET_MODEL_PATH), "", (width, height), 0.6, 0.3, 5000
    )


def detect_primary_face(
    bgr: np.ndarray, detect_max_side: int = DETECT_MAX_SIDE
) -> tuple[tuple[int, int, int, int], float] | None:
    """Trả ((x, y, w, h) toạ độ ảnh gốc, confidence) của mặt lớn nhất, hoặc None.

    Detection chạy trên bản downscale ≤ ``detect_max_side`` khi ảnh lớn; bbox
    scale ngược về toạ độ gốc và clip trong khung. Detection chỉ nhận diện —
    không phải định danh danh tính.
    """
    h, w = bgr.shape[:2]
    scale = min(1.0, detect_max_side / max(w, h))
    if scale < 1.0:
        dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
        work = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
    else:
        dw, dh, work = w, h, bgr

    _, faces = _detector(dw, dh).detect(work)
    if faces is None or len(faces) == 0:
        return None

    best = max(faces, key=lambda row: row[2] * row[3])
    inv = 1.0 / scale
    x, y = round(float(best[0]) * inv), round(float(best[1]) * inv)
    fw, fh = round(float(best[2]) * inv), round(float(best[3]) * inv)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + fw), min(h, y + fh)
    if x1 - x0 < _MIN_FACE_SIDE or y1 - y0 < _MIN_FACE_SIDE:
        return None
    return (x0, y0, x1 - x0, y1 - y0), float(best[14])


def crop_face(bgr: np.ndarray, rect: tuple[int, int, int, int], size: int) -> np.ndarray:
    """Crop tight face rect rồi resize (size×size) INTER_AREA — đúng pipeline
    train của GenConViT (erprogs/GenConViT ``face_rec``)."""
    x, y, w, h = rect
    crop = bgr[y : y + h, x : x + w]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
