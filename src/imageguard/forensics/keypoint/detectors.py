"""Detector backends — trừu hoá SIFT/ORB sau cùng interface detect() → (kps, desc).

SIFT (Scale-Invariant Feature Transform): khớp copy-move tốt ngay cả khi vùng
copy bị rotate/scale nhẹ — chính vì vậy nó là mainline. ORB nhanh hơn ~10×
nhưng descriptor nhị phân kém phân biệt hơn → fallback cho máy yếu.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet


class Backend(Protocol):
    """Contract detector: ``detect(gray)`` và ``norm`` cho BFMatcher."""

    norm: int  # cv2.NORM_L2 (float desc) hoặc cv2.NORM_HAMMING (binary desc)

    def detect(self, gray: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """Ảnh xám uint8 → keypoints OpenCV + ma trận descriptor (hoặc None)."""
        ...


class SiftBackend:
    """SIFT qua OpenCV — trích xuất đặc trưng (feature extraction).

    Keypoint: tọa độ (x,y), scale, hướng gradient tại điểm “ổn định” theo
    thang (DoG extrema). Descriptor: vector float 128 chiều mô tả histogram
    gradient lân cận — bất biến nhẹ với scale/xoay nên phù hợp copy-move.

    ``norm`` = ``NORM_L2`` vì so khớp descriptor float bằng khoảng cách Euclid.
    """

    norm = cv2.NORM_L2

    def __init__(self, nfeatures: int = 0) -> None:
        self._det = cv2.SIFT_create(nfeatures=nfeatures)

    def detect(self, gray: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """``detectAndCompute``: ảnh xám → (danh sách KeyPoint, ma trận desc N×128)."""
        return self._det.detectAndCompute(gray, None)


class OrbBackend:
    """ORB (Oriented FAST and Rotated BRIEF) — FAST góc + descriptor nhị phân.

    Descriptor BRIEF (chuỗi bit) → ``norm`` = ``NORM_HAMMING`` khi match.
    Nhanh hơn SIFT nhưng mô tả kém tinh hơn; dùng khi cần tốc độ hoặc fallback.
    """

    norm = cv2.NORM_HAMMING

    def __init__(self, nfeatures: int = 5000) -> None:
        # ORB mặc định 500 kp quá ít cho forensic — tăng lên.
        self._det = cv2.ORB_create(nfeatures=max(nfeatures, 2000))

    def detect(self, gray: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """Cùng contract với SIFT: (keypoints, ma trận descriptor nhị phân)."""
        return self._det.detectAndCompute(gray, None)


_REGISTRY = {"sift": SiftBackend, "orb": OrbBackend}


def create_backend(cfg: ThresholdSet) -> Backend:
    """Đọc ``keypoint.backend`` và ``keypoint.nfeatures`` từ config → instance backend."""
    name = str(cfg.value("keypoint", "backend")).lower()
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"keypoint.backend='{name}' không hỗ trợ (sift|orb)")
    nfeat = int(cfg.value("keypoint", "nfeatures"))
    # nfeatures > 0 áp dụng cho cả sift lẫn orb; 0 = 0 → để default của backend
    # (SIFT: 0 = không giới hạn; OrbBackend: floor 2000 cho forensic).
    if nfeat > 0:
        return cls(nfeatures=nfeat)
    return cls()
