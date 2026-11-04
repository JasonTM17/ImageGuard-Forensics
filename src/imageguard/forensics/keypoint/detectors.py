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
    """Bất kỳ detector nào có detectAndCompute đều dùng được."""

    norm: int  # cv2.NORM_L2 (float desc) hoặc cv2.NORM_HAMMING (binary desc)

    def detect(self, gray: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        ...


class SiftBackend:
    norm = cv2.NORM_L2

    def __init__(self, nfeatures: int = 0) -> None:
        self._det = cv2.SIFT_create(nfeatures=nfeatures)

    def detect(self, gray: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        return self._det.detectAndCompute(gray, None)


class OrbBackend:
    norm = cv2.NORM_HAMMING

    def __init__(self, nfeatures: int = 5000) -> None:
        # ORB mặc định 500 kp quá ít cho forensic — tăng lên.
        self._det = cv2.ORB_create(nfeatures=max(nfeatures, 2000))

    def detect(self, gray: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        return self._det.detectAndCompute(gray, None)


_REGISTRY = {"sift": SiftBackend, "orb": OrbBackend}


def create_backend(cfg: ThresholdSet) -> Backend:
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
