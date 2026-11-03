"""Mask-level fusion giữa các detector. Mode:
  "any" = union — vùng nghi bởi ÍT NHẤT một method (recall cao, nhiễu cộng dồn)
  "all" = intersection — chỉ vùng MỌI method đồng ý (precision cao, có thể rỗng)
Verdict-level fusion (ưu tiên method, INCONCLUSIVE policy) ở Phase 4.
"""

from __future__ import annotations

import cv2
import numpy as np


def fuse_masks(
    masks: list[np.ndarray | None],
    mode: str = "any",
    close_k: int = 5,
) -> np.ndarray | None:
    valid = [m for m in masks if m is not None]
    if not valid:
        return None
    shape = valid[0].shape
    valid = [
        cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        if m.shape != shape else m
        for m in valid
    ]
    if mode == "all":
        out = np.full(shape, 255, np.uint8)
        for m in valid:
            out = cv2.bitwise_and(out, m)
    else:
        out = np.zeros(shape, np.uint8)
        for m in valid:
            out = cv2.bitwise_or(out, m)
    if close_k > 1:
        k = close_k | 1
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return out
