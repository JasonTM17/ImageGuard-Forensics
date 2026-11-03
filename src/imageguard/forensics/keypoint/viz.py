"""Visualization artifacts — ảnh nối các cặp keypoint verified (debug/report)."""

from __future__ import annotations

import cv2
import numpy as np


def draw_pairs(
    bgr: np.ndarray,
    kps: list,
    pairs: np.ndarray,
    inlier_idx: np.ndarray,
) -> np.ndarray:
    out = bgr.copy()
    for k in range(len(inlier_idx)):
        i, j = pairs[inlier_idx[k]]
        p1 = tuple(int(v) for v in kps[int(i)].pt)
        p2 = tuple(int(v) for v in kps[int(j)].pt)
        color = (0, int(40 + 200 * (k / max(1, len(inlier_idx)))), 255)
        cv2.line(out, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(out, p1, 3, (255, 160, 0), -1, cv2.LINE_AA)
        cv2.circle(out, p2, 3, (0, 255, 0), -1, cv2.LINE_AA)
    return out


def overlay_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    out[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(bgr, 0.6, out, 0.4, 0)
