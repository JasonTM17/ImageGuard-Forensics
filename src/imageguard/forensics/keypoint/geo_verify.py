"""Geometric verification — match descriptors còn sót nhiễu; cặp copy-move THẬT
phải tuân cùng một phép biến đổi hình học.

Chiến lược 2 pass (đánh đổi đã ghi trong docs/algorithms/02-keypoint.md):
  1. Global affine RANSAC trên toàn bộ cặp — xử lý được rotate/scale vì
     toàn vùng copy chia sẻ MỘT transform affine, kể cả khi displacement
     vector biến thiên theo vị trí.
  2. Displacement clustering trên outliers — xử lý NHIỀU vùng copy độc lập
     kiểu translate (mỗi vùng một (dx,dy) riêng): greedy theo khoảng cách
     displacement, mỗi cluster ≥ min_cluster → RANSAC affine riêng.

Giới hạn thành thật: nhiều vùng copy với transform affine KHÁC nhau triệt để
có thể chỉ bắt được một phần — ghi vào limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet


@dataclass(slots=True)
class VerifyResult:
    """Kết quả xác minh hình học sau self-match.

    ``inlier_pairs``: chỉ số hàng trong ``pairs`` được RANSAC chấp nhận.
    ``n_clusters``: số cụm transform (global pass + cụm displacement).
    ``models``: ma trận affine 2×3 ước lượng cho từng cụm.
    """

    inlier_pairs: np.ndarray  # (M, 2) index vào mảng pairs đầu vào
    n_clusters: int
    models: list[np.ndarray] = field(default_factory=list)  # affine 2×3 per cluster


def _ransac_affine(
    src: np.ndarray, dst: np.ndarray, eps: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fit affine 2×3 (6 dof, cần ≥3 điểm không thẳng hàng) bằng RANSAC.
    Trả (model, inlier_mask) hoặc (None, None) nếu không đủ điểm.

    OpenCV: ``estimateAffine2D(dst→src)`` — copy-move nghĩa là điểm đích
    map về điểm nguồn bằng affine 2D (xoay, scale, dịch chuyển).
    RANSAC loại outlier không cùng một transform.
    """
    if len(src) < 3:
        return None, None
    model, inliers = cv2.estimateAffine2D(
        dst.reshape(-1, 1, 2),
        src.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=eps,
    )
    if model is None or inliers is None:
        return None, None
    return model, inliers.ravel().astype(bool)


def geometric_verify(
    pairs: np.ndarray,
    kps: list,
    cfg: ThresholdSet,
) -> VerifyResult:
    """Lọc cặp match bằng nhất quán hình học (spatial / geometric reasoning).

    Pass 1: một affine global cho toàn ảnh (một vùng copy xoay/scale).
    Pass 2: gom cụm theo vector dịch chuyển (dx,dy) gần nhau, RANSAC riêng
    — nhiều vùng copy-translate độc lập.

    Ngưỡng ``ransac_residual``, ``min_cluster`` từ config. Chủ đề: feature
    extraction + không gian (không morphology trên ảnh ở bước này).
    """
    if len(pairs) == 0:
        return VerifyResult(inlier_pairs=np.empty(0, np.int32), n_clusters=0)

    eps = float(cfg.value("keypoint", "ransac_residual"))
    min_cluster = int(cfg.value("keypoint", "min_cluster"))
    pts = np.array([k.pt for k in kps], dtype=np.float32)
    src_all = pts[pairs[:, 0]]
    dst_all = pts[pairs[:, 1]]

    inlier_idx: list[int] = []
    models: list[np.ndarray] = []
    remaining = np.ones(len(pairs), dtype=bool)

    # Pass 1: global affine — một vùng copy duy nhất (translate/rotate/scale).
    model, inl = _ransac_affine(src_all, dst_all, eps)
    if inl is not None and inl.sum() >= min_cluster:
        inlier_idx.extend(np.flatnonzero(inl).tolist())
        models.append(model)
        remaining[inl] = False

    # Pass 2: displacement clustering trên phần còn lại — nhiều vùng translate.
    if remaining.any():
        disp = dst_all - src_all  # (N,2)
        # Bán kính greedy: cặp cùng cụm nếu displacement gần trung bình cụm.
        tol = max(eps * 3.0, 8.0)
        clusters: list[list[int]] = []
        for i in np.flatnonzero(remaining):
            placed = False
            for c in clusters:
                cx = np.mean(disp[c], axis=0)
                if np.hypot(*(disp[i] - cx)) <= tol:
                    c.append(int(i))
                    placed = True
                    break
            if not placed:
                clusters.append([int(i)])
        n_pass2 = 0
        for c in clusters:
            if len(c) < min_cluster:
                continue
            idx = np.array(c)
            m, inl = _ransac_affine(src_all[idx], dst_all[idx], eps)
            if inl is not None and inl.sum() >= min_cluster:
                inlier_idx.extend(idx[inl].tolist())
                models.append(m)
                n_pass2 += 1
        n_clusters = (1 if models and not remaining.all() else 0) + n_pass2
    else:
        n_clusters = len(models)

    return VerifyResult(
        inlier_pairs=np.array(sorted(set(inlier_idx)), dtype=np.int32),
        n_clusters=n_clusters,
        models=models,
    )
