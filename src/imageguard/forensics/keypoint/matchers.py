"""Self-matching — bản chất copy-move: vùng bị copy có descriptor GIỐNG vùng
nguồn trong cùng ảnh, nên ta match ảnh với chính nó.

Pipeline filter (mỗi stage log số match còn lại vào `stats`):
  k=3 knnMatch → bỏ match tự nhiên (queryIdx==trainIdx, dist≈0)
  → Lowe ratio d1/d2 < ρ (ρ=0.8, Lowe 2004) — descriptor nhất quán thì
    nearest-neighbor phải gần hơn hẳn runner-up
  → spatial distance ≥ min_match_dist — cặp keypoint sát nhau thường là
    cùng một cấu trúc cục bộ, không phải copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet


@dataclass(slots=True)
class MatchPairs:
    """Cặp index keypoint (i, j) qua mọi filter — i: source, j: destination
    theo convention (không chứng minh được hướng thật, chỉ là ký hiệu)."""

    pairs: np.ndarray  # (N, 2) int — cột [i, j]
    stats: dict[str, int] = field(default_factory=dict)


def self_match(
    kps: list[cv2.KeyPoint],
    desc: np.ndarray,
    cfg: ThresholdSet,
    norm: int = cv2.NORM_L2,
) -> MatchPairs:
    """So khớp descriptor của ảnh với chính nó (copy-move: nguồn ≈ đích).

    BFMatcher (Brute-Force): với mỗi descriptor query, tìm k hàng xóm gần nhất
    trong cùng tập train=desc. Lowe ratio: chỉ giữ match nếu
    ``distance(nearest) < ρ × distance(second)`` — tránh match mơ hồ.

    Tham số: ``kps``, ``desc`` từ detector; ``norm`` khớp backend (L2/Hamming).
    Trả ``MatchPairs``: ``pairs`` (N,2) index; ``stats`` đếm qua từng bước lọc.
    Chủ đề môn: feature extraction + matching (không phải frequency/morphology).
    """
    rho = float(cfg.value("keypoint", "ratio_rho"))
    min_dist = float(cfg.value("keypoint", "min_match_dist"))
    stats: dict[str, int] = {}

    bf = cv2.BFMatcher(norm, crossCheck=False)
    # k=3: [self(dist≈0), nearest, runner-up] — ratio cần 2 neighbour thật.
    raw = bf.knnMatch(desc, desc, k=3)
    stats["knn_raw"] = len(raw)

    pairs: list[tuple[int, int]] = []
    pts = np.array([k.pt for k in kps], dtype=np.float32)
    for triplet in raw:
        # Bỏ self-match (dist≈0) rồi lấy 2 neighbour còn lại.
        real = [m for m in triplet if m.queryIdx != m.trainIdx]
        if len(real) < 2:
            continue
        m1, m2 = real[0], real[1]
        if m1.distance >= rho * m2.distance:
            continue  # ratio test fail — match không đủ "đặc trưng"
        # Lọc cặp sát nhau trong không gian ảnh.
        d = np.hypot(*(pts[m1.queryIdx] - pts[m1.trainIdx]))
        if d < min_dist:
            continue
        pairs.append((m1.queryIdx, m1.trainIdx))

    stats["after_ratio_spatial"] = len(pairs)
    # Dedupe (i,j)/(j,i): orient theo hướng hình học — displacement (dx,dy)
    # phải nằm trong canonical half-plane (dx>0, hoặc dx=0 & dy>0). Nếu sort
    # theo index thì một nửa cặp trỏ ngược → không affine nào fit hỗn hợp ±t.
    canon: set[tuple[int, int]] = set()
    for i, j in pairs:
        dx, dy = pts[j] - pts[i]
        canon.add((i, j) if (dx > 0 or (dx == 0 and dy > 0)) else (j, i))
    stats["unique_pairs"] = len(canon)
    return MatchPairs(pairs=np.array(sorted(canon), dtype=np.int32), stats=stats)
