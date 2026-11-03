"""Match + shift-vector voting + rasterize mask.

Lexicographic sort: block copy giống nhau → feature gần nhau → đứng cạnh
nhau trong thứ tự sort → chỉ cần so sánh k hàng xóm, không O(N²).

Shift voting = bộ lọc hình học thô: mọi cặp block trong MỘT vùng copy chia
sẻ CÙNG vector dịch chuyển (dx,dy); cặp nhiễu ngẫu nhiên có shift rải rác.
Giữ shift có count ≥ min_shift_votes → mask = union block của các cặp đó.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np


def match_pairs(
    feats: np.ndarray,
    pos: np.ndarray,
    sim_thresh: float,
    min_block_dist: float,
    n_neighbors: int = 10,
) -> np.ndarray:
    """Sort từ điển + so sánh k hàng xóm. Trả (M,2) index block pairs."""
    order = np.lexsort(feats.T[::-1])
    sfeats = feats[order].astype(np.float32)
    spos = pos[order]
    # sim_thresh định nghĩa trên khoảng cách chuẩn hoá theo chiều feature.
    dim = np.sqrt(sfeats.shape[1])
    n = len(sfeats)
    max_d = sim_thresh * dim
    found: list[np.ndarray] = []
    for k in range(1, n_neighbors + 1):
        ii = np.arange(n - k)
        jj = ii + k
        d = np.linalg.norm(sfeats[ii] - sfeats[jj], axis=1)
        near = d <= max_d
        if not near.any():
            continue  # hàng xóm sort vẫn có thể xa — bỏ qua, không break
        ii, jj = ii[near], jj[near]
        delta = spos[ii] - spos[jj]
        far = np.hypot(delta[:, 0], delta[:, 1]) >= min_block_dist
        if far.any():
            found.append(np.stack([ii[far], jj[far]], axis=1))
    if not found:
        return np.empty((0, 2), dtype=np.int32)
    ij = np.concatenate(found)
    ij = ij[np.lexsort((ij[:, 1], ij[:, 0]))]
    return order[ij].astype(np.int32)


def vote_mask(
    pairs: np.ndarray,
    pos: np.ndarray,
    B: int,
    shape: tuple[int, int],
    min_votes: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Trả (mask uint8, heat float32 [0,1], stats dict)."""
    stats = {"n_pairs": len(pairs), "n_shifts_kept": 0}
    H, W = shape
    count = np.zeros((H, W), np.float32)
    if len(pairs) == 0:
        return np.zeros((H, W), np.uint8), count, stats

    shifts = pos[pairs[:, 1]] - pos[pairs[:, 0]]  # (dy, dx) per pair
    votes = Counter(map(tuple, shifts.tolist()))
    kept = {s for s, c in votes.items() if c >= min_votes}
    stats["n_shifts_kept"] = len(kept)
    stats["top_shift_votes"] = max(votes.values())
    if not kept:
        return np.zeros((H, W), np.uint8), count, stats

    keep_pairs = pairs[[tuple(s) in kept for s in shifts]]
    for a, b in keep_pairs:
        for idx in (a, b):
            y, x = pos[idx]
            count[y : y + B, x : x + B] += 1.0
    mask = np.where(count > 0, 255, 0).astype(np.uint8)
    k = max(3, B // 2) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    heat = count / count.max() if count.max() > 0 else count
    return mask, heat, stats
