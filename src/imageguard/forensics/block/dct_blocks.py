"""Trích feature DCT của mọi block overlap B×B, stride 1.

DCT-II: F(u,v) = α(u)α(v) ΣΣ f(x,y)·cos[(2x+1)uπ/2B]·cos[(2y+1)vπ/2B]
Tách được thành F = D·block·Dᵀ → vector hoá theo hàng (RAM vừa phải).

Feature = `n_coeffs` hệ số zigzag đầu, LƯỢNG TỬ HOÁ (round(F/Q)) — lượng tử
hoá làm feature bất biến nhẹ với nhiễu/JPEG: hai block giống nhau sau nén
vẫn rơi cùng bin. Q tăng dần theo tần số (hệ số cao kém tin cậy hơn).
"""

from __future__ import annotations

import numpy as np


def dct_basis(B: int) -> np.ndarray:
    """Ma trận DCT-II trực giao B×B: F = D @ block @ D.T."""
    n = np.arange(B)
    u = n[:, None]
    alpha = np.where(u == 0, np.sqrt(1.0 / B), np.sqrt(2.0 / B))
    return alpha * np.cos((2 * n[None, :] + 1) * u * np.pi / (2 * B))


def zigzag_indices(B: int, k: int) -> np.ndarray:
    """k chỉ số (u,v) đầu tiên theo thứ tự zigzag — giữ hệ số tần số thấp."""
    order = sorted(
        ((u, v) for u in range(B) for v in range(B)),
        key=lambda p: (p[0] + p[1], -p[0] if (p[0] + p[1]) % 2 else p[0]),
    )
    return np.array([u * B + v for u, v in order[:k]])


def block_std_map(gray: np.ndarray, B: int) -> np.ndarray:
    """Std mỗi block qua box filter: var = E[x²] − E[x]². Blocks phẳng
    (std≈0) giống hệt nhau về mặt feature nhưng VÔ NGHĨA cho copy-move —
    chúng là nguồn false-match lớn nhất của trường phái này."""
    import cv2

    g = gray.astype(np.float32)
    H, W = g.shape
    # anchor=(0,0): giá trị tại (r,c) = thống kê block top-left (r,c) —
    # align đúng vị trí sliding_window_view.
    mean = cv2.boxFilter(g, -1, (B, B), anchor=(0, 0), normalize=True)
    mean2 = cv2.boxFilter(g * g, -1, (B, B), anchor=(0, 0), normalize=True)
    var = np.clip(mean2 - mean * mean, 0, None)
    return np.sqrt(var)[: H - B + 1, : W - B + 1]


def extract_features(
    gray: np.ndarray, B: int, n_coeffs: int = 24, flat_std_eps: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Trả (features int32 (N,K), positions int32 (N,2) — top-left mỗi block).
    Blocks có std < flat_std_eps bị loại (không texture nhận diện).

    int32: F(0,0) có thể tới B·255 — vượt int16 khi block_size ≥ 128.

    Chạy theo hàng để tránh materialize (H-B+1)(W-B+1)×B×B float trong RAM.
    """
    D = dct_basis(B).astype(np.float32)
    zz = zigzag_indices(B, n_coeffs)
    # Bảng lượng tử đơn giản: Q tăng tuyến tính theo u+v (tần số cao → bước lớn).
    uu, vv = np.divmod(zz, B)
    q = (1.0 + (uu + vv) / 4.0).astype(np.float32)

    win = np.lib.stride_tricks.sliding_window_view(gray.astype(np.float32), (B, B))
    n_rows, n_cols = win.shape[0], win.shape[1]
    stds = block_std_map(gray, B)  # (H-B+1, W-B+1) align với top-left
    feats = np.empty((n_rows * n_cols, n_coeffs), np.int32)
    for r in range(n_rows):
        blocks = win[r]  # (n_cols, B, B) view — không copy
        F = np.einsum("ux,cxy,vy->cuv", D, blocks, D, optimize=True)
        feats[r * n_cols : (r + 1) * n_cols] = np.rint(
            F.reshape(n_cols, B * B)[:, zz] / q
        ).astype(np.int32)
    pos = np.array(
        [(r, c) for r in range(n_rows) for c in range(n_cols)], dtype=np.int32
    )
    keep = stds.ravel() >= flat_std_eps
    return feats[keep], pos[keep]
