"""Keypoint copy-move on a synthetic translate forge (numpy Generator seed 23).

Một ảnh 384×384 có texture (gradient + shapes + noise) được dựng bằng
`np.random.default_rng(23)`. Copy một rectangle 140×140 bằng phép tịnh tiến
thuần (dx, dy) = (48, 36) — không xoay, không scale. KeypointCopyMove với
ThresholdSet mặc định phải ra SUSPICIOUS_INDICATION và mask phải phủ cả vùng
nguồn lẫn vùng đích (IoU > 0.2). Control cùng seed, không copy, không được
SUSPICIOUS.

Không thêm test trùng `test_easy_tier_detects_majority` hay control dùng
fixture `valid_png` đã có trong tests/test_keypoint.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.keypoint.pipeline import KeypointCopyMove


def _doc_from(bgr: np.ndarray) -> ImageDocument:
    h, w = bgr.shape[:2]
    return ImageDocument(
        array=bgr, width=w, height=h, channels=3, image_format="png",
        sha256="k" * 64, file_bytes=0, file_name="synthetic.png",
    )


def _iou(mask: np.ndarray, gt: np.ndarray) -> float:
    a, b = mask > 0, gt > 0
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 0.0


def _textured_base(seed: int = 23, size: int = 384) -> np.ndarray:
    """Nền có texture: gradient + shapes + AWGN + speckle + mid-freq noise.

    Deterministic theo `seed` — không cần file ngoài, không phụ thuộc helper
    private của synthforge.
    """
    rng = np.random.default_rng(seed)
    w = h = size
    x = np.linspace(0, 255, w, dtype=np.float32)
    y = np.linspace(0, 255, h, dtype=np.float32)
    img = np.tile(x, (h, 1)) * 0.5 + np.tile(y, (w, 1)).T * 0.5
    img = cv2.merge([img, np.roll(img, 32, axis=1), np.roll(img, 16, axis=0)])
    for _ in range(rng.integers(25, 45)):
        c = tuple(int(v) for v in rng.integers(30, 225, 3))
        if rng.random() < 0.5:
            x0, y0 = int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))
            x1 = min(w, x0 + int(rng.integers(8, 80)))
            y1 = min(h, y0 + int(rng.integers(8, 80)))
            cv2.rectangle(img, (x0, y0), (x1, y1), c, -1)
        else:
            cx = int(rng.integers(20, w - 20))
            cy = int(rng.integers(20, h - 20))
            cv2.circle(img, (cx, cy), int(rng.integers(5, 50)), c, -1)
    img += rng.normal(0, 6, img.shape).astype(np.float32)
    for _ in range(int(w * h * 0.006)):
        cx = int(rng.integers(2, w - 2))
        cy = int(rng.integers(2, h - 2))
        r = int(rng.integers(1, 3))
        cv2.circle(img, (cx, cy), r, tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    noise = rng.normal(0, 18, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (5, 5), 0)
    img += noise[:, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def test_translation_copy_move_course(ts) -> None:
    base = _textured_base()
    method = KeypointCopyMove()

    # Control: cùng generator seed, không copy — không được SUSPICIOUS.
    control = method.analyze(_doc_from(base), ts)
    assert control.verdict != Verdict.SUSPICIOUS_INDICATION

    # Copy 140×140 từ (80,80) tịnh tiến (48,36) → (128,116).
    dx, dy = 48, 36
    sx, sy, rw, rh = 80, 80, 140, 140
    forged = base.copy()
    forged[sy + dy:sy + dy + rh, sx + dx:sx + dx + rw] = base[sy:sy + rh, sx:sx + rw]

    # GT = union(vùng nguồn, vùng đích).
    gt = np.zeros((base.shape[0], base.shape[1]), np.uint8)
    gt[sy:sy + rh, sx:sx + rw] = 255
    gt[sy + dy:sy + dy + rh, sx + dx:sx + dx + rw] = 255

    result = method.analyze(_doc_from(forged), ts)
    assert result.verdict == Verdict.SUSPICIOUS_INDICATION
    assert result.mask is not None
    assert _iou(result.mask, gt) > 0.2
    # Mask phải chạm cả hai vùng — không chỉ một vùng.
    assert np.any(result.mask[sy:sy + rh, sx:sx + rw] > 0)
    assert np.any(result.mask[sy + dy:sy + dy + rh, sx + dx:sx + dx + rw] > 0)
