"""Block copy-move course — translate vs rotated paste (seed 23, 160²).

Locked outcomes (default ``ThresholdSet``, see ``plans/reports/w-l4-261106.md``):
translate → ``SUSPICIOUS_INDICATION`` with mask IoU vs source∪destination > 0;
rotated paste → ``NO_INDICATION`` on this seed (not asserted suspicious).
"""

from __future__ import annotations

import cv2
import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.block.pipeline import BlockCopyMove

# --- geometry (brief) ---------------------------------------------------------
_IMG = 160
_RECT_W = _RECT_H = 48
_SRC_X, _SRC_Y = 40, 40
_DX, _DY = 32, 24


def _doc(bgr: np.ndarray) -> ImageDocument:
    h, w = bgr.shape[:2]
    return ImageDocument(
        array=bgr,
        width=w,
        height=h,
        channels=3,
        image_format="png",
        sha256="t" * 64,
        file_bytes=0,
        file_name="course.png",
    )


def _union_rect_mask(
    h: int, w: int, rects: list[tuple[int, int, int, int]]
) -> np.ndarray:
    u = np.zeros((h, w), dtype=np.uint8)
    for x, y, rw, rh in rects:
        u[y : y + rh, x : x + rw] = 1
    return u


def _mask_union_iou(mask: np.ndarray | None, h: int, w: int, rects: list[tuple[int, int, int, int]]) -> float:
    """IoU(detector mask > 0, binary union of axis-aligned rectangles)."""
    if mask is None or mask.size == 0:
        return 0.0
    m = (mask > 0).astype(np.uint8)
    u = _union_rect_mask(h, w, rects)
    inter = int((m & u).sum())
    uni = int((m | u).sum())
    return float(inter / uni) if uni else 0.0


def _base_textured() -> np.ndarray:
    rng = np.random.default_rng(23)
    return rng.integers(0, 255, (_IMG, _IMG, 3), dtype=np.uint8)


def _translate_copy_move(base: np.ndarray) -> np.ndarray:
    out = base.copy()
    patch = base[_SRC_Y : _SRC_Y + _RECT_H, _SRC_X : _SRC_X + _RECT_W].copy()
    out[
        _SRC_Y + _DY : _SRC_Y + _RECT_H + _DY,
        _SRC_X + _DX : _SRC_X + _RECT_W + _DX,
    ] = patch
    return out


def _rotated_paste_copy_move(base: np.ndarray) -> np.ndarray:
    out = base.copy()
    patch = base[_SRC_Y : _SRC_Y + _RECT_H, _SRC_X : _SRC_X + _RECT_W]
    m = cv2.getRotationMatrix2D((_RECT_W / 2.0, _RECT_H / 2.0), 15.0, 1.0)
    patch_rot = cv2.warpAffine(
        patch,
        m,
        (_RECT_W, _RECT_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    out[
        _SRC_Y + _DY : _SRC_Y + _RECT_H + _DY,
        _SRC_X + _DX : _SRC_X + _RECT_W + _DX,
    ] = patch_rot
    return out


def _ground_truth_rects() -> list[tuple[int, int, int, int]]:
    dst_x, dst_y = _SRC_X + _DX, _SRC_Y + _DY
    return [
        (_SRC_X, _SRC_Y, _RECT_W, _RECT_H),
        (dst_x, dst_y, _RECT_W, _RECT_H),
    ]


def test_block_translate_copy_move_locked_course(ts) -> None:
    """Pure translation copy-move: locked SUSPICIOUS + overlap vs source∪dest."""
    base = _base_textured()
    forged = _translate_copy_move(base)
    r = BlockCopyMove().analyze(_doc(forged), ts)

    assert r.verdict == Verdict.SUSPICIOUS_INDICATION
    assert r.mask is not None
    iou = _mask_union_iou(r.mask, _IMG, _IMG, _ground_truth_rects())
    assert iou > 0.0


def test_block_rotated_paste_locked_course(ts) -> None:
    """15° rotated patch: honest outcome on seed 23 — not forced suspicious."""
    base = _base_textured()
    forged = _rotated_paste_copy_move(base)
    r = BlockCopyMove().analyze(_doc(forged), ts)

    assert r.verdict == Verdict.NO_INDICATION
    assert r.mask is None or int(np.count_nonzero(r.mask)) == 0
