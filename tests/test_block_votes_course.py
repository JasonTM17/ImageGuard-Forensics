"""Block copy-move — ``top_shift_votes`` on an untouched image vs a translation.

Geometry matches ``tests/test_block_course.py``: seed 23, 160×160, a 48×48
rectangle at (40, 40) shifted by (32, 24). Default thresholds lock the
untouched count at 0 and the translated count at 705.
"""

from __future__ import annotations

import numpy as np

from imageguard.core.models import ImageDocument
from imageguard.forensics.block.pipeline import BlockCopyMove

_IMG = 160
_RECT_W = _RECT_H = 48
_SRC_X, _SRC_Y = 40, 40
_DX, _DY = 32, 24

# Last verified pair when strict ordering holds on default ThresholdSet (seed 23).
_LOCKED_TOP_SHIFT_VOTES_BASE = 0
_LOCKED_TOP_SHIFT_VOTES_TRANSLATED = 705


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
        file_name="votes_course.png",
    )


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


def test_top_shift_votes_rise_after_translation(ts) -> None:
    base = _base_textured()
    translated = _translate_copy_move(base)

    det = BlockCopyMove()
    v0 = int(det.analyze(_doc(base), ts).evidence["top_shift_votes"])
    v1 = int(det.analyze(_doc(translated), ts).evidence["top_shift_votes"])

    assert v0 == _LOCKED_TOP_SHIFT_VOTES_BASE
    assert v1 == _LOCKED_TOP_SHIFT_VOTES_TRANSLATED
    assert v1 > v0
