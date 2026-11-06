"""Course lock: live dual-path disagreement still fuses to block-only suspicious."""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.block.pipeline import BlockCopyMove
from imageguard.forensics.fusion import combine
from imageguard.forensics.keypoint.pipeline import KeypointCopyMove

_IMG = 160
_RW = _RH = 48
_SX, _SY = 40, 40
_DX, _DY = 32, 24


def _doc(bgr: np.ndarray) -> ImageDocument:
    h, w = bgr.shape[:2]
    return ImageDocument(
        array=bgr,
        width=w,
        height=h,
        channels=3,
        image_format="png",
        sha256="g" * 64,
        file_bytes=0,
        file_name="fusion_disagree.png",
    )


def test_live_disagree_fusion_lists_only_block(ts) -> None:
    rng = np.random.default_rng(23)
    base = rng.integers(0, 255, (_IMG, _IMG, 3), dtype=np.uint8)
    forged = base.copy()
    forged[
        _SY + _DY : _SY + _DY + _RH,
        _SX + _DX : _SX + _DX + _RW,
    ] = base[_SY : _SY + _RH, _SX : _SX + _RW]
    doc = _doc(forged)

    block = BlockCopyMove().analyze(doc, ts)
    keypoint = KeypointCopyMove().analyze(doc, ts)
    assert block.verdict == Verdict.SUSPICIOUS_INDICATION
    assert keypoint.verdict == Verdict.INCONCLUSIVE

    overall, signals, warnings = combine([block, keypoint])
    assert overall == Verdict.SUSPICIOUS_INDICATION
    assert warnings == []
    assert signals == [MethodName.COPY_MOVE_BLOCK.value]
    assert MethodName.COPY_MOVE_KEYPOINT.value not in signals
