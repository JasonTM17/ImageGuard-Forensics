"""Course lock: on a small random translate forge, detectors may disagree.

Seed-23 uniform noise 160×160 with a 48×48 translation paste: BlockCopyMove
returns SUSPICIOUS_INDICATION while KeypointCopyMove stays INCONCLUSIVE with
exactly four verified pairs. This is an honest dual-path mismatch — do not
force keypoint to convict.
"""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.block.pipeline import BlockCopyMove
from imageguard.forensics.keypoint.pipeline import KeypointCopyMove

_IMG = 160
_RW = _RH = 48
_SX, _SY = 40, 40
_DX, _DY = 32, 24
_LOCKED_VERIFIED = 4


def _doc(bgr: np.ndarray) -> ImageDocument:
    h, w = bgr.shape[:2]
    return ImageDocument(
        array=bgr,
        width=w,
        height=h,
        channels=3,
        image_format="png",
        sha256="x" * 64,
        file_bytes=0,
        file_name="disagree.png",
    )


def test_dual_path_small_random_translate_disagrees(ts) -> None:
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
    assert int(keypoint.evidence["n_verified"]) == _LOCKED_VERIFIED
