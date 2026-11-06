"""Course lock: an unforged seed-23 noise control stays negative end to end."""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.block.pipeline import BlockCopyMove
from imageguard.forensics.fusion import combine
from imageguard.forensics.keypoint.pipeline import KeypointCopyMove

_SIZE = 160


def _doc(bgr: np.ndarray) -> ImageDocument:
    h, w = bgr.shape[:2]
    return ImageDocument(
        array=bgr,
        width=w,
        height=h,
        channels=3,
        image_format="png",
        sha256="p" * 64,
        file_bytes=0,
        file_name="control_fusion.png",
    )


def test_unforged_noise_control_keeps_dual_path_no_indication(ts) -> None:
    rng = np.random.default_rng(23)
    noise = rng.integers(0, 255, (_SIZE, _SIZE, 3), dtype=np.uint8)
    doc = _doc(noise)

    block = BlockCopyMove().analyze(doc, ts)
    keypoint = KeypointCopyMove().analyze(doc, ts)

    assert block.verdict == Verdict.NO_INDICATION
    assert keypoint.verdict == Verdict.NO_INDICATION

    overall, signals, warnings = combine([block, keypoint])
    assert overall == Verdict.NO_INDICATION
    assert signals == []
    assert warnings == []
