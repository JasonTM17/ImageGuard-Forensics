"""Course lock: a flat gray image cannot support a copy-move conviction."""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.block.pipeline import BlockCopyMove
from imageguard.forensics.keypoint.pipeline import KeypointCopyMove


def _flat_doc() -> ImageDocument:
    array = np.full((256, 256, 3), 128, dtype=np.uint8)
    return ImageDocument(
        array=array,
        width=256,
        height=256,
        channels=3,
        image_format="png",
        sha256="f" * 64,
        file_bytes=0,
        file_name="flat.png",
    )


def test_flat_image_keypoint_inconclusive_block_no_indication(ts) -> None:
    doc = _flat_doc()
    keypoint = KeypointCopyMove().analyze(doc, ts)
    block = BlockCopyMove().analyze(doc, ts)

    assert keypoint.verdict == Verdict.INCONCLUSIVE
    assert int(keypoint.evidence["n_keypoints"]) == 0
    assert block.verdict == Verdict.NO_INDICATION
    assert keypoint.verdict != Verdict.SUSPICIOUS_INDICATION
    assert block.verdict != Verdict.SUSPICIOUS_INDICATION
