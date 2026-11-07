"""Course lock: a working copy smaller than one block cannot vote a copy-move."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.block.pipeline import BlockCopyMove


def test_block_smaller_than_window_is_inconclusive() -> None:
    doc = ImageDocument(
        array=np.full((8, 8, 3), 40, np.uint8),
        width=8,
        height=8,
        channels=3,
        image_format="png",
        sha256="e" * 64,
        file_bytes=3,
        file_name="e.png",
    )
    result = BlockCopyMove().analyze(doc, load_thresholds())
    assert result.method is MethodName.COPY_MOVE_BLOCK
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.score == 0.0
    assert result.evidence == {}
    assert result.limitations == [
        "working copy 8×8 < block 16px — không trích được block"
    ]
