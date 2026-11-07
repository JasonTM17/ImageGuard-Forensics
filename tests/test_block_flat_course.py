"""Course lock: a uniform field yields no block features and no indication."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.block.pipeline import BlockCopyMove


def test_uniform_field_block_has_no_features() -> None:
    doc = ImageDocument(
        array=np.full((64, 64, 3), 80, np.uint8),
        width=64,
        height=64,
        channels=3,
        image_format="png",
        sha256="e" * 64,
        file_bytes=3,
        file_name="n.png",
    )
    result = BlockCopyMove().analyze(doc, load_thresholds())
    assert result.method is MethodName.COPY_MOVE_BLOCK
    assert result.verdict is Verdict.NO_INDICATION
    assert result.score == 0.0
    assert int(np.count_nonzero(result.mask)) == 0
    assert result.evidence == {
        "n_blocks": 0,
        "n_candidate_pairs": 0,
        "n_shifts_kept": 0,
        "top_shift_votes": 0,
        "work_size": [64, 64],
    }