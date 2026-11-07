"""Course lock: a tiny PNG degrades splicing to an unavailable cue pair."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.splicing import SplicingCues


def test_tiny_png_splicing_cue_is_unavailable() -> None:
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
    result = SplicingCues().analyze(doc, load_thresholds())
    assert result.method is MethodName.SPLICING_CUES
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.score is None
    assert result.evidence == {
        "noise_verdict": "not_applicable",
        "ela_verdict": "not_applicable",
        "cue_unavailable": True,
    }
    assert result.limitations[0] == (
        "cue suy giảm: ELA không áp dụng (non-JPEG) — "
        "không đủ cặp tín hiệu để đối chiếu"
    )
