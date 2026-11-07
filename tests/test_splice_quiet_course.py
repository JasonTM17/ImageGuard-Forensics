"""Course lock: a flat JPEG cannot splice because the noise map has no outlier."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.splicing import SplicingCues


def test_flat_jpeg_splicing_lacks_a_noise_outlier() -> None:
    doc = ImageDocument(
        array=np.full((64, 64, 3), 80, np.uint8),
        width=64,
        height=64,
        channels=3,
        image_format="jpeg",
        sha256="e" * 64,
        file_bytes=3,
        file_name="e.jpg",
    )
    result = SplicingCues().analyze(doc, load_thresholds())
    assert result.method is MethodName.SPLICING_CUES
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.score is None
    assert result.evidence == {
        "noise_verdict": "no_indication",
        "ela_verdict": "no_indication",
        "cue_unavailable": True,
    }
    assert result.limitations[0] == (
        "cue suy giảm: noise map không có outlier — "
        "không đủ cặp tín hiệu để đối chiếu"
    )
