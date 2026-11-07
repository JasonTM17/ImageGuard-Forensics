"""Course lock: a flat JPEG has zero ELA p99 and stays no-indication."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.ela import ELA


def test_flat_jpeg_ela_p99_is_zero() -> None:
    doc = ImageDocument(
        array=np.full((32, 32, 3), 80, np.uint8),
        width=32,
        height=32,
        channels=3,
        image_format="jpeg",
        sha256="e" * 64,
        file_bytes=3,
        file_name="e.jpg",
    )
    result = ELA().analyze(doc, load_thresholds())
    assert result.method is MethodName.ELA
    assert result.verdict is Verdict.NO_INDICATION
    assert result.score == 0.0
    assert result.evidence == {
        "qualities": [75, 85, 90, 95],
        "p99": 0.0,
        "per_quality_p99": {75: 0.0, 85: 0.0, 90: 0.0, 95: 0.0},
    }
