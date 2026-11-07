"""Course lock: a flat PNG has zero JPEG blockiness and no suspicious frequency cue."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.freq_view import FreqView


def test_flat_png_frequency_blockiness_is_zero() -> None:
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
    result = FreqView().analyze(doc, load_thresholds())
    assert result.method is MethodName.FREQUENCY_ANALYSIS
    assert result.verdict is Verdict.NO_INDICATION
    assert result.score == 0.0
    assert result.evidence == {
        "blockiness": 0.0,
        "boundary_grad": 0.0,
        "inner_grad": 0.0,
        "format": "png",
        "window": 8,
        "scale": 1.0,
    }
