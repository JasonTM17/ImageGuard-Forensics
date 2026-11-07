"""Course lock: ELA is not applicable on a lossless PNG document."""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.ela import ELA
from imageguard.forensics.auxiliary.freq_view import FreqView


def _png_doc() -> ImageDocument:
    array = np.full((64, 64, 3), 128, dtype=np.uint8)
    return ImageDocument(
        array=array,
        width=64,
        height=64,
        channels=3,
        image_format="png",
        sha256="p" * 64,
        file_bytes=0,
        file_name="flat.png",
    )


def test_ela_png_not_applicable_and_freq_never_suspicious(ts) -> None:
    doc = _png_doc()
    ela = ELA().analyze(doc, ts)
    freq = FreqView().analyze(doc, ts)

    assert ela.verdict == Verdict.NOT_APPLICABLE
    assert ela.verdict != Verdict.SUSPICIOUS_INDICATION
    assert freq.verdict != Verdict.SUSPICIOUS_INDICATION
    assert float(freq.evidence["blockiness"]) == 0.0
