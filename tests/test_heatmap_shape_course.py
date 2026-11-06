"""Course lock: aux heatmaps keep the source JPEG height and width.

Odd-sized seed-23 JPEG (173×211). ELA heatmap/mask and FreqView heatmap must
match ``(doc.height, doc.width)``. Neither verdict is a suspicious indication.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.ela import ELA
from imageguard.forensics.auxiliary.freq_view import FreqView


def test_aux_heatmaps_match_odd_jpeg_shape(tmp_path: Path, ts) -> None:
    rng = np.random.default_rng(23)
    array = rng.integers(0, 256, (173, 211, 3), dtype=np.uint8)
    path = tmp_path / "odd.jpg"
    assert cv2.imwrite(str(path), array, [cv2.IMWRITE_JPEG_QUALITY, 80])

    doc = safe_open(path, ts)
    assert isinstance(doc, ImageDocument)
    expected = (doc.height, doc.width)

    ela = ELA().analyze(doc, ts)
    freq = FreqView().analyze(doc, ts)

    assert ela.verdict != Verdict.SUSPICIOUS_INDICATION
    assert freq.verdict != Verdict.SUSPICIOUS_INDICATION
    assert ela.heatmap is not None and ela.heatmap.shape == expected
    assert ela.mask is not None and ela.mask.shape == expected
    assert freq.heatmap is not None and freq.heatmap.shape == expected
