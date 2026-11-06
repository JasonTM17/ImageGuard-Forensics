"""Course lock: live aux analyzers alone cannot convict a JPEG paste.

A seed-23 JPEG paste (Q95 then patch then Q75) is analyzed by ELA, NoiseMap,
and FreqView. Fusion of those three results must stay below
SUSPICIOUS_INDICATION — aux cues corroborate at most.
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
from imageguard.forensics.auxiliary.noise_map import NoiseMap
from imageguard.forensics.fusion import combine


def test_aux_only_fusion_never_suspicious_on_jpeg_paste(tmp_path: Path, ts) -> None:
    rng = np.random.default_rng(23)
    base = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    q95_path = tmp_path / "base-q95.jpg"
    assert cv2.imwrite(str(q95_path), base, [cv2.IMWRITE_JPEG_QUALITY, 95])

    image = cv2.imread(str(q95_path))
    assert image is not None
    image[60:100, 60:100] = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    paste_path = tmp_path / "paste-q75.jpg"
    assert cv2.imwrite(str(paste_path), image, [cv2.IMWRITE_JPEG_QUALITY, 75])

    doc = safe_open(paste_path, ts)
    assert isinstance(doc, ImageDocument)

    results = [
        ELA().analyze(doc, ts),
        NoiseMap().analyze(doc, ts),
        FreqView().analyze(doc, ts),
    ]
    for result in results:
        assert result.verdict != Verdict.SUSPICIOUS_INDICATION

    overall, _signals, _warnings = combine(results)
    assert overall != Verdict.SUSPICIOUS_INDICATION
