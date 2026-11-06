"""Course lock: JPEG splice yields ELA×noise overlap without a conviction.

Smooth seed-23 base saved at JPEG Q95, a Q50 noise patch pasted in, then
re-saved at Q75. SplicingCues must stay below SUSPICIOUS_INDICATION, keep a
cue mask, and lock overlap_frac == 1.0 on this fixture.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.splicing import SplicingCues

_LOCKED_OVERLAP = 1.0


def test_splicing_jpeg_paste_locks_full_ela_noise_overlap(tmp_path: Path, ts) -> None:
    rng = np.random.default_rng(23)
    h = w = 256
    ramp = np.linspace(40.0, 220.0, w, dtype=np.float64)
    base = np.stack([np.broadcast_to(ramp, (h, w))] * 3, axis=-1)
    base = np.clip(base + rng.normal(0.0, 3.0, base.shape), 0, 255).astype(np.uint8)

    q95_path = tmp_path / "base-q95.jpg"
    assert cv2.imwrite(str(q95_path), base, [cv2.IMWRITE_JPEG_QUALITY, 95])
    image = cv2.imread(str(q95_path))
    assert image is not None

    patch = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    patch_path = tmp_path / "patch-q50.jpg"
    assert cv2.imwrite(str(patch_path), patch, [cv2.IMWRITE_JPEG_QUALITY, 50])
    patch_jpeg = cv2.imread(str(patch_path))
    assert patch_jpeg is not None
    image[40:120, 40:120] = patch_jpeg

    paste_path = tmp_path / "paste-q75.jpg"
    assert cv2.imwrite(str(paste_path), image, [cv2.IMWRITE_JPEG_QUALITY, 75])

    doc = safe_open(paste_path, ts)
    assert isinstance(doc, ImageDocument)
    result = SplicingCues().analyze(doc, ts)

    assert result.verdict != Verdict.SUSPICIOUS_INDICATION
    assert result.mask is not None
    assert int(np.count_nonzero(result.mask)) > 0
    assert result.evidence.get("cue_unavailable") is not True
    assert float(result.evidence["overlap_frac"]) == _LOCKED_OVERLAP
