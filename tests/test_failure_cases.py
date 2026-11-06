"""Course failure-case limitations — one executable lock plus citations.

Locked elsewhere (citation only; do not duplicate assertions here):

- ELA on PNG is ``NOT_APPLICABLE`` — ``tests/test_ela.py::test_png_not_applicable``.
- Flat / low-texture keypoint cannot support a copy-move line of analysis —
  ``tests/test_keypoint.py::test_low_texture_inconclusive``.

This module adds an explicit check that splicing cues on a non-JPEG document stay
``INCONCLUSIVE`` without a cue mask when ELA cannot supply a mask.
"""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.splicing import SplicingCues


def test_splicing_nonjpeg_inconclusive_no_cue_mask(ts) -> None:
    """In-memory PNG + default ``ThresholdSet``: no ELA cue pair → no mask."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    doc = ImageDocument(
        array=arr,
        width=64,
        height=64,
        channels=3,
        image_format="png",
        sha256="a" * 64,
        file_bytes=0,
        file_name="in_memory.png",
    )
    r = SplicingCues().analyze(doc, ts)
    assert r.verdict == Verdict.INCONCLUSIVE
    assert r.mask is None
    assert r.evidence.get("cue_unavailable") is True
