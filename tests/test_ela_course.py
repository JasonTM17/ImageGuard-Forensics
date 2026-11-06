"""ELA paste-order behavior for a JPEG re-save."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.ela import ELA


def test_ela_paste_order_q95_then_q75(tmp_path: Path, ts) -> None:
    rng = np.random.default_rng(23)
    base = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    first_path = tmp_path / "first.jpg"
    assert cv2.imwrite(str(first_path), base, [cv2.IMWRITE_JPEG_QUALITY, 95])

    image = cv2.imread(str(first_path))
    assert image is not None
    patch_source = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    image[60:100, 60:100] = patch_source[60:100, 60:100]

    second_path = tmp_path / "pasted.jpg"
    assert cv2.imwrite(str(second_path), image, [cv2.IMWRITE_JPEG_QUALITY, 75])

    doc = safe_open(second_path, ts)
    assert isinstance(doc, ImageDocument)
    result = ELA().analyze(doc, ts)
    assert result.heatmap is not None

    inside_mean = float(result.heatmap[60:100, 60:100].mean())
    outside_mask = np.ones(result.heatmap.shape, dtype=bool)
    outside_mask[60:100, 60:100] = False
    outside_mean = float(result.heatmap[outside_mask].mean())
    assert inside_mean > outside_mean, (
        f"ELA paste order: inside mean {inside_mean:.8f}; "
        f"outside mean {outside_mean:.8f}"
    )
    # ELA is auxiliary and must not produce a suspicious verdict by itself.
    assert result.verdict != Verdict.SUSPICIOUS_INDICATION
