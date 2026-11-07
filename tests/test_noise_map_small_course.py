"""Course lock: the global noise map refuses an image smaller than two blocks."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.noise_map import NoiseMap


def test_noise_map_too_small_is_not_applicable() -> None:
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
    result = NoiseMap().analyze(doc, load_thresholds())
    assert result.method is MethodName.NOISE_INCONSISTENCY
    assert result.verdict is Verdict.NOT_APPLICABLE
    assert result.score is None
    assert result.limitations[0] == "ảnh 8×8 quá nhỏ cho block 16px"
