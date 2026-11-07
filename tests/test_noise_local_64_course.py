"""Course lock: 64px is still too small for the noise-local neighborhood."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.noise_local import NoiseLocal


def test_64px_field_is_too_small_for_noise_local() -> None:
    doc = ImageDocument(
        array=np.full((64, 64, 3), 80, np.uint8),
        width=64,
        height=64,
        channels=3,
        image_format="png",
        sha256="e" * 64,
        file_bytes=3,
        file_name="n.png",
    )
    result = NoiseLocal().analyze(doc, load_thresholds())
    assert result.method is MethodName.NOISE_LOCAL
    assert result.verdict is Verdict.NOT_APPLICABLE
    assert result.score is None
    assert result.mask is None
    assert result.evidence == {"variant": "two_tier_low_sigma"}
    assert result.limitations[0] == (
        "ảnh quá nhỏ cho block 16px + láng giềng 5"
    )
