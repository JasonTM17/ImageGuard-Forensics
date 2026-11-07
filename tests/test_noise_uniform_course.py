"""Course lock: a uniform field has no noise-map outliers."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.noise_map import NoiseMap


def test_uniform_field_noise_map_has_no_outliers() -> None:
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
    result = NoiseMap().analyze(doc, load_thresholds())
    assert result.method is MethodName.NOISE_INCONSISTENCY
    assert result.verdict is Verdict.NO_INDICATION
    assert result.score == 0.0
    assert result.mask is None
    assert result.evidence == {
        "sigma_median": 0.0,
        "sigma_mad": 0.0,
        "outlier_frac": 0.0,
        "block": 16,
        "scale": 1.0,
        "n_outlier_blocks": 0,
    }
