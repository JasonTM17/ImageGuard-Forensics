"""Course lock: a flat patch has a zero noise residual."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.models import ImageDocument
from imageguard.core.preprocess import preprocess_noise_residual


def test_flat_patch_noise_residual_is_zero() -> None:
    doc = ImageDocument(
        array=np.full((8, 8, 3), 40, np.uint8),
        width=8,
        height=8,
        channels=3,
        image_format="png",
        sha256="c" * 64,
        file_bytes=2,
        file_name="c.png",
    )
    residual = preprocess_noise_residual(doc, load_thresholds())
    assert residual.shape == (8, 8)
    assert residual.dtype == np.float32
    assert float(np.max(np.abs(residual))) == 0.0
