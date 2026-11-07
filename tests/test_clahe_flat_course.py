"""Course lock: CLAHE on a flat 8x8 patch expands the single gray level to 255."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.models import ImageDocument
from imageguard.core.preprocess import preprocess_keypoint


def test_flat_patch_clahe_is_uniform_255() -> None:
    doc = ImageDocument(
        array=np.full((8, 8, 3), 22, np.uint8),
        width=8,
        height=8,
        channels=3,
        image_format="png",
        sha256="e" * 64,
        file_bytes=3,
        file_name="e.png",
    )
    gray = preprocess_keypoint(doc, load_thresholds())
    assert gray.shape == (8, 8)
    assert gray.dtype == np.uint8
    assert np.unique(gray).tolist() == [255]
