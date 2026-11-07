"""Course lock: the block preprocessor median removes an isolated spike."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.models import ImageDocument
from imageguard.core.preprocess import preprocess_block


def test_median_preprocess_removes_single_white_pixel() -> None:
    spike = np.zeros((8, 8, 3), np.uint8)
    spike[4, 4] = (255, 255, 255)
    doc = ImageDocument(
        array=spike,
        width=8,
        height=8,
        channels=3,
        image_format="png",
        sha256="e" * 64,
        file_bytes=3,
        file_name="s.png",
    )
    gray = preprocess_block(doc, load_thresholds())
    assert gray.shape == (8, 8)
    assert gray.dtype == np.uint8
    assert int(gray[4, 4]) == 0
    assert int(gray.sum()) == 0
