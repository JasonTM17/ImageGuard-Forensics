"""Course lock: a small document stays full resolution in the working copy."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.models import ImageDocument
from imageguard.core.preprocess import to_gray, working_copy


def test_working_copy_small_image_scale_is_one() -> None:
    bgr = np.zeros((8, 8, 3), np.uint8)
    bgr[:] = (10, 20, 30)
    doc = ImageDocument(
        array=bgr.copy(),
        width=8,
        height=8,
        channels=3,
        image_format="png",
        sha256="b" * 64,
        file_bytes=1,
        file_name="t.png",
    )
    gray, scale = working_copy(doc, load_thresholds())
    assert scale == 1.0
    assert doc.resolution_reduced is False
    assert doc.scale_factor == 1.0
    assert gray.shape == (8, 8)
    assert int(gray[0, 0]) == 22
    assert np.array_equal(gray, to_gray(bgr))
