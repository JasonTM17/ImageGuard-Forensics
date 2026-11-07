"""Course lock: a small document is the aux working copy itself."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.models import ImageDocument
from imageguard.core.preprocess import aux_working_copy


def test_aux_working_copy_small_image_is_identity() -> None:
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
    work, scale = aux_working_copy(doc, load_thresholds())
    assert scale == 1.0
    assert work is doc
    assert work.resolution_reduced is False
    assert work.scale_factor == 1.0
