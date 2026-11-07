"""Course lock: upscale_to_doc is a no-op on a matching shape and nearest-upsamples otherwise."""

from __future__ import annotations

import numpy as np

from imageguard.core.models import ImageDocument
from imageguard.core.preprocess import upscale_to_doc


def test_upscale_to_doc_noop_and_nearest() -> None:
    doc = ImageDocument(
        array=np.zeros((4, 6, 3), np.uint8),
        width=6,
        height=4,
        channels=3,
        image_format="png",
        sha256="a" * 64,
        file_bytes=0,
        file_name="x.png",
    )
    matched = np.zeros((4, 6), np.uint8)
    assert upscale_to_doc(matched, doc) is matched
    small = np.array([[0, 255], [128, 200]], np.uint8)
    out = upscale_to_doc(small, doc)
    assert out.shape == (4, 6)
    assert out.tolist() == [
        [0, 0, 0, 255, 255, 255],
        [0, 0, 0, 255, 255, 255],
        [128, 128, 128, 200, 200, 200],
        [128, 128, 128, 200, 200, 200],
    ]
