"""Course lock: a flat patch has no keypoints and stays inconclusive."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.keypoint.pipeline import KeypointCopyMove


def test_flat_patch_keypoint_is_insufficient_texture() -> None:
    doc = ImageDocument(
        array=np.full((32, 32, 3), 40, np.uint8),
        width=32,
        height=32,
        channels=3,
        image_format="png",
        sha256="e" * 64,
        file_bytes=3,
        file_name="e.png",
    )
    result = KeypointCopyMove().analyze(doc, load_thresholds())
    assert result.method is MethodName.COPY_MOVE_KEYPOINT
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.score == 0
    assert result.evidence == {"n_keypoints": 0}
    assert result.limitations == [
        (
            "insufficient texture: 0 keypoints < k_min=50 — "
            "ảnh quá phẳng/mờ để keypoint method kết luận"
        )
    ]
