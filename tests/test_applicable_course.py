"""Course lock: JPEG-only methods are not applicable to other formats."""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.io import check_applicable
from imageguard.core.models import ImageDocument


def _doc(fmt: str) -> ImageDocument:
    return ImageDocument(
        array=np.zeros((8, 8, 3), np.uint8),
        width=8,
        height=8,
        channels=3,
        image_format=fmt,
        sha256="d" * 64,
        file_bytes=1,
        file_name=f"t.{fmt}",
    )


def test_check_applicable_jpeg_gate() -> None:
    assert check_applicable(_doc("png"), True) is Verdict.NOT_APPLICABLE
    assert check_applicable(_doc("jpeg"), True) is None
    assert check_applicable(_doc("png"), False) is None
