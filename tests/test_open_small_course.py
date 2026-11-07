"""Course lock: an 8x8 image is below the forensic minimum side."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from imageguard.core.errors import ErrorCode, ErrorResult
from imageguard.core.io import safe_open


def test_eight_pixel_png_is_too_small(tmp_path: Path) -> None:
    path = tmp_path / "tiny.png"
    cv2.imwrite(str(path), np.full((8, 8, 3), 40, np.uint8))
    result = safe_open(path)
    assert isinstance(result, ErrorResult)
    assert result.code == ErrorCode.IMAGE_TOO_SMALL
    assert result.message == (
        "Ảnh 8×8 quá nhỏ (min_side=64) — không đủ tín hiệu forensic"
    )
    assert result.detail == {"width": 8, "height": 8}
