"""Shared fixtures — ảnh test sinh bằng cv2.imencode (không cần file vật lý thật
cho valid case) + tmp_path cho edge cases."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.config import load_thresholds


@pytest.fixture()
def ts():
    return load_thresholds()


@pytest.fixture()
def valid_png(tmp_path: Path) -> Path:
    img = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    p = tmp_path / "ảnh_test.png"  # unicode path — Windows hay gãy đây
    # cv2.imwrite fail âm thầm với unicode path trên Windows → imencode + write.
    p.write_bytes(cv2.imencode(".png", img)[1].tobytes())
    return p


@pytest.fixture()
def valid_jpg(tmp_path: Path) -> Path:
    img = np.random.default_rng(1).integers(0, 255, (300, 200, 3), dtype=np.uint8)
    p = tmp_path / "photo.jpg"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return p
