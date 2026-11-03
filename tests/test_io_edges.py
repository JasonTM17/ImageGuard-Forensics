"""safe_open edge cases — mỗi nhánh lỗi phải đúng ErrorCode, không raise."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imageguard.core.errors import ErrorCode, ErrorResult
from imageguard.core.io import safe_open
from imageguard.core.models import ImageDocument


def test_open_valid_png(valid_png: Path, ts) -> None:
    doc = safe_open(valid_png, ts)
    assert isinstance(doc, ImageDocument)
    assert doc.array.shape == (256, 256, 3)
    assert doc.image_format == "png"
    assert len(doc.sha256) == 64


def test_open_valid_jpeg(valid_jpg: Path, ts) -> None:
    doc = safe_open(valid_jpg, ts)
    assert isinstance(doc, ImageDocument)
    assert doc.image_format == "jpeg"


def test_missing_file(ts) -> None:
    r = safe_open("khong_ton_tai.png", ts)
    assert isinstance(r, ErrorResult) and r.code == ErrorCode.PATH_INVALID


def test_unsupported_format(tmp_path: Path, ts) -> None:
    p = tmp_path / "fake.png"
    p.write_bytes(b"this is not an image at all")
    r = safe_open(p, ts)
    assert isinstance(r, ErrorResult) and r.code == ErrorCode.UNSUPPORTED_FORMAT


def test_truncated_jpeg(tmp_path: Path, valid_jpg: Path, ts) -> None:
    p = tmp_path / "trunc.jpg"
    p.write_bytes(valid_jpg.read_bytes()[: valid_jpg.stat().st_size // 3])
    r = safe_open(p, ts)
    # Truncated: header probe có thể đọc được dims nhưng decode fail,
    # hoặc Pillow vẫn decode phần đọc được — cả hai đều chấp nhận, miễn không crash.
    assert isinstance(r, (ErrorResult, ImageDocument))


def test_too_many_pixels(tmp_path: Path, ts, monkeypatch: pytest.MonkeyPatch) -> None:
    # Khai header lớn mà không cần tạo ảnh thật — monkeypatch ngưỡng xuống.
    import imageguard.core.limits as lim

    monkeypatch.setattr(lim, "MAX_DECODE_SIDE", 100)
    img = np.zeros((256, 256, 3), np.uint8)
    p = tmp_path / "big.png"
    import cv2

    cv2.imwrite(str(p), img)
    r = safe_open(p, ts)
    assert isinstance(r, ErrorResult) and r.code == ErrorCode.TOO_MANY_PIXELS


def test_exif_dict_returned(valid_jpg: Path, ts) -> None:
    doc = safe_open(valid_jpg, ts)
    assert isinstance(doc, ImageDocument)
    assert "present" in doc.exif and "missing" in doc.exif
