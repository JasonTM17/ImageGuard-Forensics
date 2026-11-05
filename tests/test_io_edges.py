"""safe_open edge cases — mỗi nhánh lỗi phải đúng ErrorCode, không raise."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from imageguard.core.errors import ErrorCode, ErrorResult
from imageguard.core.io import safe_open
from imageguard.core.models import ImageDocument


def _png_header_only(width: int, height: int) -> bytes:
    """PNG chỉ có vỉa: signature + IHDR(width×height) + IEND, KHÔNG pixel data.

    IHDR khai 20000×20000 → Pillow ném DecompressionBombError ngay khi đọc
    header; dùng để chứng minh header probe phải trả ErrorResult, không raise.
    """
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_chunk = (
        struct.pack(">I", len(ihdr))
        + b"IHDR"
        + ihdr
        + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    )
    iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND"))
    return sig + ihdr_chunk + iend_chunk


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


def test_png_header_bomb_returns_error(tmp_path: Path, ts) -> None:
    # 45-byte PNG vỉa, IHDR khai 20000×20000 — Pillow bắn DecompressionBombError
    # trong Image.open (header probe), safe_open phải trả TOO_MANY_PIXELS.
    p = tmp_path / "bomb.png"
    p.write_bytes(_png_header_only(20000, 20000))
    assert p.stat().st_size == 45  # vỉa tối thiểu, không có IDAT
    r = safe_open(p, ts)
    assert isinstance(r, ErrorResult)
    assert r.code == ErrorCode.TOO_MANY_PIXELS


def test_exif_dict_returned(valid_jpg: Path, ts) -> None:
    doc = safe_open(valid_jpg, ts)
    assert isinstance(doc, ImageDocument)
    assert "present" in doc.exif and "missing" in doc.exif
