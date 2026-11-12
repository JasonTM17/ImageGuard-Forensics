"""safe_open edge cases — mỗi nhánh lỗi phải đúng ErrorCode, không raise."""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.config import ThresholdSet
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


def test_safe_open_uses_the_bytes_it_hashed(
    tmp_path: Path, ts, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_a = np.full((256, 256, 3), (10, 40, 200), dtype=np.uint8)
    image_b = np.full((256, 256, 3), (230, 150, 20), dtype=np.uint8)
    ok_a, encoded_a = cv2.imencode(".png", image_a)
    ok_b, encoded_b = cv2.imencode(".png", image_b)
    assert ok_a and ok_b
    raw_a = encoded_a.tobytes()
    raw_b = encoded_b.tobytes()
    path = tmp_path / "substituted.png"
    path.write_bytes(raw_a)

    original_open = Path.open

    class _ReplaceAfterRead:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            self._handle.close()

        def fileno(self) -> int:
            return self._handle.fileno()

        def read(self, size: int = -1) -> bytes:
            data = self._handle.read(size)
            with original_open(path, "wb") as out:
                out.write(raw_b)
            return data

    def open_then_replace(self: Path, *args, **kwargs):
        handle = original_open(self, *args, **kwargs)
        return _ReplaceAfterRead(handle) if self == path else handle

    # The path is rewritten to image B right after the read; the decoded and
    # hashed bytes must still be the ones that were read (image A).
    monkeypatch.setattr(Path, "open", open_then_replace)

    doc = safe_open(path, ts)

    assert isinstance(doc, ImageDocument)
    assert doc.sha256 == hashlib.sha256(raw_a).hexdigest()
    assert doc.file_bytes == len(raw_a)
    np.testing.assert_array_equal(doc.array, image_a)


def test_safe_open_rejects_read_buffer_over_max_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "oversized.png"
    path.write_bytes(b"x")
    cfg = ThresholdSet(raw={"limits": {"max_file_bytes": {"value": 3}}})
    original_open = Path.open

    class _Oversized:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            self._handle.close()

        def fileno(self) -> int:
            return self._handle.fileno()

        def read(self, size: int = -1) -> bytes:
            return b"four"

    def open_oversized(self: Path, *args, **kwargs):
        handle = original_open(self, *args, **kwargs)
        return _Oversized(handle) if self == path else handle

    monkeypatch.setattr(Path, "open", open_oversized)

    result = safe_open(path, cfg)

    assert isinstance(result, ErrorResult)
    assert result.code == ErrorCode.FILE_TOO_LARGE
