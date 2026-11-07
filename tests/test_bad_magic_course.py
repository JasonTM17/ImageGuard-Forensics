"""Course lock: unknown magic bytes are an unsupported format."""

from __future__ import annotations

from pathlib import Path

from imageguard.core.errors import ErrorCode, ErrorResult
from imageguard.core.io import safe_open


def test_unknown_magic_is_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "note.bin"
    path.write_bytes(b"NOTANIMAGEFORMAT!")
    result = safe_open(path)
    assert isinstance(result, ErrorResult)
    assert result.code == ErrorCode.UNSUPPORTED_FORMAT
    assert result.message == (
        "Magic bytes không khớp định dạng hỗ trợ (jpeg/png/bmp/tiff/webp)"
    )
    assert result.detail == {"head": "4e4f54414e494d414745464f524d4154"}
