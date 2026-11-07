"""Course lock: a missing path is PATH_INVALID and names only the file."""

from __future__ import annotations

from pathlib import Path

from imageguard.core.errors import ErrorCode, ErrorResult
from imageguard.core.io import safe_open


def test_missing_file_names_only_the_basename(tmp_path: Path) -> None:
    path = tmp_path / "no-such-imageguard-file.png"
    result = safe_open(path)
    assert isinstance(result, ErrorResult)
    assert result.code == ErrorCode.PATH_INVALID
    assert result.message == (
        "Không phải file tồn tại: no-such-imageguard-file.png"
    )
    assert result.detail == {}
