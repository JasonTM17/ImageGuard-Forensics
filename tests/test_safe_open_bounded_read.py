"""Bounded single-handle reads in safe_open — size probe and read share one fd."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.config import ThresholdSet
from imageguard.core.errors import ErrorCode, ErrorResult
from imageguard.core.io import safe_open


def _tiny_png_bytes() -> bytes:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_safe_open_accepts_file_exactly_at_max_bytes(tmp_path: Path) -> None:
    raw = _tiny_png_bytes()
    max_bytes = len(raw)
    path = tmp_path / "at_limit.png"
    path.write_bytes(raw)
    cfg = ThresholdSet(raw={"limits": {"max_file_bytes": {"value": max_bytes}}})

    result = safe_open(path, cfg)

    assert not isinstance(result, ErrorResult)
    assert result.file_bytes == max_bytes


def test_safe_open_rejects_one_byte_over_max_bytes(tmp_path: Path) -> None:
    raw = _tiny_png_bytes()
    max_bytes = len(raw) - 1
    path = tmp_path / "one_over.png"
    path.write_bytes(raw)
    cfg = ThresholdSet(raw={"limits": {"max_file_bytes": {"value": max_bytes}}})

    result = safe_open(path, cfg)

    assert isinstance(result, ErrorResult)
    assert result.code == ErrorCode.FILE_TOO_LARGE
    assert result.detail.get("size") == len(raw)


def test_safe_open_bounded_when_fstat_under_reports_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _tiny_png_bytes()
    max_bytes = len(raw) - 1
    path = tmp_path / "lying_stat.png"
    path.write_bytes(raw)
    cfg = ThresholdSet(raw={"limits": {"max_file_bytes": {"value": max_bytes}}})
    read_caps: list[int] = []
    real_fstat = os.fstat
    real_open = Path.open

    def fake_fstat(fd: int, /) -> os.stat_result:
        st = real_fstat(fd)
        if fd == tracked_fd[0]:
            return os.stat_result(
                (
                    st.st_mode,
                    st.st_ino,
                    st.st_dev,
                    st.st_nlink,
                    st.st_uid,
                    st.st_gid,
                    1,
                    st.st_atime,
                    st.st_mtime,
                    st.st_ctime,
                )
            )
        return st

    tracked_fd: list[int | None] = [None]

    def counting_open(self: Path, *args: object, **kwargs: object):
        f = real_open(self, *args, **kwargs)
        if self == path:
            tracked_fd[0] = f.fileno()
        original_read = f.read

        def bounded_read(size: int = -1, /) -> bytes:
            if size < 0:
                size = max_bytes + 1
            cap = min(size, max_bytes + 1)
            read_caps.append(cap)
            return original_read(cap)

        f.read = bounded_read  # type: ignore[method-assign]
        return f

    monkeypatch.setattr(os, "fstat", fake_fstat)
    monkeypatch.setattr(Path, "open", counting_open)

    result = safe_open(path, cfg)

    assert isinstance(result, ErrorResult)
    assert result.code == ErrorCode.FILE_TOO_LARGE
    assert read_caps
    assert max(read_caps) <= max_bytes + 1


def test_safe_open_unreadable_path_returns_file_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "gone.png"
    path.write_bytes(_tiny_png_bytes())
    real_open = Path.open

    def fail_open(self: Path, *args: object, **kwargs: object):
        if self == path:
            raise OSError(13, "Permission denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_open)

    result = safe_open(path)

    assert isinstance(result, ErrorResult)
    assert result.code == ErrorCode.FILE_UNREADABLE
