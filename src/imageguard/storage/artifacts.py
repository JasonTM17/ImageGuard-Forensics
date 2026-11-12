"""Artifact store — file outputs theo analysis_id, DB chỉ lưu relative path.

Layout: <root>/<analysis_id>/<kind>.<ext>  (kind: mask|heatmap|ela|pairs|report)
Guard: filename sanitize — cấm `..`/path separator để chặn traversal.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
from pathlib import Path

import cv2
import numpy as np

from imageguard.core.paths import data_root

DEFAULT_ROOT = data_root() / "artifacts"

_SAFE = re.compile(r"[a-z0-9_\-]+", re.IGNORECASE | re.ASCII)
_SAFE_FILE = re.compile(
    r"[a-z0-9_\-]+(\.[a-z0-9]{1,8})?", re.IGNORECASE | re.ASCII,
)


def _check_name(name: str) -> str:
    if _SAFE.fullmatch(name) is None:
        raise ValueError(f"artifact name không hợp lệ: {name!r}")
    return name


def _check_filename(filename: str) -> str:
    if _SAFE_FILE.fullmatch(filename) is None:
        raise ValueError(f"artifact filename không hợp lệ: {filename!r}")
    return filename


def _contained(root: Path, candidate: Path) -> bool:
    """True iff candidate == root hoặc nằm bên trong root (kể cả khác prefix string)."""
    r = os.path.normcase(str(root.resolve()))
    p = os.path.normcase(str(candidate.resolve()))
    return p == r or p.startswith(r + os.sep)


class ArtifactStore:
    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def rel_path(self, analysis_id: str, filename: str) -> str:
        _check_name(analysis_id)
        _check_filename(filename)
        return f"{analysis_id}/{filename}"

    def put_bytes(self, analysis_id: str, filename: str, data: bytes) -> str:
        rel = self.rel_path(analysis_id, filename)
        root = self.root.resolve()
        dst = (root / rel).resolve()
        if not _contained(root, dst):
            raise ValueError(f"rel_path thoát khỏi artifact root: {rel!r}")
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        dst = (root / rel).resolve()
        if not _contained(root, dst):
            raise ValueError(f"rel_path thoát khỏi artifact root: {rel!r}")
        tmp_path: Path | None = None
        fd = -1
        try:
            for _ in range(64):
                candidate = dst.parent / f".{secrets.token_hex(16)}.tmp"
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                try:
                    fd = os.open(candidate, flags, 0o600)
                    tmp_path = candidate
                    break
                except FileExistsError:
                    continue
            else:
                raise OSError("could not allocate temporary artifact file")
            with os.fdopen(fd, "wb") as stream:
                fd = -1
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, dst)
            tmp_path = None
        except BaseException:
            if fd >= 0:
                os.close(fd)
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        return rel

    def put_image(self, analysis_id: str, filename: str, img: np.ndarray) -> str:
        ok, buf = cv2.imencode(Path(filename).suffix or ".png", img)
        if not ok:
            raise ValueError(f"không encode được {filename}")
        return self.put_bytes(analysis_id, filename, buf.tobytes())

    def abs_path(self, rel: str) -> Path:
        if not rel or Path(rel).is_absolute():
            raise ValueError(f"rel_path thoát khỏi artifact root: {rel!r}")
        root = self.root.resolve()
        p = (root / rel).resolve()
        if not _contained(root, p):
            raise ValueError(f"rel_path thoát khỏi artifact root: {rel!r}")
        return p

    def read(self, rel: str) -> bytes:
        return self.abs_path(rel).read_bytes()

    def delete_analysis(self, analysis_id: str) -> None:
        _check_name(analysis_id)  # chặn "..", path separator, absolute path
        root = self.root.resolve()
        d = (root / analysis_id).resolve()
        if d != root and _contained(root, d) and d.is_dir():
            shutil.rmtree(d)
