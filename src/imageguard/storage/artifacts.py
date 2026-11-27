"""Artifact store — file outputs theo analysis_id, DB chỉ lưu relative path.

Layout: <root>/<analysis_id>/<kind>.<ext>  (kind: mask|heatmap|ela|pairs|report)
Guard: filename sanitize — cấm `..`/path separator để chặn traversal.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

from imageguard.core.paths import data_root

# Tên dir report chuẩn: str(uuid4()) — sweep mồ côi chỉ công nhận dạng này.
_REPORT_DIR_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)

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
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else data_root() / "artifacts"

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
        # Reports may move between Windows and POSIX. A drive-relative path
        # (C:file), rooted path (\file), or UNC path is unsafe on either host;
        # is_absolute() alone does not reject the first two on Windows.
        if not rel or PureWindowsPath(rel).anchor or Path(rel).is_absolute():
            raise ValueError(f"rel_path thoát khỏi artifact root: {rel!r}")
        root = self.root.resolve()
        # Accept both separator spellings for safe relative paths, so a
        # backslash traversal cannot turn into a harmless filename on POSIX.
        p = (root / rel.replace("\\", "/")).resolve()
        if p == root or not _contained(root, p):
            raise ValueError(f"rel_path thoát khỏi artifact root: {rel!r}")
        return p

    def read(self, rel: str) -> bytes:
        return self.abs_path(rel).read_bytes()

    def _reserve_dir(self, analysis_id: str) -> None:
        """Tạo độc quyền <root>/<analysis_id> trước persist.

        Dùng chung guards với put/delete; mkdir không exist_ok → raise
        FileExistsError khi dir đã có (không bao giờ đụng dữ liệu có sẵn).
        """
        _check_name(analysis_id)
        root = self.root.resolve()
        d = (root / analysis_id).resolve()
        if d == root or not _contained(root, d):
            raise ValueError(
                f"analysis_id thoát khỏi artifact root: {analysis_id!r}")
        d.parent.mkdir(parents=True, exist_ok=True)
        try:
            d.mkdir()
        except FileExistsError:
            raise FileExistsError(
                f"artifact dir cho report {analysis_id} đã tồn tại") from None

    def delete_analysis(self, analysis_id: str) -> None:
        _check_name(analysis_id)  # chặn "..", path separator, absolute path
        root = self.root.resolve()
        d = (root / analysis_id).resolve()
        if d != root and _contained(root, d) and d.is_dir():
            shutil.rmtree(d)

    def sweep_orphan_report_dirs(
        self, known_ids: set[str]
    ) -> tuple[list[str], list[str]]:
        """Xóa dir report-id UUID không còn row Analysis nào trỏ tới.

        Crash giữa ``_reserve_dir`` và DB write để lại dir mồ côi; startup
        gọi hàm này để thu hồi. Chỉ đụng **con trực tiếp** của root và tên
        đúng dạng UUID (str(uuid4)) — file/dir người dùng đặt vào artifacts
        root không bao giờ bị coi là mồ côi. Trả ``(swept, skipped)``; lỗi
        liệt kê root → trả rỗng, KHÔNG xóa gì (fail-closed).
        """
        swept: list[str] = []
        skipped: list[str] = []
        try:
            entries = list(self.root.iterdir())
        except OSError:
            return swept, skipped
        for entry in entries:
            name = entry.name
            if name in known_ids or not _REPORT_DIR_RE.fullmatch(name):
                continue
            if not entry.is_dir():
                continue
            try:
                shutil.rmtree(entry)
            except OSError:
                # File bị khóa (AV/viewer) — để lại cho lần sweep sau.
                skipped.append(name)
                continue
            swept.append(name)
        return swept, skipped
