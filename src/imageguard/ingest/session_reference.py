"""Session reference store — một ảnh tham chiếu trong phiên làm việc.

Operator chụp/lấy một khung hình làm mốc (reference) rồi phân tích các khung
sau đối chiếu với nó. Store giữ đúng hai file dưới ``data/session_ref/``:

- ``reference.png``  — bản copy bytes của file nguồn (không giải mã lại,
  giữ nguyên dấu vết cảm biến).
- ``reference.json`` — sidecar tối thiểu ``{sha256, basename}``: đủ để
  evidence đối chiếu mà không lưu đường dẫn tuyệt đối.

Mọi ghi đều resolve + kiểm tra containment trong root — path nào thoát khỏi
root bị reject (guard chung với ``storage.artifacts``).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path

from imageguard.core.paths import data_root

REFERENCE_IMAGE = "reference.png"
REFERENCE_SIDECAR = "reference.json"


def _contained(root: Path, candidate: Path) -> bool:
    """True iff candidate == root hoặc nằm bên trong root (kể cả khác prefix string)."""
    r = os.path.normcase(str(root.resolve()))
    p = os.path.normcase(str(candidate.resolve()))
    return p == r or p.startswith(r + os.sep)


def _atomic_write(dst: Path, data: bytes) -> None:
    """Ghi qua file tạm cùng thư mục + os.replace — crash giữa chừng không
    để lại reference.png cắt nửa cho lần phân tích sau."""
    tmp = dst.parent / f".{dst.name}.{secrets.token_hex(8)}.tmp"
    try:
        tmp.write_bytes(data)
        os.replace(tmp, dst)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


class SessionReferenceStore:
    """Giữ/đọc/xoá ảnh tham chiếu phiên trong một root cố định."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else data_root() / "session_ref"

    def _inside_root(self, name: str) -> Path:
        """Resolve ``root/name`` và reject kết quả thoát khỏi root."""
        root = self.root.resolve()
        p = (root / name).resolve()
        if p == root or not _contained(root, p):
            raise ValueError(f"reference path thoát khỏi session_ref root: {name!r}")
        return p

    def set_reference(self, source: Path) -> Path:
        """Copy bytes ``source`` → ``root/reference.png`` + ghi sidecar sha256/basename."""
        src = Path(source)
        data = src.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        dst = self._inside_root(REFERENCE_IMAGE)
        _atomic_write(dst, data)
        side = self._inside_root(REFERENCE_SIDECAR)
        _atomic_write(
            side,
            json.dumps(
                {"sha256": sha, "basename": src.name},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        return dst

    def get_reference(self) -> Path | None:
        """Đường dẫn ảnh tham chiếu hiện có, hoặc None khi chưa đặt."""
        p = self._inside_root(REFERENCE_IMAGE)
        return p if p.is_file() else None

    def sidecar(self) -> dict | None:
        """Đọc ``reference.json`` (sha256 + basename) — None nếu thiếu/hỏng."""
        p = self._inside_root(REFERENCE_SIDECAR)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def clear(self) -> None:
        """Xoá reference.png + reference.json; no-op khi chưa tồn tại."""
        for name in (REFERENCE_IMAGE, REFERENCE_SIDECAR):
            self._inside_root(name).unlink(missing_ok=True)
