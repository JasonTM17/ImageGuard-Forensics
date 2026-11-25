"""Session reference store — một ảnh tham chiếu trong phiên làm việc.

Operator chụp/lấy một khung hình làm mốc (reference) rồi phân tích các khung
sau đối chiếu với nó. Store giữ đúng hai file dưới ``data/session_ref/``:

- ``reference.png``  — bản PNG lossless đã qua ``safe_open``/EXIF transpose;
  pixel giữ nguyên nhưng metadata nhạy cảm như EXIF/GPS không được copy.
- ``reference.json`` — sidecar tối thiểu gồm sha256 bản chuẩn hoá, sha256 nguồn và lifetime
  UTC — không lưu basename hay đường dẫn nguồn. Reference chỉ được dùng khi hash còn khớp và chưa hết hạn.

Mọi tên file là basename cố định ngay dưới root. Không resolve symlink đích làm
đường dẫn ghi/xoá — symlink chỉ bị thay thế hoặc xoá ở chính entry trong root,
không bao giờ đi theo victim bên ngoài.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2

from imageguard.core import limits
from imageguard.core.errors import ErrorResult
from imageguard.core.io import safe_open
from imageguard.core.paths import data_root

REFERENCE_IMAGE = "reference.png"
REFERENCE_SIDECAR = "reference.json"
REFERENCE_SIDECAR_VERSION = 1
REFERENCE_TTL = timedelta(hours=8)
_SIDECAR_MAX_BYTES = 4096


def _atomic_write(dst: Path, data: bytes) -> None:
    """Ghi qua file tạm cùng thư mục + os.replace — crash giữa chừng không
    để lại reference.png cắt nửa cho lần phân tích sau."""
    if dst.is_symlink():
        dst.unlink()
    tmp = dst.parent / f".{dst.name}.{secrets.token_hex(8)}.tmp"
    try:
        tmp.write_bytes(data)
        os.replace(tmp, dst)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


class SessionReferenceStore:
    """Giữ/đọc/xoá ảnh tham chiếu phiên trong một root cố định."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else data_root() / "session_ref"

    def _inside_root(self, name: str) -> Path:
        """Trả đường dẫn lexical ngay dưới root; reject mọi basename không hợp lệ.

        Không resolve final component: nếu nó là symlink, caller có thể xoá hoặc
        thay chính symlink đó thay vì vô tình đụng file victim bên ngoài root.
        """
        if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError(f"reference path không phải basename an toàn: {name!r}")
        return self.root.resolve() / name

    def set_reference(self, source: Path) -> Path:
        """Chuẩn hoá ``source`` thành PNG lossless và lưu kèm lifetime/hash.

        ``safe_open`` chạy trước khi ghi để chặn định dạng không hỗ trợ, file
        quá lớn và decompression bomb. PNG encode là lossless ở pixel nên vi
        kết cấu cảm biến phục vụ cue vẫn được giữ; metadata EXIF/GPS không copy.
        """
        src = Path(source)
        doc = safe_open(src)
        if isinstance(doc, ErrorResult):
            raise ValueError(  # noqa: TRY004 - ErrorResult mã hoá nội dung ảnh không an toàn, không phải sai kiểu Python
                "ảnh tham chiếu không hợp lệ hoặc vượt giới hạn an toàn"
            )
        ok, encoded = cv2.imencode(".png", doc.array)
        if not ok:
            raise ValueError("không chuẩn hoá được ảnh tham chiếu")
        data = encoded.tobytes()
        if len(data) > limits.MAX_FILE_BYTES:
            raise ValueError("ảnh tham chiếu chuẩn hoá vượt giới hạn an toàn")
        sha = hashlib.sha256(data).hexdigest()
        now = datetime.now(UTC)
        root = self.root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        dst = self._inside_root(REFERENCE_IMAGE)
        _atomic_write(dst, data)
        side = self._inside_root(REFERENCE_SIDECAR)
        _atomic_write(
            side,
            json.dumps(
                {
                    "version": REFERENCE_SIDECAR_VERSION,
                    "sha256": sha,
                    "source_sha256": doc.sha256,
                    "created_at": now.isoformat(),
                    "expires_at": (now + REFERENCE_TTL).isoformat(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        return dst

    def _valid_stored_reference(self, path: Path) -> bool:
        try:
            if path.is_symlink() or not path.is_file():
                return False
            if path.stat().st_size > limits.MAX_FILE_BYTES:
                return False
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return False
        sidecar = self.sidecar()
        if sidecar is None:
            return False
        expected_keys = {"version", "sha256", "source_sha256", "created_at", "expires_at"}
        if set(sidecar.keys()) != expected_keys:
            return False
        if sidecar.get("version") != REFERENCE_SIDECAR_VERSION:
            return False
        if not _valid_sha256(sidecar.get("sha256")) or sidecar["sha256"] != digest:
            return False
        if not _valid_sha256(sidecar.get("source_sha256")):
            return False
        created = _parse_utc(sidecar.get("created_at"))
        expires = _parse_utc(sidecar.get("expires_at"))
        if created is None or expires is None or expires - created != REFERENCE_TTL:
            return False
        now = datetime.now(UTC)
        return created <= now < expires

    def get_reference(self) -> Path | None:
        """Đường dẫn reference còn hạn và còn nguyên hash, hoặc None.

        Reference là bản copy do app sở hữu. Khi sidecar/hash/lifetime không hợp
        lệ, store xoá copy vô hiệu để tránh vô tình dùng mốc cũ hoặc đã bị sửa.
        """
        try:
            p = self._inside_root(REFERENCE_IMAGE)
        except ValueError:
            return None
        if not p.exists() and not p.is_symlink():
            return None
        if self._valid_stored_reference(p):
            return p
        try:
            self.clear()
        except OSError:
            pass
        return None

    def sidecar(self) -> dict | None:
        """Đọc ``reference.json`` có giới hạn kích thước — None nếu thiếu/hỏng."""
        try:
            p = self._inside_root(REFERENCE_SIDECAR)
            if p.is_symlink() or not p.is_file() or p.stat().st_size > _SIDECAR_MAX_BYTES:
                return None
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def clear(self) -> None:
        """Xoá reference.png + reference.json; no-op khi chưa tồn tại."""
        for name in (REFERENCE_IMAGE, REFERENCE_SIDECAR):
            self._inside_root(name).unlink(missing_ok=True)
