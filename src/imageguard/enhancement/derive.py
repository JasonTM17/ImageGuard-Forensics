"""Lõi xử lý Enhance không phụ thuộc Qt.

Gọi ``restore`` (và tùy chọn NAFNet) từ evaluation, ghi PNG + sidecar JSON
(gốc SHA-256, tham số, thời điểm). Tách lớp này để luồng tính toán không kéo
PySide6 hay cơ sở dữ liệu — phù hợp kiểm thử và mô tả pipeline khôi phục (C2/C3).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from imageguard.evaluation.enhance_ops import restore as restore_classical

_PAIR_LOCK = threading.RLock()


def derive(image: np.ndarray, kind: str, params: dict) -> tuple[np.ndarray, dict]:
    """Điểm vào Enhance: ``nafnet`` gọi mô hình học sâu, còn lại gọi ``restore`` cổ điển."""
    source = np.asarray(image).copy()
    if kind == "nafnet":
        from imageguard.enhancement.nafnet import infer

        return infer(source)
    return restore_classical(source, kind, params)


def _sha256_file(path: Path) -> str:
    """Băm SHA-256 theo khối — xác minh file gốc khớp ``parent_sha256`` khi ghi sibling."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guard_destination_for_publish(
    path: Path, source: Path, *, overwrite: bool
) -> None:
    """Không ghi đè ảnh nguồn và không ghi chồng file đích khi ``overwrite=False``."""
    if _same_as_source(path, source):
        raise ValueError("destination must differ from the source image")
    if path.exists() and not overwrite:
        raise ValueError(f"destination already exists: {path.name}")


def _write_exclusive_atomic(
    dest: Path,
    data: bytes,
    source: Path,
    *,
    overwrite: bool,
) -> None:
    """Ghi qua file tạm ``O_EXCL`` rồi ``os.replace`` — tránh file dở khi lỗi giữa chừng."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _guard_destination_for_publish(dest, source, overwrite=overwrite)
    tmp_path: Path | None = None
    fd = -1
    try:
        for _ in range(64):
            candidate = dest.parent / f".{secrets.token_hex(16)}.tmp"
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
            raise OSError("could not allocate temporary file for destination")
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _guard_destination_for_publish(dest, source, overwrite=overwrite)
        os.replace(tmp_path, dest)
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


def _same_as_source(path: Path, source: Path) -> bool:
    """So sánh cùng inode/resolve để ảnh derive không thay thế nhầm file gốc."""
    if path.resolve() == source.resolve():
        return True
    if not path.exists():
        return False
    try:
        return path.samefile(source)
    except OSError:
        return True


def write_sibling(
    source: Path,
    dest_png: Path,
    image: np.ndarray,
    method: str,
    params: dict,
    parent_sha256: str,
    *,
    overwrite: bool = False,
    model_sha256: str | None = None,
) -> Path:
    """Ghi PNG kết quả khôi phục và sidecar JSON (method, params, hash, UTC).

    Không sửa file nguồn; ``overwrite`` phải bật mới thay file đích đã tồn tại.
    """
    with _PAIR_LOCK:
        return _write_sibling(
            source, dest_png, image, method, params, parent_sha256,
            overwrite=overwrite, model_sha256=model_sha256,
        )


def _write_sibling(
    source: Path,
    dest_png: Path,
    image: np.ndarray,
    method: str,
    params: dict,
    parent_sha256: str,
    *,
    overwrite: bool,
    model_sha256: str | None,
) -> Path:
    source_path = Path(source)
    dest = Path(dest_png)
    json_path = dest.with_suffix(".json")
    if json_path == dest:
        raise ValueError(
            "destination must not be a .json path: the provenance sidecar "
            "would overwrite the encoded PNG"
        )
    if _same_as_source(dest, source_path) or _same_as_source(json_path, source_path):
        raise ValueError("destination must differ from the source image")
    if dest.exists() or json_path.exists():
        if not overwrite:
            name = dest.name if dest.exists() else json_path.name
            raise ValueError(f"destination already exists: {name}")
        if _same_as_source(dest, source_path) or _same_as_source(
            json_path, source_path
        ):
            raise ValueError("destination must differ from the source image")

    computed_parent = _sha256_file(source_path)
    if parent_sha256 != computed_parent:
        raise ValueError("parent_sha256 does not match source file")

    ok, encoded = cv2.imencode(".png", np.asarray(image))
    if not ok:
        raise ValueError("OpenCV could not encode the image as PNG")
    payload = encoded.tobytes()
    if _same_as_source(dest, source_path) or _same_as_source(json_path, source_path):
        raise ValueError("destination must differ from the source image")
    record = {
        "derived": True,
        "parent_sha256": computed_parent,
        "derived_sha256": hashlib.sha256(payload).hexdigest(),
        "method": method,
        "params": _json_safe(params),
        "model_sha256": model_sha256,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    json_bytes = (
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _publish_pair(dest, payload, json_path, json_bytes, source_path, overwrite=overwrite)
    return json_path


def _optional_hash(path: Path) -> str | None:
    try:
        return _sha256_file(path)
    except FileNotFoundError:
        return None


def _backup_png(dest: Path) -> tuple[Path | None, str | None]:
    """Keep previous bytes on disk before either destination is published."""
    try:
        original = dest.open("rb")
    except FileNotFoundError:
        return None, None
    backup = None
    try:
        with original, tempfile.NamedTemporaryFile(
            mode="wb", prefix=".enhance-", suffix=".bak", dir=dest.parent, delete=False
        ) as stream:
            backup = Path(stream.name)
            digest = hashlib.sha256()
            for chunk in iter(lambda: original.read(1024 * 1024), b""):
                stream.write(chunk)
                digest.update(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        return backup, digest.hexdigest()
    except BaseException:
        original.close()
        if backup is not None:
            backup.unlink(missing_ok=True)
        raise


def _publish_pair(
    dest: Path, payload: bytes, sidecar: Path, metadata: bytes, source: Path,
    *, overwrite: bool,
) -> None:
    # The lock covers preflight as well as publication and recovery. It protects
    # cooperating threads only; separate processes must use distinct outputs.
    previous_sidecar = _optional_hash(sidecar)
    backup, previous_png = _backup_png(dest)
    png_published = False
    try:
        _write_exclusive_atomic(dest, payload, source, overwrite=overwrite)
        png_published = True
        _write_exclusive_atomic(sidecar, metadata, source, overwrite=overwrite)
    except BaseException as publication_error:
        if not png_published:
            if backup is not None:
                backup.unlink(missing_ok=True)
            raise
        try:
            current_png = _optional_hash(dest)
            if _optional_hash(sidecar) != previous_sidecar:
                raise RuntimeError("sidecar changed during publication")
            if current_png != previous_png:
                if current_png != hashlib.sha256(payload).hexdigest():
                    raise RuntimeError("PNG changed during publication")
                if _same_as_source(dest, source) or _same_as_source(sidecar, source):
                    raise RuntimeError("destination aliases the source")
                if backup is None:
                    dest.unlink()
                else:
                    if _sha256_file(backup) != previous_png:
                        raise RuntimeError("recovery backup changed")
                    os.replace(backup, dest)
                    backup = None
        except BaseException:  # noqa: BLE001 - keep recovery bytes even if interrupted
            recovery = (
                f"backup {backup.name}, SHA-256 {previous_png}"
                if backup is not None else "no previous PNG existed"
            )
            raise RuntimeError(
                f"incomplete derived export; recovery required for {dest.name}: {recovery}"
            ) from publication_error
        if backup is not None:
            backup.unlink(missing_ok=True)
        raise
    if backup is not None:
        backup.unlink(missing_ok=True)


def _json_safe(value: object) -> object:
    """Chuyển ndarray/scalar NumPy trong metadata thành kiểu JSON serializable."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
