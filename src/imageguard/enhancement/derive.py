"""Qt-free enhancement engine for the Enhance tab.

Thin boundary over the evaluation restoration operators: the UI compute path
never imports PySide6, AnalysisService, or a database. Derived images are
written next to the source with a ``.json`` provenance sibling.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from imageguard.evaluation.enhance_ops import restore as restore_classical


def derive(image: np.ndarray, kind: str, params: dict) -> tuple[np.ndarray, dict]:
    """Apply a viewing operator and return (array, metadata)."""
    source = np.asarray(image).copy()
    if kind == "nafnet":
        from imageguard.enhancement.nafnet import infer

        return infer(source)
    return restore_classical(source, kind, params)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of file bytes (streamed reads)."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guard_destination_for_publish(
    path: Path, source: Path, *, overwrite: bool
) -> None:
    """Refuse publishing when the target is missing, taken, or the source file."""
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
    """Write bytes via an exclusive temp file and atomic replace."""
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
    """True when ``path`` is the source file, including after a junction swap."""
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
    """Write ``image`` as PNG and a ``.json`` provenance sibling.

    The source file is never written. An existing destination is removed only
    when ``overwrite`` is true, and only after the resolved PNG and sidecar
    are checked again against the source.
    """
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
    _write_exclusive_atomic(dest, payload, source_path, overwrite=overwrite)
    _write_exclusive_atomic(json_path, json_bytes, source_path, overwrite=overwrite)
    return json_path


def _json_safe(value: object) -> object:
    """Normalize numpy containers/scalars so ``json.dumps`` never fails."""
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
