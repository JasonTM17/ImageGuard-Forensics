"""Qt-free enhancement engine for the Enhance tab.

Thin boundary over the evaluation restoration operators: the UI compute path
never imports PySide6, AnalysisService, or a database. Derived images are
written next to the source with a ``.json`` provenance sibling.
"""

from __future__ import annotations

import hashlib
import json
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

    ok, encoded = cv2.imencode(".png", np.asarray(image))
    if not ok:
        raise ValueError("OpenCV could not encode the image as PNG")
    payload = encoded.tobytes()
    if _same_as_source(dest, source_path) or _same_as_source(json_path, source_path):
        raise ValueError("destination must differ from the source image")
    if dest.exists():
        if not overwrite:
            raise ValueError(f"destination already exists: {dest.name}")
        dest.unlink()
    if json_path.exists():
        if not overwrite:
            raise ValueError(f"destination already exists: {json_path.name}")
        json_path.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)

    record = {
        "derived": True,
        "parent_sha256": parent_sha256,
        "derived_sha256": hashlib.sha256(payload).hexdigest(),
        "method": method,
        "params": _json_safe(params),
        "model_sha256": model_sha256,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    json_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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
