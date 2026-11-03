"""Artifact store — file outputs theo analysis_id, DB chỉ lưu relative path.

Layout: <root>/<analysis_id>/<kind>.<ext>  (kind: mask|heatmap|ela|pairs|report)
Guard: filename sanitize — cấm `..`/path separator để chặn traversal.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ROOT = Path("data/artifacts")

_SAFE = re.compile(r"^[a-z0-9_\-]+$", re.I)


def _check_name(name: str) -> str:
    if not _SAFE.match(name):
        raise ValueError(f"artifact name không hợp lệ: {name!r}")
    return name


class ArtifactStore:
    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def rel_path(self, analysis_id: str, filename: str) -> str:
        _check_name(analysis_id)
        _check_name(filename.rsplit(".", 1)[0])
        return f"{analysis_id}/{filename}"

    def put_bytes(self, analysis_id: str, filename: str, data: bytes) -> str:
        rel = self.rel_path(analysis_id, filename)
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        return rel

    def put_image(self, analysis_id: str, filename: str, img: np.ndarray) -> str:
        ok, buf = cv2.imencode(Path(filename).suffix or ".png", img)
        if not ok:
            raise ValueError(f"không encode được {filename}")
        return self.put_bytes(analysis_id, filename, buf.tobytes())

    def abs_path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"rel_path thoát khỏi artifact root: {rel!r}")
        return p

    def read(self, rel: str) -> bytes:
        return self.abs_path(rel).read_bytes()

    def delete_analysis(self, analysis_id: str) -> None:
        d = self.root / analysis_id
        if d.is_dir():
            shutil.rmtree(d)
