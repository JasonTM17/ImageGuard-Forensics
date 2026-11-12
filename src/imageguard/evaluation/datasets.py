"""Dataset adapters — protocol + SynthAdapter.

Sample: {image_path, gt_mask_path(None=authentic), label, category, base_image_id}.
Split by base_image_id → chống leakage (biến thể cùng ảnh gốc cùng phía split).
Deterministic: split = hash(base_image_id) % 2 → dev|test, không random state.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from imageguard.core import limits

_MANIFEST_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_HASH_CHUNK = 65536


def _validate_manifest_id(stem: str) -> None:
    if not stem or not _MANIFEST_ID.fullmatch(stem):
        raise ValueError(
            f"manifest id invalid or unsafe: {stem!r} "
            "(allowed: letters, digits, _, -, .; no path segments)"
        )


def _path_under_root(root: Path, *parts: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root / Path(*parts)).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"path escapes dataset root: {candidate}")
    return root.joinpath(*parts)


def _digest_file_sha256(digest: hashlib._Hash, path: Path) -> None:
    """Append the same bytes as whole-file read: file\\0 + len + content."""
    digest.update(b"file\0")
    size = path.stat().st_size
    digest.update(int(size).to_bytes(8, "big"))
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)


@dataclass(frozen=True, slots=True)
class Sample:
    image_path: Path
    gt_mask_path: Path | None  # None = authentic
    label: str  # "forged" | "authentic"
    category: str  # ví dụ "translate/none", "rotate/jpeg80" — per-condition report
    base_image_id: str


class DatasetAdapter(Protocol):
    name: str

    def samples(self, split: str = "dev") -> Iterable[Sample]:
        ...

    def version(self) -> str:
        ...


def split_of(base_id: str) -> str:
    """Deterministic split: hash ổn định của base_image_id → dev hoặc test.
    ~50/50 theo nghịch lý sinh nhật — đủ cho dev calibration vs locked test."""
    h = int(hashlib.sha256(base_id.encode()).hexdigest()[:8], 16)
    return "dev" if h % 2 == 0 else "test"


class SynthAdapter:
    """Đọc manifest synthforge (meta.json + *_O/_F/_M.png)."""

    name = "synthforge"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        mf = self.root / "meta.json"
        self.entries: list[dict] = []
        if mf.exists():
            try:
                manifest_bytes = mf.stat().st_size
            except OSError as exc:
                raise ValueError(f"cannot stat manifest: {exc}") from exc
            if manifest_bytes > limits.MAX_FILE_BYTES:
                raise ValueError(
                    f"meta.json too large: {manifest_bytes} bytes "
                    f"(limit {limits.MAX_FILE_BYTES})"
                )
            self.entries = json.loads(mf.read_text(encoding="utf-8"))
            if not isinstance(self.entries, list):
                raise TypeError("meta.json must be a JSON array")
            for entry in self.entries:
                if not isinstance(entry, dict):
                    raise TypeError("each manifest entry must be an object")
                _validate_manifest_id(str(entry.get("id", "")))

    def version(self) -> str:
        mf = self.root / "meta.json"
        if not mf.exists():
            return "absent"

        named_files: dict[str, Path] = {mf.name: mf}
        for entry in self.entries:
            stem = str(entry["id"])
            _validate_manifest_id(stem)
            ext = ".jpg" if entry.get("format") == "jpeg" else ".png"
            original = _path_under_root(self.root, f"{stem}_O.png")
            if not original.is_file():
                original = _path_under_root(self.root, f"{stem}_O{ext}")
            for path in (
                original,
                _path_under_root(self.root, f"{stem}_F{ext}"),
                _path_under_root(self.root, f"{stem}_M.png"),
            ):
                named_files[path.name] = path

        digest = hashlib.sha256()
        for filename in sorted(named_files):
            path = named_files[filename]
            digest.update(filename.encode("utf-8"))
            digest.update(b"\0")
            if path.is_file():
                _digest_file_sha256(digest, path)
            else:
                digest.update(b"<missing>")
            digest.update(b"\0")
        return digest.hexdigest()[:12]

    def samples(self, split: str = "dev") -> Iterable[Sample]:
        for e in self.entries:
            stem = str(e["id"])
            _validate_manifest_id(stem)
            spec = e.get("spec") or {}
            # Hai schema manifest: synthforge {postproc, rotate_deg, ...} và
            # modelforge {kind, shape, rect, src/dst/foreign} — không KeyError.
            shape = spec.get("shape", spec.get("kind", "unknown"))
            cat = f"{shape}/{spec.get('postproc', 'none')}/rot{spec.get('rotate_deg', 0)}"
            base_id = f"{stem}-seed{e.get('seed', stem)}"
            if split_of(base_id) != split:
                continue
            ext = ".jpg" if e.get("format") == "jpeg" else ".png"
            f = _path_under_root(self.root, f"{stem}_F{ext}")
            m = _path_under_root(self.root, f"{stem}_M.png")
            o = _path_under_root(self.root, f"{stem}_O.png")
            if not o.exists():
                o = _path_under_root(self.root, f"{stem}_O{ext}")
            if f.exists() and m.exists():
                yield Sample(f, m, "forged", cat, base_id)
            if o.exists():
                # Ảnh gốc = authentic, gt mask rỗng — đo false-positive.
                yield Sample(o, None, "authentic", cat, base_id)
