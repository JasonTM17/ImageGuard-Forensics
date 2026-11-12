"""Security hardening for evaluation masks and synth manifest loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core import limits
from imageguard.core.models import ImageDocument
from imageguard.evaluation.datasets import SynthAdapter
from imageguard.evaluation.runner import _load_mask
from imageguard.evaluation.synthforge import generate_set


def _reference_version_whole_file_read(root: Path) -> str:
    """Inline reference: pre-hardening whole-file version digest."""
    mf = root / "meta.json"
    if not mf.exists():
        return "absent"

    adapter = SynthAdapter.__new__(SynthAdapter)
    adapter.root = Path(root)
    adapter.entries = json.loads(mf.read_text(encoding="utf-8"))

    named_files: dict[str, Path] = {mf.name: mf}
    for entry in adapter.entries:
        stem = entry["id"]
        ext = ".jpg" if entry.get("format") == "jpeg" else ".png"
        original = root / f"{stem}_O.png"
        if not original.is_file():
            original = root / f"{stem}_O{ext}"
        for path in (
            original,
            root / f"{stem}_F{ext}",
            root / f"{stem}_M.png",
        ):
            named_files[path.name] = path

    digest = hashlib.sha256()
    for filename in sorted(named_files):
        path = named_files[filename]
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            content = path.read_bytes()
            digest.update(b"file\0")
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def test_synth_version_hash_unchanged_for_valid_generate_set(tmp_path: Path) -> None:
    d = tmp_path / "synth"
    generate_set(3, seed=19, out_dir=d, tier="easy")
    assert SynthAdapter(d).version() == _reference_version_whole_file_read(d)


def test_manifest_rejects_oversized_meta_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(limits, "MAX_FILE_BYTES", 32)
    (tmp_path / "meta.json").write_bytes(b"[" + b" " * 40 + b"]")
    with pytest.raises(ValueError, match="meta.json too large"):
        SynthAdapter(tmp_path)


def test_manifest_rejects_path_escape_id(tmp_path: Path) -> None:
    meta = [{"id": "../../escape", "seed": 1, "spec": {"kind": "erase"}}]
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest id invalid"):
        SynthAdapter(tmp_path)


def test_load_mask_rejects_oversized_file_before_decode(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(limits, "MAX_FILE_BYTES", 64)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask_path = tmp_path / "big_mask.png"
    mask_path.write_bytes(cv2.imencode(".png", mask)[1].tobytes() + b"\0" * 80)
    doc = ImageDocument(
        array=np.zeros((8, 8, 3), dtype=np.uint8),
        width=8,
        height=8,
        channels=3,
        image_format="png",
        sha256="x",
        file_bytes=1,
        file_name="image.png",
    )
    assert _load_mask(mask_path, doc) is None


def test_load_mask_rejects_header_over_pixel_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(limits, "MAX_PIXELS", 16)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask_path = tmp_path / "mask.png"
    mask_path.write_bytes(cv2.imencode(".png", mask)[1].tobytes())
    doc = ImageDocument(
        array=np.zeros((8, 8, 3), dtype=np.uint8),
        width=8,
        height=8,
        channels=3,
        image_format="png",
        sha256="x",
        file_bytes=1,
        file_name="image.png",
    )
    assert _load_mask(mask_path, doc) is None


def test_load_mask_dimension_mismatch_still_raises(tmp_path: Path) -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask_path = tmp_path / "mask.png"
    mask_path.write_bytes(cv2.imencode(".png", mask)[1].tobytes())
    doc = ImageDocument(
        array=np.zeros((8, 8, 3), dtype=np.uint8),
        width=8,
        height=8,
        channels=3,
        image_format="png",
        sha256="x",
        file_bytes=1,
        file_name="image.png",
    )
    with pytest.raises(ValueError, match="reject"):
        _load_mask(mask_path, doc)
