"""Contract tests cho SessionReferenceStore — ảnh tham chiếu cùng phiên.

Store giữ đúng 2 file (reference.png + reference.json{sha256,basename}) dưới
root; mọi đường dẫn resolve phải nằm trong root (chặn traversal). Dùng
tmp_path — không đụng data/ thật, không mở camera.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.paths import data_root
from imageguard.ingest.session_reference import SessionReferenceStore


def _png_bytes(seed: int = 0) -> bytes:
    img = np.random.default_rng(seed).integers(
        0, 255, (64, 64, 3), dtype=np.uint8
    )
    return cv2.imencode(".png", img)[1].tobytes()


def test_set_get_clear_roundtrip(tmp_path: Path) -> None:
    store = SessionReferenceStore(root=tmp_path / "session_ref")
    src = tmp_path / "capture_01.png"
    payload = _png_bytes(seed=7)
    src.write_bytes(payload)

    dst = store.set_reference(src)
    assert dst == store.root.resolve() / "reference.png"
    assert dst.is_file()
    assert dst.read_bytes() == payload  # copy nguyên byte, không decode/re-encode
    assert (store.root / "reference.json").is_file()
    assert store.get_reference() == dst

    store.clear()
    assert store.get_reference() is None
    assert not (store.root / "reference.png").exists()
    assert not (store.root / "reference.json").exists()


def test_sidecar_only_sha256_and_basename(tmp_path: Path) -> None:
    store = SessionReferenceStore(root=tmp_path / "ref")
    src = tmp_path / "khung mặt mốc.png"  # unicode basename
    payload = _png_bytes(seed=3)
    src.write_bytes(payload)

    store.set_reference(src)
    sidecar_path = store.root / "reference.json"
    raw = sidecar_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    assert set(data.keys()) == {"sha256", "basename"}
    assert data["sha256"] == hashlib.sha256(payload).hexdigest()
    assert data["basename"] == src.name
    # Không đường dẫn tuyệt đối / parent dir nào rò rỉ vào sidecar.
    assert str(tmp_path) not in raw
    assert str(src.parent) not in raw


def test_get_reference_none_when_empty(tmp_path: Path) -> None:
    store = SessionReferenceStore(root=tmp_path / "empty_ref")
    assert store.get_reference() is None


def test_set_reference_replaces_previous(tmp_path: Path) -> None:
    store = SessionReferenceStore(root=tmp_path / "ref")
    src_a = tmp_path / "a.png"
    src_b = tmp_path / "b.png"
    bytes_a, bytes_b = _png_bytes(seed=1), _png_bytes(seed=2)
    src_a.write_bytes(bytes_a)
    src_b.write_bytes(bytes_b)

    store.set_reference(src_a)
    store.set_reference(src_b)
    assert store.get_reference().read_bytes() == bytes_b
    data = json.loads((store.root / "reference.json").read_text(encoding="utf-8"))
    assert data["sha256"] == hashlib.sha256(bytes_b).hexdigest()
    assert data["basename"] == "b.png"


def test_clear_is_noop_when_nothing_stored(tmp_path: Path) -> None:
    store = SessionReferenceStore(root=tmp_path / "ref")
    store.clear()  # không raise khi file chưa tồn tại
    assert store.get_reference() is None


def test_default_root_is_data_session_ref() -> None:
    # Chỉ kiểm thuộc tính root — KHÔNG tạo file trong data/ thật.
    assert SessionReferenceStore().root == data_root() / "session_ref"


def test_rejects_paths_escaping_root(tmp_path: Path) -> None:
    store = SessionReferenceStore(root=tmp_path / "ref")
    with pytest.raises(ValueError):
        store._inside_root("../escape.png")
    with pytest.raises(ValueError):
        store._inside_root("nested/../../escape.png")
    # Tên hợp lệ nằm trong root được chấp nhận.
    ok = store._inside_root("reference.png")
    assert ok.parent == store.root.resolve()
