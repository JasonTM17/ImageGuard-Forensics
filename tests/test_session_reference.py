"""Contract tests cho SessionReferenceStore — ảnh tham chiếu cùng phiên.

Store giữ đúng 2 file (reference.png + reference.json có hash/lifetime) dưới
root; mọi entry là basename trực tiếp trong root và symlink không được đi theo
victim bên ngoài. Dùng tmp_path — không đụng data/ thật, không mở camera.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.paths import data_root
from imageguard.ingest.session_reference import REFERENCE_TTL, SessionReferenceStore


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
    assert dst.read_bytes() == payload  # PNG cv2 deterministic; pixel luôn lossless
    assert (store.root / "reference.json").is_file()
    assert store.get_reference() == dst

    store.clear()
    assert store.get_reference() is None
    assert not (store.root / "reference.png").exists()
    assert not (store.root / "reference.json").exists()


def test_sidecar_hashes_without_source_basename(tmp_path: Path) -> None:
    store = SessionReferenceStore(root=tmp_path / "ref")
    src = tmp_path / "khung mặt mốc.png"  # unicode basename
    payload = _png_bytes(seed=3)
    src.write_bytes(payload)

    store.set_reference(src)
    sidecar_path = store.root / "reference.json"
    raw = sidecar_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    assert set(data.keys()) == {"version", "sha256", "source_sha256", "created_at", "expires_at"}
    assert data["version"] == 1
    stored = (store.root / "reference.png").read_bytes()
    assert data["sha256"] == hashlib.sha256(stored).hexdigest()
    assert data["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert src.name not in raw
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
    assert data["source_sha256"] == hashlib.sha256(bytes_b).hexdigest()
    assert "b.png" not in json.dumps(data, ensure_ascii=False)


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


def test_invalid_source_is_rejected_without_writing_reference(tmp_path: Path) -> None:
    store = SessionReferenceStore(root=tmp_path / "ref")
    src = tmp_path / "not-image.png"
    src.write_bytes(b"this is not an image")
    with pytest.raises(ValueError, match="không hợp lệ"):
        store.set_reference(src)
    assert store.get_reference() is None
    assert not (store.root / "reference.png").exists()
    assert not (store.root / "reference.json").exists()


def test_reference_is_normalized_png_with_same_pixels(tmp_path: Path) -> None:
    store = SessionReferenceStore(root=tmp_path / "ref")
    rng = np.random.default_rng(11)
    img = rng.integers(0, 255, (80, 96, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    src = tmp_path / "mốc nguồn.jpg"
    src.write_bytes(encoded.tobytes())

    dst = store.set_reference(src)
    stored_raw = dst.read_bytes()
    assert stored_raw.startswith(b"\x89PNG\r\n\x1a\n")
    original = cv2.imdecode(np.frombuffer(src.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    stored = cv2.imdecode(np.frombuffer(stored_raw, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(stored, original)
    data = json.loads((store.root / "reference.json").read_text(encoding="utf-8"))
    assert data["sha256"] != hashlib.sha256(src.read_bytes()).hexdigest()
    assert data["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert src.name not in json.dumps(data, ensure_ascii=False)


def test_hash_mismatch_invalidates_and_clears_internal_copy(tmp_path: Path) -> None:
    store = SessionReferenceStore(root=tmp_path / "ref")
    src = tmp_path / "a.png"
    src.write_bytes(_png_bytes(seed=1))
    store.set_reference(src)
    (store.root / "reference.png").write_bytes(_png_bytes(seed=2))

    assert store.get_reference() is None
    assert not (store.root / "reference.png").exists()
    assert not (store.root / "reference.json").exists()


def test_expired_reference_invalidates_and_clears_internal_copy(tmp_path: Path) -> None:
    store = SessionReferenceStore(root=tmp_path / "ref")
    src = tmp_path / "a.png"
    src.write_bytes(_png_bytes(seed=1))
    store.set_reference(src)
    sidecar = store.root / "reference.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    created = datetime.now(UTC) - timedelta(hours=9)
    data["created_at"] = created.isoformat()
    data["expires_at"] = (created + REFERENCE_TTL).isoformat()
    sidecar.write_text(json.dumps(data), encoding="utf-8")

    assert store.get_reference() is None
    assert not (store.root / "reference.png").exists()
    assert not (store.root / "reference.json").exists()


def test_set_reference_replaces_symlink_without_touching_victim(tmp_path: Path) -> None:
    store = SessionReferenceStore(root=tmp_path / "ref")
    store.root.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    link = store.root / "reference.png"
    try:
        os.symlink(victim, link)
    except OSError as exc:
        pytest.skip(f"symlink không khả dụng trên host này: {type(exc).__name__}")

    src = tmp_path / "a.png"
    src.write_bytes(_png_bytes(seed=1))
    store.set_reference(src)

    assert victim.read_text(encoding="utf-8") == "keep me"
    assert not link.is_symlink()
    assert store.get_reference() == link.resolve()


def test_clear_removes_symlink_entry_without_touching_victim(tmp_path: Path) -> None:
    store = SessionReferenceStore(root=tmp_path / "ref")
    store.root.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    link = store.root / "reference.png"
    try:
        os.symlink(victim, link)
    except OSError as exc:
        pytest.skip(f"symlink không khả dụng trên host này: {type(exc).__name__}")

    store.clear()
    assert victim.read_text(encoding="utf-8") == "keep me"
    assert not link.exists() and not link.is_symlink()


def test_normalized_reference_over_size_limit_fails_closed_before_write(tmp_path: Path, monkeypatch) -> None:
    import imageguard.ingest.session_reference as ref_mod

    store = SessionReferenceStore(root=tmp_path / "ref")
    src = tmp_path / "a.png"
    src.write_bytes(_png_bytes(seed=1))

    def oversized_png(_array, _ext):
        return True, np.zeros(ref_mod.limits.MAX_FILE_BYTES + 1, dtype=np.uint8)

    monkeypatch.setattr(ref_mod.cv2, "imencode", oversized_png)
    with pytest.raises(ValueError, match="chuẩn hoá vượt giới hạn"):
        store.set_reference(src)
    assert not (store.root / "reference.png").exists()
    assert not (store.root / "reference.json").exists()
