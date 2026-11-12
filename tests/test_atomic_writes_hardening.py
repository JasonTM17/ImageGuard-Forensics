"""Hardening tests for atomic exclusive sibling and artifact writes."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.enhancement import derive as derive_mod
from imageguard.enhancement.derive import write_sibling
from imageguard.storage.artifacts import ArtifactStore

_UNSHARP = {"ksize": 5, "sigma": 1.0, "amount": 0.7}


def _source_png(tmp_path: Path) -> tuple[Path, bytes, str]:
    image = np.full((24, 26, 3), 90, dtype=np.uint8)
    source = tmp_path / "goc.png"
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    payload = encoded.tobytes()
    source.write_bytes(payload)
    parent_sha = hashlib.sha256(payload).hexdigest()
    return source, payload, parent_sha


def test_write_sibling_rejects_hardlink_destination_without_mutating_source(
    tmp_path: Path,
) -> None:
    source, original, parent_sha = _source_png(tmp_path)
    dest = tmp_path / "goc.view.png"
    os.link(source, dest)
    image = np.full((24, 26, 3), 120, dtype=np.uint8)

    with pytest.raises(ValueError, match="destination must differ"):
        write_sibling(source, dest, image, "unsharp", dict(_UNSHARP), parent_sha)

    assert source.read_bytes() == original


def test_write_sibling_publish_guard_blocks_late_hardlink_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, original, parent_sha = _source_png(tmp_path)
    dest = tmp_path / "goc.view.png"
    image = np.full((24, 26, 3), 120, dtype=np.uint8)
    calls = 0
    real_guard = derive_mod._guard_destination_for_publish

    def _inject_hardlink(path: Path, src: Path, *, overwrite: bool) -> None:
        nonlocal calls
        calls += 1
        if calls == 2 and path == dest and not dest.exists():
            os.link(source, dest)
        real_guard(path, src, overwrite=overwrite)

    monkeypatch.setattr(derive_mod, "_guard_destination_for_publish", _inject_hardlink)

    with pytest.raises(ValueError, match="destination must differ"):
        write_sibling(source, dest, image, "unsharp", dict(_UNSHARP), parent_sha)

    assert source.read_bytes() == original


def test_write_sibling_failure_leaves_no_stray_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, original, parent_sha = _source_png(tmp_path)
    dest = tmp_path / "goc.view.png"
    image = np.full((24, 26, 3), 120, dtype=np.uint8)

    def _boom(_src: object, _dst: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(derive_mod.os, "replace", _boom)
    with pytest.raises(OSError):
        write_sibling(source, dest, image, "unsharp", dict(_UNSHARP), parent_sha)

    assert source.read_bytes() == original
    temps = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert temps == []


def test_put_bytes_uses_unpredictable_temp_not_pid_pattern(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "art")
    store.put_bytes("aid", "report.json", b"one")
    aid_dir = tmp_path / "art" / "aid"
    temps = [p.name for p in aid_dir.iterdir() if p.name.endswith(".tmp")]
    assert temps == []
    assert all(f".{os.getpid()}.tmp" not in name for name in temps)


def test_put_bytes_publish_escape_via_symlink_parent_rejected(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "art")
    store.root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    aid_link = store.root / "aid"
    try:
        os.symlink(outside, aid_link, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are not available on this platform")

    with pytest.raises(ValueError, match="artifact root"):
        store.put_bytes("aid", "report.json", b"payload")


def test_put_bytes_failure_leaves_no_stray_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "art")
    store.put_bytes("aid", "report.json", b"original")

    def _boom(_src: object, _dst: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("imageguard.storage.artifacts.os.replace", _boom)
    with pytest.raises(OSError):
        store.put_bytes("aid", "report.json", b"replacement")

    assert store.read("aid/report.json") == b"original"
    aid_dir = tmp_path / "art" / "aid"
    temps = [p for p in aid_dir.iterdir() if p.name.endswith(".tmp")]
    assert temps == []
