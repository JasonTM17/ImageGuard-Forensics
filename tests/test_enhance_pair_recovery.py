"""Controlled publication failures must preserve derived-output provenance."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.enhancement import derive as writer


def _png(value):
    return cv2.imencode(".png", np.full((48, 64, 3), value, np.uint8))[1].tobytes()


@pytest.fixture
def pair(tmp_path):
    source, dest = tmp_path / "source.png", tmp_path / "view.png"
    source.write_bytes(_png(70))
    return source, dest, hashlib.sha256(source.read_bytes()).hexdigest()


def _export(pair, *, value=180, overwrite=True, params=None):
    source, dest, parent = pair
    return writer.write_sibling(source, dest, np.full((48, 64, 3), value, np.uint8),
                                "unsharp", {} if params is None else params, parent,
                                overwrite=overwrite)


def _seed(pair, state):
    _, dest, _ = pair
    if state in {"png", "pair"}:
        dest.write_bytes(_png(10))
    if state in {"json", "pair"}:
        dest.with_suffix(".json").write_bytes(b'{"previous":"metadata"}')


def _files(folder):
    return {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}


@pytest.mark.parametrize("state", ["absent", "png", "json", "pair"])
def test_sidecar_failure_restores_pre_call_pair_bytes(pair, monkeypatch, state):
    _seed(pair, state)
    source, dest, _ = pair
    before = _files(dest.parent)
    real = writer._write_exclusive_atomic

    def fail_sidecar(path, *args, **kwargs):
        if path == dest.with_suffix(".json"):
            raise OSError("controlled sidecar failure")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(writer, "_write_exclusive_atomic", fail_sidecar)
    with pytest.raises(OSError, match="controlled sidecar failure"):
        _export(pair, overwrite=state != "absent")
    assert _files(dest.parent) == before
    assert source.read_bytes() == _png(70)


def test_first_publication_failure_preserves_pair_and_cleans_backup(pair, monkeypatch):
    _seed(pair, "pair")
    before = _files(pair[1].parent)

    def fail(*_args, **_kwargs):
        raise OSError("controlled PNG failure")

    monkeypatch.setattr(writer, "_write_exclusive_atomic", fail)
    with pytest.raises(OSError, match="controlled PNG failure"):
        _export(pair)
    assert _files(pair[1].parent) == before


def test_serialization_failure_happens_before_pair_publication(pair):
    _seed(pair, "pair")
    before = _files(pair[1].parent)
    with pytest.raises(ValueError):
        _export(pair, params={"invalid": float("nan")})
    assert _files(pair[1].parent) == before


def test_backup_sync_failure_aborts_before_publication(pair, monkeypatch):
    _seed(pair, "pair")
    before = _files(pair[1].parent)

    def fail_sync(_fd):
        raise OSError("controlled backup sync failure")

    monkeypatch.setattr(writer.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="controlled backup sync failure"):
        _export(pair)
    assert _files(pair[1].parent) == before


def test_atomic_sidecar_replace_failure_restores_pair_and_cleans_temps(pair, monkeypatch):
    _seed(pair, "pair")
    _, dest, _ = pair
    before = _files(dest.parent)
    real = writer.os.replace

    def fail_sidecar(source, target):
        if Path(target) == dest.with_suffix(".json"):
            raise OSError("controlled sidecar replace failure")
        return real(source, target)

    monkeypatch.setattr(writer.os, "replace", fail_sidecar)
    with pytest.raises(OSError, match="controlled sidecar replace failure"):
        _export(pair)
    assert _files(dest.parent) == before


def test_successful_overwrite_cleans_backup_and_matches_hash(pair):
    _seed(pair, "pair")
    _, dest, _ = pair
    sidecar = _export(pair)
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == json.loads(
        sidecar.read_text()
    )["derived_sha256"]
    assert set(_files(dest.parent)) == {"source.png", "view.png", "view.json"}


def test_rollback_failure_retains_previous_png_in_durable_backup(pair, monkeypatch):
    _seed(pair, "pair")
    _, dest, _ = pair
    previous = dest.read_bytes()
    real_write, real_replace = writer._write_exclusive_atomic, writer.os.replace

    def fail_sidecar(path, *args, **kwargs):
        if path == dest.with_suffix(".json"):
            raise OSError("controlled sidecar failure")
        return real_write(path, *args, **kwargs)

    def fail_restore(source, target):
        if Path(source).suffix == ".bak" and Path(target) == dest:
            raise OSError("controlled restore failure")
        return real_replace(source, target)

    monkeypatch.setattr(writer, "_write_exclusive_atomic", fail_sidecar)
    monkeypatch.setattr(writer.os, "replace", fail_restore)
    with pytest.raises(RuntimeError, match="recovery") as error:
        _export(pair)
    backups = list(dest.parent.glob("*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == previous
    assert backups[0].name in str(error.value)
    assert hashlib.sha256(previous).hexdigest() in str(error.value)
    assert str(dest.parent) not in str(error.value)
    assert error.value.__cause__ is not None


@pytest.mark.parametrize("changed", ["png", "json"])
def test_rollback_does_not_overwrite_external_replacement(pair, monkeypatch, changed):
    _seed(pair, "pair")
    _, dest, _ = pair
    previous_png = dest.read_bytes()
    sidecar = dest.with_suffix(".json")
    real = writer._write_exclusive_atomic

    def replace_outside_writer(path, *args, **kwargs):
        if path == sidecar:
            (dest if changed == "png" else sidecar).write_bytes(b"external writer")
            raise OSError("controlled sidecar failure")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(writer, "_write_exclusive_atomic", replace_outside_writer)
    with pytest.raises(RuntimeError, match="recovery"):
        _export(pair)
    assert (dest if changed == "png" else sidecar).read_bytes() == b"external writer"
    backups = list(dest.parent.glob("*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == previous_png


def test_cooperating_threads_cannot_publish_mismatched_pair(pair, monkeypatch):
    _, dest, _ = pair
    first_png_written, release_first, second_done = (threading.Event() for _ in range(3))
    errors = []
    real = writer._write_exclusive_atomic

    def pause_first_png(path, *args, **kwargs):
        result = real(path, *args, **kwargs)
        if path == dest and threading.current_thread().name == "first-export":
            first_png_written.set()
            assert release_first.wait(3)
        return result

    def run(value):
        try:
            _export(pair, value=value)
        except Exception as error:  # noqa: BLE001 - propagate thread failures to the assertion
            errors.append(error)
        finally:
            if value == 220:
                second_done.set()

    monkeypatch.setattr(writer, "_write_exclusive_atomic", pause_first_png)
    first = threading.Thread(target=run, args=(20,), name="first-export")
    second = threading.Thread(target=run, args=(220,), name="second-export")
    first.start()
    try:
        assert first_png_written.wait(2)
        second.start()
        # Without a pair lock the second export completes while the first is paused.
        second_done.wait(0.3)
    finally:
        release_first.set()
        first.join(3)
        if second.ident is not None:
            second.join(3)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    metadata = json.loads(dest.with_suffix(".json").read_text())
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == metadata["derived_sha256"]
