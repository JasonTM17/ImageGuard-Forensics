"""Regression: custody fsync/single-write, commands finally, chunked JPEG EOI.

Fix 3 (ingest/custody.py): append ghi payload+"\\n" bằng MỘT write + flush +
os.fsync — crash giữa hai write cũ làm rách JSONL.

Fix 4 (ingest/commands.py): finally dùng getattr(session_factory, "kw", None)
— factory không phải sessionmaker không được ném AttributeError che mất return
code thật.

Fix 5 (ingest/http_pull.py): _stream_jpeg khi không có Content-Length
(remaining is None) phải từ chối body KHÔNG kết thúc bằng JPEG EOI \\xff\\xd9
bằng _AttemptFailure("connection_error") — không publish JPEG đọc cụt.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from imageguard.ingest import commands, custody, http_pull


def test_custody_append_fsync_and_single_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = custody.CustodyLog(tmp_path / "logs" / "custody.jsonl")
    fsync_calls: list[int] = []
    real_fsync = custody.os.fsync

    def fake_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(custody.os, "fsync", fake_fsync)
    log.append(
        event="ingested",
        image_sha256="ab" * 32,
        basename="shot.jpg",
        camera_label="cam-1",
        file_mtime="2026-11-24T00:00:00+00:00",
        analysis_id=None,
        config_hash="h",
        tool_version="0.1.0",
    )
    assert len(fsync_calls) == 1  # code cũ: không fsync → 0 → RED

    data = log.path.read_bytes()
    assert data.endswith(b"\n")
    lines = data.decode("utf-8").splitlines()
    assert len(lines) == 1  # đúng một dòng JSONL
    record = json.loads(lines[0])
    assert record["event"] == "ingested"
    assert record["basename"] == "shot.jpg"


def test_commands_dispose_tolerates_non_sessionmaker_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        commands, "load_ingest_config",
        lambda _p=None: SimpleNamespace(data_dir=tmp_path),
    )
    monkeypatch.setattr(
        commands, "load_thresholds",
        lambda: SimpleNamespace(snapshot_hash=lambda: "h"),
    )
    # Factory giả KHÔNG có .kw — finally cũ (session_factory.kw) ném AttributeError.
    factory = SimpleNamespace()
    monkeypatch.setattr(commands, "make_session_factory", lambda _p: factory)

    class _BoomIngestor:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def scan_once(self, _watch_dir: Path):
            raise RuntimeError("scan blew up")

    monkeypatch.setattr(commands, "FolderIngestor", _BoomIngestor)
    watch = tmp_path / "watch"
    watch.mkdir()
    rc = commands.run_ingest("once", watch)
    # finally phải nhường đường: return code thật (1) được giữ nguyên.
    assert rc == 1
    assert "ingest scan failed" in capsys.readouterr().err


class _StubResponse:
    """Response giả: headers + read() theo chunk + length (Content-Length)."""

    def __init__(self, chunks: list[bytes], length: int | None) -> None:
        self.headers = {"Content-Type": "image/jpeg"}
        self._chunks = list(chunks)
        self.length = length

    def read(self, n: int = -1) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


def test_http_pull_rejects_truncated_chunked_jpeg(tmp_path: Path) -> None:
    # SOI hợp lệ + body cụt, KHÔNG có EOI \xff\xd9; length=None (chunked, không
    # Content-Length) → phải từ chối bằng connection_error, không publish.
    body = b"\xff\xd8\xff" + b"\x11" * 128
    resp = _StubResponse([body[:64], body[64:]], length=None)
    with pytest.raises(http_pull._AttemptFailure) as excinfo:
        http_pull._stream_jpeg(
            resp, tmp_path, "shot.jpg", 10 * 1024 * 1024, ())
    assert excinfo.value.reason == "connection_error"
    assert excinfo.value.retryable is True
    assert list(tmp_path.iterdir()) == []  # không file nào được publish


def test_http_pull_accepts_chunked_jpeg_with_eoi(tmp_path: Path) -> None:
    # Chunked hợp lệ (length=None) nhưng kết thúc đúng EOI \xff\xd9 → publish.
    body = b"\xff\xd8\xff" + b"\x22" * 100 + b"\xff\xd9"
    resp = _StubResponse([body[:50], body[50:]], length=None)
    name = http_pull._stream_jpeg(
        resp, tmp_path, "ok.jpg", 10 * 1024 * 1024, ())
    assert name == "ok.jpg"
    assert (tmp_path / "ok.jpg").read_bytes() == body
