"""Unit tests for `CustodyLog` append-only semantics and event de-duplication.

`latest_events` is last-wins per ``(basename, image_sha256)``; malformed or
partial JSON lines are skipped; a missing file yields ``{}``.
"""

from __future__ import annotations

import json
from pathlib import Path

from imageguard.ingest.custody import CustodyLog


def _append(
    log: CustodyLog,
    *,
    event: str = "analyzed",
    image_sha256: str = "a" * 64,
    basename: str = "img.jpg",
) -> None:
    log.append(
        event=event,
        image_sha256=image_sha256,
        basename=basename,
        camera_label="door camera",
        file_mtime="2024-01-01T00:00:00+00:00",
        analysis_id="analysis-1",
        config_hash="hash",
        tool_version="0.1.0",
    )


def test_append_only_preserves_history(tmp_path: Path) -> None:
    log = CustodyLog(tmp_path / "custody.jsonl")
    _append(log, event="deferred")
    _append(log, event="analyzed")

    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    events = [json.loads(line)["event"] for line in lines]
    assert events == ["deferred", "analyzed"]  # append giữ nguyên thứ tự


def test_latest_events_last_wins_per_key(tmp_path: Path) -> None:
    log = CustodyLog(tmp_path / "custody.jsonl")
    sha = "b" * 64
    _append(log, event="deferred", image_sha256=sha, basename="a.jpg")
    _append(log, event="analyzed", image_sha256=sha, basename="a.jpg")

    assert log.latest_events()[("a.jpg", sha)] == "analyzed"


def test_latest_events_keys_distinct_per_sha256(tmp_path: Path) -> None:
    log = CustodyLog(tmp_path / "custody.jsonl")
    _append(log, event="analyzed", image_sha256="c" * 64, basename="same.jpg")
    _append(log, event="error", image_sha256="d" * 64, basename="same.jpg")

    latest = log.latest_events()
    assert latest[("same.jpg", "c" * 64)] == "analyzed"
    assert latest[("same.jpg", "d" * 64)] == "error"
    assert len(latest) == 2


def test_corrupt_and_partial_lines_skipped(tmp_path: Path) -> None:
    p = tmp_path / "custody.jsonl"
    # dòng corrupt + dòng JSON hợp lệ nhưng không phải dict + dòng thiếu field
    p.write_text(
        "{not json}\n[1, 2, 3]\n{\"basename\": \"x.jpg\"}\n",
        encoding="utf-8",
    )
    log = CustodyLog(p)
    _append(log, event="analyzed", image_sha256="e" * 64, basename="ok.jpg")

    latest = log.latest_events()
    assert latest == {("ok.jpg", "e" * 64): "analyzed"}


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    log = CustodyLog(tmp_path / "does-not-exist.jsonl")
    assert log.latest_events() == {}
