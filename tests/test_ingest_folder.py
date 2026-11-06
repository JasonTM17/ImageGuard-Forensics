"""Folder-ingest behavior, custody records, and CLI boundary tests."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from imageguard.core.errors import ErrorCode, ErrorResult
from imageguard.ingest.config import IngestConfig
from imageguard.ingest.folder import FolderIngestor
from imageguard.storage.models import SCHEMA_VERSION


def _jpeg_bytes(seed: int = 1) -> bytes:
    pixels = np.random.default_rng(seed).integers(
        0, 255, (64, 64, 3), dtype=np.uint8
    )
    ok, encoded = cv2.imencode(".jpg", pixels)
    assert ok
    return encoded.tobytes()


class _RecordingService:
    def __init__(self, result_factory=None) -> None:
        self.calls: list[Path] = []
        self.result_factory = result_factory

    def analyze(self, request):
        self.calls.append(request.path)
        if self.result_factory is not None:
            return self.result_factory(request)
        return SimpleNamespace(report_id=f"analysis-{len(self.calls)}")


def _config(data_dir: Path, *, max_pending: int = 1) -> IngestConfig:
    return IngestConfig(
        camera_label="door camera",
        max_pending=max_pending,
        methods=("ela",),
        data_dir=data_dir,
        consent_note="operator supplied note",
    )


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_tmp_file_waits_for_rename_to_stable_jpeg(tmp_path: Path) -> None:
    watch_dir = tmp_path / "incoming"
    watch_dir.mkdir()
    partial = watch_dir / "sample.jpg.tmp"
    partial.write_bytes(_jpeg_bytes())
    service = _RecordingService()
    ingestor = FolderIngestor(_config(tmp_path / "data"), service, "hash", settle_delay=0)

    first = ingestor.scan_once(watch_dir)
    assert first.analyzed == 0
    assert service.calls == []
    assert not ingestor.config.custody_path.exists()

    ready = watch_dir / "sample.jpg"
    partial.rename(ready)
    second = ingestor.scan_once(watch_dir)
    assert second.analyzed == 1
    assert service.calls == [ready]


def test_repeated_sha_is_submitted_once_per_session(tmp_path: Path) -> None:
    watch_dir = tmp_path / "incoming"
    watch_dir.mkdir()
    contents = _jpeg_bytes()
    (watch_dir / "a.jpg").write_bytes(contents)
    (watch_dir / "b.jpg").write_bytes(contents)
    service = _RecordingService()
    ingestor = FolderIngestor(
        _config(tmp_path / "data", max_pending=2), service, "hash", settle_delay=0
    )

    summary = ingestor.scan_once(watch_dir)

    assert len(service.calls) == 1
    assert summary.analyzed == 1
    assert summary.skipped_duplicate == 1
    assert [event["event"] for event in _read_events(ingestor.config.custody_path)] == [
        "analyzed", "skipped_duplicate"
    ]


def test_deferred_file_stays_on_disk_and_is_picked_up_later(tmp_path: Path) -> None:
    watch_dir = tmp_path / "incoming"
    watch_dir.mkdir()
    first_path = watch_dir / "a.jpg"
    deferred_path = watch_dir / "b.jpg"
    first_path.write_bytes(_jpeg_bytes(1))
    deferred_path.write_bytes(_jpeg_bytes(2))
    config = _config(tmp_path / "data", max_pending=1)
    first_service = _RecordingService()
    first_ingestor = FolderIngestor(config, first_service, "hash", settle_delay=0)

    first_summary = first_ingestor.scan_once(watch_dir)
    assert first_summary.analyzed == 1
    assert first_summary.deferred == 1
    assert deferred_path.is_file()

    later_service = _RecordingService()
    later_ingestor = FolderIngestor(config, later_service, "hash", settle_delay=0)
    later_summary = later_ingestor.scan_once(watch_dir)
    assert later_summary.analyzed == 1
    assert later_service.calls == [deferred_path]
    assert [event["event"] for event in _read_events(config.custody_path)] == [
        "analyzed", "deferred", "analyzed"
    ]


def test_custody_fields_are_trust_labeled_and_store_only_basename(
    tmp_path: Path,
) -> None:
    watch_dir = tmp_path / "incoming"
    watch_dir.mkdir()
    image = watch_dir / "capture.jpg"
    image.write_bytes(_jpeg_bytes())
    config = _config(tmp_path / "data")
    ingestor = FolderIngestor(
        config, _RecordingService(), "threshold-snapshot", settle_delay=0
    )

    ingestor.scan_once(watch_dir)

    record = _read_events(config.custody_path)[0]
    assert record["record_version"] == 1
    assert record["basename"] == image.name
    assert record["camera_label"] == {
        "value": "door camera", "trust": "operator_supplied"
    }
    assert record["file_mtime"]["trust"] == "filesystem_unverified"
    assert record["analysis_id"] == "analysis-1"
    assert record["config_hash"] == "threshold-snapshot"
    assert record["consent_note"] == "operator supplied note"
    assert len(record["image_sha256"]) == 64
    serialized = json.dumps(record, ensure_ascii=False)
    assert re.search(r"\b[A-Za-z]:[\\/]", serialized) is None
    assert str(tmp_path) not in serialized


def test_error_result_is_logged_and_does_not_abort_scan(tmp_path: Path) -> None:
    watch_dir = tmp_path / "incoming"
    watch_dir.mkdir()
    image = watch_dir / "bad.jpg"
    image.write_bytes(_jpeg_bytes())
    service = _RecordingService(
        lambda _request: ErrorResult(ErrorCode.DECODE_FAILED, "decode failed")
    )
    ingestor = FolderIngestor(_config(tmp_path / "data"), service, "hash", settle_delay=0)

    summary = ingestor.scan_once(watch_dir)

    assert summary.errors == 1
    record = _read_events(ingestor.config.custody_path)[0]
    assert record["event"] == "error"
    assert record["error_code"] == "DECODE_FAILED"
    assert record["analysis_id"] is None


def test_ingest_does_not_enter_core_or_forensics_import_graph() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "imageguard"
    for package in ("core", "forensics"):
        for path in (source_root / package).rglob("*.py"):
            assert "imageguard.ingest" not in path.read_text(encoding="utf-8")


def test_storage_schema_version_is_unchanged() -> None:
    assert SCHEMA_VERSION == 1


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    source_dir = Path(__file__).resolve().parents[1] / "src"
    prior_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(source_dir)
        if not prior_pythonpath
        else os.pathsep.join((str(source_dir), prior_pythonpath))
    )
    return subprocess.run(
        [sys.executable, "-m", "imageguard", "ingest", *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_cli_once_scans_and_returns_zero_when_one_image_errors(tmp_path: Path) -> None:
    watch_dir = tmp_path / "incoming"
    watch_dir.mkdir()
    (watch_dir / "broken.jpg").write_bytes(b"\xff\xd8\xffbroken")
    (watch_dir / "good.jpg").write_bytes(_jpeg_bytes())
    config_path = tmp_path / "ingest.yaml"
    config_path.write_text(
        "camera_label: test camera\nmax_pending: 2\nmethods: [ela]\n"
        "data_dir: data\n",
        encoding="utf-8",
    )

    result = _run_cli(
        tmp_path, "once", str(watch_dir), "--config", str(config_path)
    )

    assert result.returncode == 0, result.stderr
    assert "analyzed=1" in result.stdout
    assert "errors=1" in result.stdout
    assert "Traceback" not in result.stderr
    data_dir = tmp_path / "data"
    assert (data_dir / "ingest" / "custody.jsonl").is_file()
    with sqlite3.connect(data_dir / "imageguard.db") as connection:
        analysis_count = connection.execute(
            "SELECT count(*) FROM analyses"
        ).fetchone()[0]
    assert analysis_count == 1
    assert list((data_dir / "artifacts").glob("*/*.json"))


@pytest.mark.parametrize(
    ("directory", "config_text"),
    [
        ("incoming", "max_pending: 0\n"),
        ("missing", "methods: [ela]\ndata_dir: data\n"),
    ],
)
def test_cli_returns_nonzero_for_bad_config_or_missing_directory(
    tmp_path: Path, directory: str, config_text: str
) -> None:
    (tmp_path / "incoming").mkdir()
    config_path = tmp_path / "ingest.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    result = _run_cli(
        tmp_path, "once", directory, "--config", str(config_path)
    )

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
