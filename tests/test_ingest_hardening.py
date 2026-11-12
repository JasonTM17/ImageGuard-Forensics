"""Ingest hardening: size limits before hash, custody hash match, data_dir rules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from imageguard.core.errors import ErrorCode
from imageguard.ingest import folder as folder_mod
from imageguard.ingest.config import IngestConfig, IngestConfigError, load_ingest_config
from imageguard.ingest.folder import FolderIngestor, _sha256_file


def _encoded_image(seed: int) -> bytes:
    pixels = np.random.default_rng(seed).integers(
        0, 255, (64, 64, 3), dtype=np.uint8
    )
    ok, encoded = cv2.imencode(".jpg", pixels)
    assert ok
    return encoded.tobytes()


class _RecordingService:
    def __init__(self, report_sha: str | None = None) -> None:
        self.calls: list[Path] = []
        self._report_sha = report_sha

    def analyze(self, request):
        self.calls.append(request.path)
        scan_sha = hashlib.sha256(request.path.read_bytes()).hexdigest()
        sha = self._report_sha if self._report_sha is not None else scan_sha
        return SimpleNamespace(report_id="r1", image_sha256=sha)


def _config(data_dir: Path) -> IngestConfig:
    return IngestConfig(
        camera_label="hardening",
        max_pending=2,
        methods=("ela",),
        data_dir=data_dir,
    )


def _events(custody_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in custody_path.read_text(encoding="utf-8").splitlines()
    ]


def test_oversize_file_rejected_before_full_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("imageguard.ingest.folder.MAX_FILE_BYTES", 50)
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    huge = watch_dir / "huge.jpg"
    payload = _encoded_image(99)
    assert len(payload) > 50
    huge.write_bytes(payload)
    config = _config(tmp_path / "data")
    service = _RecordingService()
    ingestor = FolderIngestor(config, service, "hash", settle_delay=0)

    with patch.object(folder_mod, "_sha256_file", wraps=_sha256_file) as spy:
        summary = ingestor.scan_once(watch_dir)
        spy.assert_not_called()

    assert (summary.analyzed, summary.errors) == (0, 1)
    assert service.calls == []
    record = _events(config.custody_path)[0]
    assert record["event"] == "error"
    assert record["error_code"] == ErrorCode.FILE_TOO_LARGE.value

    with patch.object(folder_mod, "_sha256_file", wraps=_sha256_file) as spy:
        assert ingestor.scan_once(watch_dir).has_activity is False
        spy.assert_not_called()


def test_custody_hash_mismatch_logs_file_unreadable(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    image = watch_dir / "stable.jpg"
    image.write_bytes(_encoded_image(12))
    wrong_sha = "0" * 64
    config = _config(tmp_path / "data")
    service = _RecordingService(report_sha=wrong_sha)
    summary = FolderIngestor(
        config, service, "hash", settle_delay=0
    ).scan_once(watch_dir)

    assert (summary.analyzed, summary.errors) == (0, 1)
    record = _events(config.custody_path)[0]
    assert record["event"] == "error"
    assert record["error_code"] == ErrorCode.FILE_UNREADABLE.value
    assert record["analysis_id"] is None


@pytest.mark.parametrize(
    "data_dir_line",
    [
        "data_dir: '\\private'\n",
        "data_dir: D:relative\n",
        'data_dir: "/private"\n',
    ],
)
def test_config_rejects_root_relative_or_drive_relative_data_dir(
    tmp_path: Path, data_dir_line: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "bad.yaml"
    path.write_text("methods: [ela]\n" + data_dir_line, encoding="utf-8")

    with pytest.raises(IngestConfigError, match="must stay relative"):
        load_ingest_config(path)
