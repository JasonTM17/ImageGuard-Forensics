"""Folder ingest optional ``enhance_sibling`` — derive view PNG next to source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.errors import ErrorCode, ErrorResult
from imageguard.enhancement.derive import write_sibling
from imageguard.ingest.config import IngestConfig
from imageguard.ingest.folder import FolderIngestor
from imageguard.services.analysis_service import AnalysisRequest, AnalysisService


def _jpeg_bytes(seed: int = 1) -> bytes:
    pixels = np.random.default_rng(seed).integers(
        0, 255, (64, 64, 3), dtype=np.uint8
    )
    ok, encoded = cv2.imencode(".jpg", pixels)
    assert ok
    return encoded.tobytes()


def _config(data_dir: Path, *, enhance_sibling: bool) -> IngestConfig:
    return IngestConfig(
        camera_label="test",
        max_pending=1,
        methods=("ela",),
        data_dir=data_dir,
        enhance_sibling=enhance_sibling,
    )


def test_default_no_view_sibling(tmp_path: Path) -> None:
    watch_dir = tmp_path / "incoming"
    watch_dir.mkdir()
    source = watch_dir / "photo.jpg"
    source.write_bytes(_jpeg_bytes())
    calls: list[Path] = []
    service = AnalysisService()

    def track(req: AnalysisRequest) -> object:
        calls.append(req.path)
        return service.__class__.analyze(service, req)

    service.analyze = track  # type: ignore[method-assign]
    ingestor = FolderIngestor(_config(tmp_path / "data", enhance_sibling=False), service, "hash", settle_delay=0)

    ingestor.scan_once(watch_dir)

    assert calls == [source]
    assert not (watch_dir / "photo.view.png").exists()
    assert not (watch_dir / "photo.view.json").exists()


def test_enhance_sibling_writes_sidecars_preserves_source(tmp_path: Path) -> None:
    watch_dir = tmp_path / "incoming"
    watch_dir.mkdir()
    source = watch_dir / "photo.jpg"
    payload = _jpeg_bytes(42)
    source.write_bytes(payload)
    before_sha = hashlib.sha256(payload).hexdigest()

    calls: list[Path] = []
    service = AnalysisService()

    def track(req: AnalysisRequest) -> object:
        calls.append(req.path)
        return service.__class__.analyze(service, req)

    service.analyze = track  # type: ignore[method-assign]
    ingestor = FolderIngestor(_config(tmp_path / "data", enhance_sibling=True), service, "hash", settle_delay=0)

    summary = ingestor.scan_once(watch_dir)

    assert summary.analyzed == 1
    assert calls == [source]
    assert source.read_bytes() == payload
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_sha

    view_png = watch_dir / "photo.view.png"
    view_json = watch_dir / "photo.view.json"
    assert view_png.is_file()
    assert view_json.is_file()

    record = json.loads(view_json.read_text(encoding="utf-8"))
    assert record["derived"] is True
    assert record["model_sha256"] is None
    assert record["parent_sha256"] == before_sha
    assert "parent_name" not in record
    assert str(watch_dir) not in view_json.read_text(encoding="utf-8")


def test_sibling_reread_failure_uses_fixed_custody_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watch_dir = tmp_path / "incoming"
    watch_dir.mkdir()
    source = watch_dir / "photo.jpg"
    payload = _jpeg_bytes(7)
    source.write_bytes(payload)

    def unreadable(*_args: object, **_kwargs: object) -> ErrorResult:
        return ErrorResult(ErrorCode.FILE_UNREADABLE, r"D:\secret\photo.jpg")

    monkeypatch.setattr("imageguard.ingest.folder.safe_open", unreadable)
    data_dir = tmp_path / "data"
    ingestor = FolderIngestor(
        _config(data_dir, enhance_sibling=True), AnalysisService(), "hash", settle_delay=0
    )
    summary = ingestor.scan_once(watch_dir)
    log = (data_dir / "ingest" / "custody.jsonl").read_text(encoding="utf-8")

    assert summary.analyzed == 1
    assert "enhance_sibling_failed" in log
    assert "FILE_UNREADABLE" not in log
    assert "secret" not in log
    assert not (watch_dir / "photo.view.png").exists()
    assert source.read_bytes() == payload


def test_write_sibling_still_refuses_destination_same_as_source(
    tmp_path: Path,
) -> None:
    image = np.full((24, 26, 3), 90, dtype=np.uint8)
    source = tmp_path / "goc.png"
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    source.write_bytes(encoded.tobytes())
    parent_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="destination must differ"):
        write_sibling(
            source, source, image, "unsharp",
            {"ksize": 5, "sigma": 1.0, "amount": 0.7},
            parent_sha,
        )
