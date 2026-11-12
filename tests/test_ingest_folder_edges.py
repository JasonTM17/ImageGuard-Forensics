"""Edge cases for folder ingest, its config, and the custody log.

Covers empty and nested folders, unicode/spaced names, duplicate content,
non-image and zero-byte files, corrupt JPEGs, directory links, ``max_pending``
boundaries, config validation, append-only custody records, basename-only
logging, and deterministic ordering.

Every artifact lives under ``tmp_path``; image data comes from a seeded RNG;
no network, real database, or analysis service is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from imageguard.core.errors import ErrorCode, ErrorResult
from imageguard.ingest.config import (
    IngestConfig,
    IngestConfigError,
    load_ingest_config,
)
from imageguard.ingest.folder import FolderIngestor


def _encoded_image(seed: int) -> bytes:
    """Return deterministic JPEG bytes for a seeded 64x64 RGB image."""
    pixels = np.random.default_rng(seed).integers(
        0, 255, (64, 64, 3), dtype=np.uint8
    )
    ok, encoded = cv2.imencode(".jpg", pixels)
    assert ok
    return encoded.tobytes()


class _RecordingService:
    """AnalysisService stand-in that records submitted paths."""

    def __init__(self, result_factory=None) -> None:
        self.calls: list[Path] = []
        self._result_factory = result_factory

    def analyze(self, request):
        self.calls.append(request.path)
        if self._result_factory is not None:
            return self._result_factory(request)
        content_sha = hashlib.sha256(request.path.read_bytes()).hexdigest()
        return SimpleNamespace(
            report_id=f"analysis-{len(self.calls)}",
            image_sha256=content_sha,
        )


def _config(data_dir: Path, *, max_pending: int = 1) -> IngestConfig:
    return IngestConfig(
        camera_label="edge camera",
        max_pending=max_pending,
        methods=("ela",),
        data_dir=data_dir,
    )


def _ingestor(config: IngestConfig, service: _RecordingService) -> FolderIngestor:
    return FolderIngestor(config, service, "config-hash", settle_delay=0)


def _events(custody_path: Path) -> list[dict]:
    text = custody_path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _link_directory(link: Path, target: Path) -> bool:
    """Create a directory symlink or junction; return False when unsupported."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        pass
    if os.name != "nt":
        return False
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return completed.returncode == 0 and link.is_dir()


def _remove_directory_link(link: Path) -> None:
    """Remove a directory symlink or junction without touching its target."""
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        link.rmdir()


def test_empty_folder_reports_no_activity_and_writes_no_log(
    tmp_path: Path,
) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    config = _config(tmp_path / "data")
    service = _RecordingService()

    summary = _ingestor(config, service).scan_once(watch_dir)

    assert (
        summary.analyzed,
        summary.errors,
        summary.skipped_duplicate,
        summary.deferred,
    ) == (0, 0, 0, 0)
    assert summary.has_activity is False
    assert service.calls == []
    assert not config.custody_path.exists()


def test_nested_folder_contents_are_not_scanned(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    nested = watch_dir / "nested"
    nested.mkdir()
    inner = nested / "inner.jpg"
    inner.write_bytes(_encoded_image(1))
    top = watch_dir / "top.jpg"
    top.write_bytes(_encoded_image(2))
    config = _config(tmp_path / "data")
    service = _RecordingService()

    summary = _ingestor(config, service).scan_once(watch_dir)

    assert summary.analyzed == 1
    assert service.calls == [top]
    assert [r["basename"] for r in _events(config.custody_path)] == ["top.jpg"]
    assert inner.is_file()


def test_unicode_and_spaced_names_are_analyzed_once(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    names = ["ảnh cổng chính.jpg", "capture 01 - front (west).jpg"]
    for index, name in enumerate(names):
        (watch_dir / name).write_bytes(_encoded_image(index + 20))
    config = _config(tmp_path / "data", max_pending=2)
    service = _RecordingService()
    ingestor = _ingestor(config, service)

    summary = ingestor.scan_once(watch_dir)

    assert summary.analyzed == 2
    assert {path.name for path in service.calls} == set(names)
    assert sorted(r["basename"] for r in _events(config.custody_path)) == sorted(
        names
    )

    repeated = ingestor.scan_once(watch_dir)

    assert repeated.has_activity is False
    assert len(service.calls) == 2


def test_duplicate_content_under_two_names_is_skipped(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    payload = _encoded_image(7)
    first = watch_dir / "a.jpg"
    second = watch_dir / "b.jpg"
    first.write_bytes(payload)
    second.write_bytes(payload)
    config = _config(tmp_path / "data", max_pending=2)
    service = _RecordingService()

    summary = _ingestor(config, service).scan_once(watch_dir)

    assert (summary.analyzed, summary.skipped_duplicate) == (1, 1)
    assert service.calls == [first]
    records = _events(config.custody_path)
    assert [r["event"] for r in records] == ["analyzed", "skipped_duplicate"]
    assert records[1]["basename"] == "b.jpg"
    assert records[1]["analysis_id"] is None


def test_non_image_and_zero_byte_files_are_ignored(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    (watch_dir / "notes.txt").write_text("not an image", encoding="utf-8")
    (watch_dir / "empty.jpg").write_bytes(b"")
    (watch_dir / "fake.png").write_bytes(b"not a png at all")
    (watch_dir / "folder.jpg").mkdir()
    config = _config(tmp_path / "data")
    service = _RecordingService()

    summary = _ingestor(config, service).scan_once(watch_dir)

    assert summary.has_activity is False
    assert service.calls == []
    assert not config.custody_path.exists()


def test_corrupt_jpeg_is_logged_as_decode_error_and_not_retried(
    tmp_path: Path,
) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    (watch_dir / "broken.jpg").write_bytes(b"\xff\xd8\xffbroken")
    (watch_dir / "good.jpg").write_bytes(_encoded_image(4))

    def result_for(request):
        if request.path.name == "broken.jpg":
            return ErrorResult(ErrorCode.DECODE_FAILED, "decode failed")
        content_sha = hashlib.sha256(request.path.read_bytes()).hexdigest()
        return SimpleNamespace(report_id="analysis-ok", image_sha256=content_sha)

    config = _config(tmp_path / "data", max_pending=2)
    service = _RecordingService(result_for)
    ingestor = _ingestor(config, service)

    summary = ingestor.scan_once(watch_dir)

    assert (summary.analyzed, summary.errors) == (1, 1)
    records = _events(config.custody_path)
    assert [r["event"] for r in records] == ["error", "analyzed"]
    assert records[0]["basename"] == "broken.jpg"
    assert records[0]["error_code"] == "DECODE_FAILED"
    assert records[0]["analysis_id"] is None

    assert ingestor.scan_once(watch_dir).has_activity is False


def test_directory_link_is_not_descended(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    inner = outside / "inside.jpg"
    inner.write_bytes(_encoded_image(5))
    link = watch_dir / "linked"
    if not _link_directory(link, outside):
        pytest.skip("directory links are not available on this platform")
    config = _config(tmp_path / "data")
    service = _RecordingService()
    try:
        summary = _ingestor(config, service).scan_once(watch_dir)
    finally:
        _remove_directory_link(link)

    assert summary.analyzed == 0
    assert service.calls == []
    assert inner.is_file()


def test_file_symlink_is_skipped_with_path_invalid_error(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    source = outside / "source.jpg"
    source.write_bytes(_encoded_image(6))
    link = watch_dir / "linked.jpg"
    try:
        os.symlink(source, link)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are not available on this platform")
    config = _config(tmp_path / "data")
    service = _RecordingService()
    ingestor = _ingestor(config, service)
    try:
        summary = ingestor.scan_once(watch_dir)
        assert (
            summary.analyzed,
            summary.errors,
            summary.skipped_duplicate,
            summary.deferred,
        ) == (0, 1, 0, 0)
        assert ingestor.scan_once(watch_dir).has_activity is False
    finally:
        link.unlink()

    assert service.calls == []
    records = _events(config.custody_path)
    assert len(records) == 1
    assert records[0]["basename"] == "linked.jpg"
    assert records[0]["event"] == "error"
    assert records[0]["error_code"] == "PATH_INVALID"


@pytest.mark.parametrize(
    ("file_count", "expected_analyzed", "expected_deferred"),
    [(1, 1, 0), (2, 2, 0), (3, 2, 1)],
    ids=["one-below", "exactly-at", "one-above"],
)
def test_max_pending_boundary(
    tmp_path: Path,
    file_count: int,
    expected_analyzed: int,
    expected_deferred: int,
) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    names = ["a.jpg", "b.jpg", "c.jpg"][:file_count]
    for index, name in enumerate(names):
        (watch_dir / name).write_bytes(_encoded_image(index + 30))
    config = _config(tmp_path / "data", max_pending=2)
    service = _RecordingService()

    summary = _ingestor(config, service).scan_once(watch_dir)

    assert summary.analyzed == expected_analyzed
    assert summary.deferred == expected_deferred
    assert [path.name for path in service.calls] == names[:expected_analyzed]
    for name in names[expected_analyzed:]:
        assert (watch_dir / name).is_file()
    expected_events = ["analyzed"] * expected_analyzed
    expected_events += ["deferred"] * expected_deferred
    assert [r["event"] for r in _events(config.custody_path)] == expected_events


def test_partial_config_uses_documented_defaults(tmp_path: Path) -> None:
    path = tmp_path / "ingest.yaml"
    path.write_text("camera_label: lab camera\n", encoding="utf-8")

    config = load_ingest_config(path)

    assert config.max_pending == 1
    assert config.methods == ("keypoint", "block", "ela")
    assert config.data_dir == Path("data")
    assert config.consent_note is None
    assert config.http is None
    assert config.enhance_sibling is False


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ('max_pending: "2"\n', "max_pending must be a positive integer"),
        ("max_pending: true\n", "max_pending must be a positive integer"),
        ("max_pending: -1\n", "max_pending must be a positive integer"),
        ("methods: ela\n", "methods must be a non-empty list"),
        ("data_dir: 7\n", "data_dir must be a relative path"),
    ],
    ids=["string", "boolean", "negative", "methods-not-list", "path-type"],
)
def test_config_rejects_wrong_type_or_negative_values(
    tmp_path: Path, body: str, match: str
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(IngestConfigError, match=match):
        load_ingest_config(path)


def test_custody_log_appends_without_rewriting_earlier_lines(
    tmp_path: Path,
) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    (watch_dir / "a.jpg").write_bytes(_encoded_image(8))
    config = _config(tmp_path / "data")
    ingestor = _ingestor(config, _RecordingService())

    ingestor.scan_once(watch_dir)
    first_text = config.custody_path.read_text(encoding="utf-8")
    first_records = _events(config.custody_path)

    (watch_dir / "b.jpg").write_bytes(_encoded_image(9))
    ingestor.scan_once(watch_dir)

    second_text = config.custody_path.read_text(encoding="utf-8")
    assert second_text.startswith(first_text)
    assert len(second_text.splitlines()) == 2
    assert _events(config.custody_path)[: len(first_records)] == first_records


def test_custody_records_stay_basename_only(tmp_path: Path) -> None:
    watch_dir = tmp_path / "evidence inlet" / "nested"
    watch_dir.mkdir(parents=True)
    image = watch_dir / "ảnh 01.jpg"
    image.write_bytes(_encoded_image(10))
    config = _config(tmp_path / "data")
    service = _RecordingService()

    _ingestor(config, service).scan_once(watch_dir)

    text = config.custody_path.read_text(encoding="utf-8")
    record = _events(config.custody_path)[0]
    assert record["basename"] == image.name
    assert str(tmp_path) not in text
    assert str(watch_dir) not in text
    assert re.search(r"\b[A-Za-z]:[\\/]", text) is None
    assert "/" not in record["basename"]
    assert "\\" not in record["basename"]


@pytest.mark.parametrize("write_order", [(0, 1, 2), (2, 1, 0)])
def test_analyzed_order_is_casefold_deterministic(
    tmp_path: Path, write_order: tuple[int, ...]
) -> None:
    names = ["zebra 9.jpg", "Alpha 1.jpg", "mango 5.jpg"]
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    for index in write_order:
        (watch_dir / names[index]).write_bytes(_encoded_image(index + 40))
    config = _config(tmp_path / "data", max_pending=3)
    service = _RecordingService()

    summary = _ingestor(config, service).scan_once(watch_dir)

    assert summary.analyzed == 3
    expected = ["Alpha 1.jpg", "mango 5.jpg", "zebra 9.jpg"]
    assert [path.name for path in service.calls] == expected
    assert [r["basename"] for r in _events(config.custody_path)] == expected
