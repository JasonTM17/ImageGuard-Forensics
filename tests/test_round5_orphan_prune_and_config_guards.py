"""Vòng 5 — quyết định do user ủy quyền: orphan sweep, watch-mode prune,
config guards (ela qualities rỗng, DCT n_coeffs vượt B²).

Mỗi test khóa một quyết định thiết kế (xem journal 2026-11-24 vòng 4/5).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from imageguard import storage
from imageguard.core.config import ThresholdSet
from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.ela import ELA
from imageguard.forensics.block.dct_blocks import zigzag_indices
from imageguard.forensics.block.pipeline import BlockCopyMove
from imageguard.ingest.config import IngestConfig
from imageguard.ingest.folder import FolderIngestor
from imageguard.services.analysis_service import reclaim_orphan_artifacts
from imageguard.storage.artifacts import ArtifactStore
from imageguard.storage.db import make_session_factory
from imageguard.storage.repository import AnalysisRepository, ImageRepository


class _OverridingCfg:
    """Proxy cfg: override một (section, key); còn lại delegate ThresholdSet."""

    def __init__(self, base: ThresholdSet, section: str, key: str, value) -> None:
        self._base = base
        self._section = section
        self._key = key
        self._value = value

    def value(self, section: str, key: str):
        if (section, key) == (self._section, self._key):
            return self._value
        return self._base.value(section, key)

    def limit(self, key: str, fallback: int) -> int:
        return self._base.limit(key, fallback)

    def snapshot_hash(self) -> str:
        return self._base.snapshot_hash()


def _jpeg_doc(h: int = 32, w: int = 32, seed: int = 5) -> ImageDocument:
    array = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return ImageDocument(
        array=array,
        width=w,
        height=h,
        channels=3,
        image_format="jpeg",
        sha256="a" * 64,
        file_bytes=array.nbytes,
        file_name="photo.jpg",
    )


def _seed_analysis(tmp_path: Path, rid: str) -> None:
    sf = make_session_factory(tmp_path / "app.db")
    images = ImageRepository(sf)
    analyses = AnalysisRepository(sf)
    asset = images.get_or_create(
        f"sha-{rid[:8]}", "img.png", 4, 4, "png")
    analyses.save_report(rid, asset.id, {
        "tool_version": "t", "overall_verdict": "inconclusive",
        "limitations": [], "methods": []}, "hash", 1)
    sf.kw["bind"].dispose()


# ---------- Orphan sweep (startup reclaim) ----------

def test_sweep_reclaims_orphan_dirs_only(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    kept_rid = str(__import__("uuid").uuid4())
    orphan_rid = str(__import__("uuid").uuid4())
    (root / kept_rid).mkdir(parents=True)
    (root / kept_rid / "report.json").write_text("{}", encoding="utf-8")
    (root / orphan_rid).mkdir(parents=True)
    (root / "important-user-dir").mkdir()
    (root / "readme.txt").write_text("stay", encoding="utf-8")
    _seed_analysis(tmp_path, kept_rid)
    sf = make_session_factory(tmp_path / "app.db")
    store = ArtifactStore(root)
    analyses = AnalysisRepository(sf)

    swept, skipped = reclaim_orphan_artifacts(store, analyses)

    sf.kw["bind"].dispose()
    assert swept == [orphan_rid]
    assert skipped == []
    assert (root / kept_rid).is_dir()  # có row → giữ nguyên
    assert (root / "important-user-dir").is_dir()  # không phải UUID → không đụng
    assert (root / "readme.txt").is_file()
    assert not (root / orphan_rid).exists()


def test_sweep_fail_closed_when_db_unreadable(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "artifacts"
    orphan = str(__import__("uuid").uuid4())
    (root / orphan).mkdir(parents=True)
    store = ArtifactStore(root)
    sf = make_session_factory(tmp_path / "app.db")
    analyses = AnalysisRepository(sf)

    def boom() -> set[str]:
        raise RuntimeError("db unavailable")
    monkeypatch.setattr(analyses, "all_ids", boom)

    swept, skipped = reclaim_orphan_artifacts(store, analyses)
    sf.kw["bind"].dispose()
    assert (swept, skipped) == ([], [])
    assert (root / orphan).is_dir()  # fail-closed: KHÔNG xóa gì khi DB lỗi


def test_sweep_skips_locked_orphan_and_continues(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "artifacts"
    locked = str(__import__("uuid").uuid4())
    other = str(__import__("uuid").uuid4())
    for name in (locked, other):
        (root / name).mkdir(parents=True)
    store = ArtifactStore(root)
    sf = make_session_factory(tmp_path / "app.db")
    analyses = AnalysisRepository(sf)

    original_rmtree = storage.artifacts.shutil.rmtree

    def selective_rmtree(path, *args, **kwargs):
        if Path(path).name == locked:
            raise OSError("file in use")
        return original_rmtree(path, *args, **kwargs)
    monkeypatch.setattr(storage.artifacts.shutil, "rmtree", selective_rmtree)

    swept, skipped = reclaim_orphan_artifacts(store, analyses)
    sf.kw["bind"].dispose()
    assert swept == [other]
    assert skipped == [locked]
    assert (root / locked).is_dir()  # chờ lần sweep sau
    assert not (root / other).exists()


# ---------- Watch-mode prune (_handled/_seen_hashes hữu hạn) ----------

def _jpeg_bytes(seed: int = 1) -> bytes:
    pixels = np.random.default_rng(seed).integers(
        0, 255, (64, 64, 3), dtype=np.uint8
    )
    ok, encoded = cv2.imencode(".jpg", pixels)
    assert ok
    return encoded.tobytes()


def _config(data_dir: Path, *, max_pending: int = 1) -> IngestConfig:
    return IngestConfig(
        camera_label="door camera",
        max_pending=max_pending,
        methods=("ela",),
        data_dir=data_dir,
        consent_note="operator supplied note",
    )


class _RecordingService:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def analyze(self, request):
        self.calls.append(request.path)
        return SimpleNamespace(
            report_id=f"analysis-{len(self.calls)}",
            image_sha256=None,
        )


def test_watch_prune_drops_gone_file_entries(tmp_path: Path) -> None:
    watch = tmp_path / "incoming"
    watch.mkdir()
    a = watch / "a.jpg"
    a.write_bytes(_jpeg_bytes(7))
    service = _RecordingService()
    ingestor = FolderIngestor(_config(tmp_path / "data"), service, "h", settle_delay=0)

    first = ingestor.scan_once(watch)
    assert first.analyzed == 1 and service.calls == [a]
    sha_a = next(iter(ingestor._seen_hashes))
    assert a in ingestor._handled

    a.unlink()
    second = ingestor.scan_once(watch)  # prune cuối scan phải dọn cache chết
    assert second.analyzed == 0

    assert a not in ingestor._handled
    assert sha_a not in ingestor._seen_hashes
    assert not ingestor._sha_refs  # không còn path tham chiếu → bộ nhớ rỗng


def test_watch_prune_keeps_duplicate_suppression_for_present_files(
        tmp_path: Path) -> None:
    watch = tmp_path / "incoming"
    watch.mkdir()
    payload = _jpeg_bytes(9)
    a = watch / "a.jpg"
    copy = watch / "copy.jpg"
    a.write_bytes(payload)
    service = _RecordingService()
    ingestor = FolderIngestor(_config(tmp_path / "data"), service, "h", settle_delay=0)

    first = ingestor.scan_once(watch)
    assert first.analyzed == 1  # a analyzed; copy chưa tồn tại

    copy.write_bytes(payload)  # nội dung trùng, tên khác
    second = ingestor.scan_once(watch)
    assert second.analyzed == 0
    assert second.skipped_duplicate == 1  # sha còn present ref → vẫn bị chặn

    sha = next(iter(ingestor._seen_hashes))
    assert copy in ingestor._sha_refs[sha]
    assert service.calls == [a]  # không re-analyze

    a.unlink()  # còn copy tham chiếu → sha phải được giữ
    ingestor.scan_once(watch)
    assert sha in ingestor._seen_hashes

    copy.unlink()  # mọi ref biến mất → prune được sha
    ingestor.scan_once(watch)
    assert sha not in ingestor._seen_hashes
    assert not ingestor._sha_refs


# ---------- Config guards (chỉ vỡ khi user tự đổi yaml) ----------

def test_ela_empty_qualities_abstains_instead_of_crashing(ts) -> None:
    doc = _jpeg_doc()
    proxy = _OverridingCfg(ts, "ela", "qualities", [])

    result = ELA().analyze(doc, proxy)

    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.score is None
    assert any("qualities rỗng" in lim for lim in result.limitations)
    # Sanity: cùng doc với cfg shipped vẫn chạy verdict hợp lệ.
    sane = ELA().analyze(doc, ts)
    assert isinstance(sane.verdict, Verdict)


def test_dct_zigzag_validates_k_range() -> None:
    with pytest.raises(ValueError, match="n_coeffs"):
        zigzag_indices(8, 65)
    with pytest.raises(ValueError, match="n_coeffs"):
        zigzag_indices(8, -1)
    # k=0 là course-lock "0 hệ số → mảng rỗng" (test_zigzag_empty_course).
    assert len(zigzag_indices(8, 0)) == 0
    assert len(zigzag_indices(8, 64)) == 64
    assert len(zigzag_indices(8, 1)) == 1


def test_block_detector_fails_loud_on_oversized_n_coeffs(ts) -> None:
    doc = ImageDocument(
        array=np.random.default_rng(3).integers(
            0, 256, (64, 64, 3), dtype=np.uint8),
        width=64, height=64, channels=3, image_format="png",
        sha256="b" * 64, file_bytes=64 * 64 * 3, file_name="x.png")
    proxy = _OverridingCfg(ts, "block", "n_coeffs", 10_000)

    # Detector raise rõ ràng — service-isolation chuyển thành verdict error
    # kèm message (đã test ở test_analysis_persist_recovery).
    with pytest.raises(ValueError, match="n_coeffs"):
        BlockCopyMove().analyze(doc, proxy)
