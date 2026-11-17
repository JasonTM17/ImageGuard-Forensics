"""Regression: persist-failure compensation (P2 custody repair).

Một persist fail phải để lại MỘT durable state: hoặc cleanup sạch (không
Analysis/MethodResult/Artifact row, không artifact dir), hoặc báo rõ cleanup
không hoàn tất + recovery — không bao giờ "save bình thường" với custody dở
dang. SQLite + ArtifactStore thật trong tmp_path, method tổng hợp thay
load_method; fault inject đúng từng điểm publication/registration.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import func, select

import imageguard.services.analysis_service as asvc
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.errors import ErrorResult
from imageguard.core.io import safe_open
from imageguard.core.models import AlgorithmResult, ForensicReport
from imageguard.services.analysis_service import AnalysisRequest, AnalysisService
from imageguard.services.report_service import ReportService
from imageguard.storage.artifacts import ArtifactStore
from imageguard.storage.db import make_session_factory
from imageguard.storage.models import (
    Analysis,
    Artifact,
    ImageAsset,
    MethodResult,
)
from imageguard.storage.repository import AnalysisRepository, ImageRepository


class _FakeAlgo:
    """Method tổng hợp — mask + heatmap → hai image-write thật qua store."""

    def analyze(self, doc, cfg, cancel) -> AlgorithmResult:
        h, w = doc.height, doc.width
        return AlgorithmResult(
            method=MethodName.NOISE_LOCAL,
            verdict=Verdict.INCONCLUSIVE,
            evidence={},
            limitations=[],
            mask=np.zeros((h, w), dtype=np.uint8),
            heatmap=np.zeros((h, w), dtype=np.float64),
            duration_ms=1)


@pytest.fixture()
def env(tmp_path: Path, ts):
    sf = make_session_factory(tmp_path / "app.db")
    store = ArtifactStore(tmp_path / "art")
    images = ImageRepository(sf)
    analyses = AnalysisRepository(sf)
    svc = AnalysisService(ts, images, analyses, store)
    yield SimpleNamespace(
        tmp=tmp_path, sf=sf, store=store, images=images,
        analyses=analyses, svc=svc)
    sf.kw["bind"].dispose()  # Windows: nhả file SQLite trước khi xoá tmp


def _use_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asvc, "load_method", lambda spec: _FakeAlgo())


def _analyze(env, png: Path) -> ForensicReport:
    out = env.svc.analyze(AnalysisRequest(
        path=png, methods=["noise_local"], include_exif=False))
    assert isinstance(out, ForensicReport)
    return out


def _fault_put_bytes(env, monkeypatch: pytest.MonkeyPatch,
                     fail_names: set[str]) -> None:
    orig = env.store.put_bytes

    def wrapped(analysis_id, filename, data):
        if filename in fail_names:
            raise OSError(f"controlled publication fault: {filename}")
        return orig(analysis_id, filename, data)

    monkeypatch.setattr(env.store, "put_bytes", wrapped)


def _fault_report_registration(env, monkeypatch: pytest.MonkeyPatch) -> None:
    orig = env.analyses.add_artifact

    def wrapped(analysis_id, kind, method, rel_path):
        if kind == "report":
            raise OSError("controlled report registration fault")
        return orig(analysis_id, kind, method, rel_path)

    monkeypatch.setattr(env.analyses, "add_artifact", wrapped)


def _raiser(exc: Exception):
    def f(*a, **k):
        raise exc
    return f


_PERSIST_WARN = "lưu lịch sử/artifact thất bại"


def _warn_lines(out: ForensicReport) -> list[str]:
    return [lim for lim in out.limitations if _PERSIST_WARN in lim]


def _counts(env, rid: str) -> tuple[int, int, int]:
    """(Analysis, MethodResult, Artifact) rows của rid."""
    with env.sf() as s:
        return (
            s.scalar(select(func.count(Analysis.id))
                     .where(Analysis.id == rid)) or 0,
            s.scalar(select(func.count(MethodResult.id))
                     .where(MethodResult.analysis_id == rid)) or 0,
            s.scalar(select(func.count(Artifact.id))
                     .where(Artifact.analysis_id == rid)) or 0)


def _assert_clean_custody(env, rid: str) -> None:
    assert _counts(env, rid) == (0, 0, 0)
    assert env.analyses.get(rid) is None
    assert not (env.tmp / "art" / rid).exists()


_PUT_BYTES_FAULTS = {
    "first_image_write": {"noise_local_mask.png"},
    "later_image_write": {"noise_local_heat.png"},
    "report_publish": {"report.json"},
}


@pytest.mark.parametrize("point", [*_PUT_BYTES_FAULTS, "report_registration"])
def test_failed_persist_leaves_no_partial_custody(
        env, monkeypatch: pytest.MonkeyPatch, valid_png: Path,
        point: str) -> None:
    _use_fake(monkeypatch)
    source_bytes = valid_png.read_bytes()

    ok = _analyze(env, valid_png)  # report trước — phải nguyên vẹn
    assert not _warn_lines(ok)
    assert env.analyses.get(ok.report_id) is not None

    if point == "report_registration":
        _fault_report_registration(env, monkeypatch)
    else:
        _fault_put_bytes(env, monkeypatch, _PUT_BYTES_FAULTS[point])

    out = _analyze(env, valid_png)
    assert out.report_id != ok.report_id
    assert _warn_lines(out), "persist fail phải báo limitation"
    _assert_clean_custody(env, out.report_id)

    # report trước + shared ImageAsset + source bytes nguyên vẹn.
    assert env.analyses.get(ok.report_id) is not None
    assert (env.tmp / "art" / ok.report_id / "report.json").is_file()
    with env.sf() as s:
        assert s.scalar(select(func.count(ImageAsset.id))) == 1
    assert valid_png.read_bytes() == source_bytes


@pytest.mark.parametrize(
    "fail_db,fail_art",
    [(True, False), (False, True), (True, True)],
    ids=["db_cleanup", "artifact_cleanup", "both"])
def test_cleanup_failure_reports_incomplete_recovery(
        env, monkeypatch: pytest.MonkeyPatch, valid_png: Path,
        fail_db: bool, fail_art: bool) -> None:
    _use_fake(monkeypatch)
    # registration fault: report.json ĐÃ publish → dir giữ file khi rmtree fail.
    _fault_report_registration(env, monkeypatch)
    if fail_db:
        monkeypatch.setattr(env.analyses, "delete",
                            _raiser(RuntimeError("controlled db cleanup")))
    if fail_art:
        monkeypatch.setattr(env.store, "delete_analysis",
                            _raiser(OSError("controlled artifact cleanup")))

    out = _analyze(env, valid_png)
    rid = out.report_id
    warn = "\n".join(_warn_lines(out))
    assert rid in warn
    assert "không hoàn tất" in warn and "recovery" in warn
    if fail_db:
        assert "database" in warn and "RuntimeError" in warn
        assert env.analyses.get(rid) is not None  # DB cleanup đã fail
    else:
        assert _counts(env, rid) == (0, 0, 0)  # DB cleanup đã chạy xong
    if fail_art:
        assert "artifacts" in warn and "OSError" in warn
        assert (env.tmp / "art" / rid).is_dir()  # dir còn — documented limit
        orphan = ReportService(env.store).read_report_json(rid)
        assert orphan and orphan["report_id"] == rid  # orphan exportable: limit
    else:
        assert not (env.tmp / "art" / rid).exists()  # artifacts vẫn được dọn
    # Recovery text mới: UUID + stage + loại exception — không raw message,
    # không absolute path.
    assert "controlled" not in warn
    assert str(env.tmp) not in warn


def test_recovery_error_chains_original(
        env, monkeypatch: pytest.MonkeyPatch, valid_png: Path, ts) -> None:
    doc = safe_open(valid_png, ts)
    assert not isinstance(doc, ErrorResult)
    result = _FakeAlgo().analyze(doc, ts, None)
    rid = str(uuid.uuid4())
    report = ForensicReport(
        report_id=rid, image_sha256=doc.sha256, image_name=doc.file_name,
        image_width=doc.width, image_height=doc.height,
        created_at="2026-01-01T00:00:00+00:00", tool_version="test",
        methods=[result], overall_verdict=Verdict.INCONCLUSIVE)
    _fault_put_bytes(env, monkeypatch, {"report.json"})
    monkeypatch.setattr(env.analyses, "delete",
                        _raiser(RuntimeError("controlled db cleanup")))
    monkeypatch.setattr(env.store, "delete_analysis",
                        _raiser(OSError("controlled artifact cleanup")))

    with pytest.raises(asvc._PersistRecoveryError) as ei:
        env.svc._persist(report, doc, [result], 1)
    assert isinstance(ei.value.__cause__, OSError)
    assert "controlled publication fault" in str(ei.value.__cause__)
    assert rid in str(ei.value)
    assert "controlled" not in str(ei.value)
    assert str(env.tmp) not in str(ei.value)


def test_existing_analysis_row_rejects_without_deletion(
        env, monkeypatch: pytest.MonkeyPatch, valid_png: Path) -> None:
    _use_fake(monkeypatch)
    ok = _analyze(env, valid_png)
    rid = ok.report_id
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(rid))

    out = _analyze(env, valid_png)
    assert out.report_id == rid
    assert _warn_lines(out), "reject phải báo limitation"
    row = env.analyses.get(rid)
    assert row is not None
    assert {a.kind for a in row.artifacts} == {"mask", "heatmap", "report"}
    assert (env.tmp / "art" / rid / "report.json").is_file()


def test_existing_artifact_dir_rejects_without_deletion(
        env, monkeypatch: pytest.MonkeyPatch, valid_png: Path) -> None:
    _use_fake(monkeypatch)
    rid = str(uuid.uuid4())
    d = env.tmp / "art" / rid
    d.mkdir(parents=True)
    (d / "sentinel.txt").write_text("pre-existing", encoding="utf-8")
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(rid))

    out = _analyze(env, valid_png)
    assert out.report_id == rid
    assert _warn_lines(out)
    assert env.analyses.get(rid) is None
    assert (d / "sentinel.txt").read_text(encoding="utf-8") == "pre-existing"


def test_normal_persist_still_works(
        env, monkeypatch: pytest.MonkeyPatch, valid_png: Path) -> None:
    _use_fake(monkeypatch)
    out = _analyze(env, valid_png)
    assert not _warn_lines(out)
    row = env.analyses.get(out.report_id)
    assert row is not None
    assert {a.kind for a in row.artifacts} == {"mask", "heatmap", "report"}
    d = env.tmp / "art" / out.report_id
    assert {p.name for p in d.iterdir()} == {
        "noise_local_mask.png", "noise_local_heat.png", "report.json"}
    rj = json.loads((d / "report.json").read_text(encoding="utf-8"))
    assert rj["report_id"] == out.report_id
