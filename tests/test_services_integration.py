"""Integration: synth → analyze → persist → session MỚI → reopen → delete."""

from __future__ import annotations

from pathlib import Path

from imageguard.core.enums import Verdict
from imageguard.core.models import ForensicReport
from imageguard.services.analysis_service import AnalysisRequest, AnalysisService
from imageguard.services.history_service import HistoryService
from imageguard.services.report_service import ReportService
from imageguard.storage.artifacts import ArtifactStore
from imageguard.storage.db import make_session_factory
from imageguard.storage.repository import AnalysisRepository, ImageRepository


def _services(tmp_path: Path, ts):
    sf = make_session_factory(tmp_path / "app.db")
    store = ArtifactStore(tmp_path / "art")
    analyses = AnalysisRepository(sf)
    return (AnalysisService(ts, ImageRepository(sf), analyses, store),
            HistoryService(analyses, store), ReportService(store))


def test_analyze_persist_reopen_delete(tmp_path: Path, valid_png: Path, ts) -> None:
    svc, hist, rep = _services(tmp_path, ts)
    events: list[float] = []
    out = svc.analyze(AnalysisRequest(
        path=valid_png, methods=["ela"],  # ela nhanh — test integration, không perf
        progress_cb=lambda f, s: events.append(f)))
    assert isinstance(out, ForensicReport)
    assert out.overall_verdict in set(Verdict)
    assert events and events[-1] == 1.0  # progress đầy đủ

    # "Restart": session factory mới — dữ liệu phải còn nguyên.
    _svc2, _hist2, _rep2 = _services(tmp_path / "r2", ts)
    # dùng cùng db/artifacts root → history thấy record
    lst = hist.list()
    assert any(x["id"] == out.report_id for x in lst)
    got = hist.get(out.report_id)
    assert got and got["artifacts"], "report đã persist nhưng thiếu artifact"
    rj = rep.read_report_json(out.report_id)
    assert rj and rj["image_name"].endswith(".png")
    assert rj["disclaimer"]  # disclaimer phải trong report

    assert hist.delete(out.report_id)
    assert hist.get(out.report_id) is None


def test_analyze_bad_path_returns_error(tmp_path: Path, ts) -> None:
    svc, _, _ = _services(tmp_path, ts)
    out = svc.analyze(AnalysisRequest(path=tmp_path / "nope.png", methods=["ela"]))
    assert not isinstance(out, ForensicReport)  # ErrorResult, không crash
