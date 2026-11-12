"""HistoryService stats aggregation and delete of an unknown id."""

from __future__ import annotations

from pathlib import Path

from imageguard.services.history_service import HistoryService
from imageguard.storage.artifacts import ArtifactStore
from imageguard.storage.db import make_session_factory
from imageguard.storage.repository import AnalysisRepository, ImageRepository


def _report_dict(verdict: str) -> dict:
    return {
        "tool_version": "0.1.0",
        "overall_verdict": verdict,
        "limitations": [],
        "methods": [{
            "method": "copy_move_keypoint",
            "verdict": verdict,
            "score": 0.0,
            "score_definition": "d",
            "evidence": {},
            "limitations": [],
            "duration_ms": 5,
        }],
    }


def test_stats_aggregates_verdicts_and_duration(tmp_path: Path) -> None:
    sf = make_session_factory(tmp_path / "h.db")
    images = ImageRepository(sf)
    analyses = AnalysisRepository(sf)

    a = images.get_or_create("ab" * 32, "a.png", 8, 8, "png")
    b = images.get_or_create("cd" * 32, "b.png", 8, 8, "png")
    analyses.save_report("r1", a.id, _report_dict("suspicious_indication"), "h", 10)
    analyses.save_report("r2", b.id, _report_dict("no_indication"), "h", 20)
    analyses.save_report("r3", b.id, _report_dict("no_indication"), "h", 30)

    hist = HistoryService(analyses, ArtifactStore(tmp_path / "art"))
    stats = hist.stats()

    assert stats["n_analyses"] == 3
    assert stats["by_verdict"] == {"suspicious_indication": 1, "no_indication": 2}
    assert stats["avg_duration_ms"] == 20.0
    assert stats["label"] == "thống kê phiên — không phải benchmark"


def test_delete_unknown_id_returns_true_and_is_noop(tmp_path: Path) -> None:
    sf = make_session_factory(tmp_path / "h.db")
    analyses = AnalysisRepository(sf)
    store = ArtifactStore(tmp_path / "art")
    hist = HistoryService(analyses, store)

    # Hợp đồng hiện tại: delete luôn trả True (idempotent), không crash với id lạ.
    assert hist.delete("does-not-exist") is True
    assert hist.get("does-not-exist") is None
    assert not (tmp_path / "art" / "does-not-exist").exists()
