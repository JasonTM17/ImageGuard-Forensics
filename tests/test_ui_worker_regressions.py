"""Regression: AnalysisWorker BaseException handling + history selection reset.

Fix 1 (ui/workers.py): AnalysisWorker.run catches BaseException — SystemExit/
KeyboardInterrupt emit `failed` (với prefix tên loại) rồi mới re-raise; Exception
thường chỉ emit failed. Nếu không báo failed, màn hình analyze khóa vĩnh viễn
(_worker không bao giờ reset).

Fix 2 (ui/screens/history.py): refresh() clear selection TRƯỚC khi refill —
currentRow cũ không được trỏ sang bản ghi khác sau khi một row bị xóa.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from imageguard.services.analysis_service import AnalysisRequest
from imageguard.ui.workers import AnalysisWorker


class _FakeService:
    """AnalysisService giả: analyze() luôn raise exception được tiêm vào."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def analyze(self, req: AnalysisRequest) -> Any:
        raise self._exc


def _request(tmp_path: Path) -> AnalysisRequest:
    return AnalysisRequest(
        path=tmp_path / "img.jpg",
        methods=["keypoint"],
        include_exif=False,
    )


def test_analysis_worker_base_exception_emits_failed_and_reraises(
    qtbot, tmp_path: Path
) -> None:
    worker = AnalysisWorker(_FakeService(SystemExit("boom")), _request(tmp_path))
    failed: list[str] = []
    # Cùng thread (gọi run() trực tiếp) → signal phát đồng bộ, không cần event loop.
    worker.signals.failed.connect(failed.append)
    with pytest.raises(SystemExit):
        worker.run()
    assert len(failed) == 1
    assert "SystemExit" in failed[0]


def test_analysis_worker_exception_emits_failed_only(
    qtbot, tmp_path: Path
) -> None:
    worker = AnalysisWorker(_FakeService(RuntimeError("x")), _request(tmp_path))
    failed: list[str] = []
    worker.signals.failed.connect(failed.append)
    worker.run()  # Exception thường: nuốt sau khi báo failed, không re-raise
    assert failed == ["x"]


def _report_dict(verdict: str) -> dict[str, Any]:
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


def _seed(history, rid: str, name: str, sha: str) -> None:
    from imageguard.storage.repository import ImageRepository

    images = ImageRepository(history.analyses._sf)
    img = images.get_or_create(sha, name, 8, 8, "png")
    history.analyses.save_report(
        rid, img.id, _report_dict("no_indication"), "h", 10)


def test_history_refresh_clears_stale_selection(qtbot, tmp_path: Path) -> None:
    from imageguard.ui.app import build_services
    from imageguard.ui.screens.history import HistoryScreen

    _, history = build_services(
        db_path=tmp_path / "t.db", artifacts_root=tmp_path / "art")
    _seed(history, "r1", "a.png", "01" * 32)
    _seed(history, "r2", "b.png", "02" * 32)
    screen = HistoryScreen(history)
    qtbot.addWidget(screen)
    screen.refresh()
    assert screen.table.rowCount() == 2

    screen.table.setCurrentCell(0, 0)
    assert screen.table.currentRow() == 0
    selected = screen._selected_id()
    assert selected in {"r1", "r2"}

    # Xóa đúng row đang được chọn TRỰC TIẾP khỏi DB (giả lập xóa bên ngoài),
    # rồi refresh: selection cũ không được trỏ sang bản ghi khác.
    history.analyses.delete(selected)
    screen.refresh()
    assert screen.table.rowCount() == 1
    assert screen.table.currentRow() == -1
    assert screen._selected_id() is None
