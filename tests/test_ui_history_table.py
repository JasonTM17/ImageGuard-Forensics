"""History table UX — column sizing, Vietnamese headers, tooltips."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QHeaderView

from imageguard.ui import strings


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


def _seed_one_row(history) -> None:
    from imageguard.storage.repository import ImageRepository

    images = ImageRepository(history.analyses._sf)
    img = images.get_or_create("ab" * 32, "ảnh_mẫu.png", 8, 8, "png")
    history.analyses.save_report("r1", img.id, _report_dict("suspicious_indication"), "h", 10)


@pytest.fixture()
def services(tmp_path: Path):
    from imageguard.ui.app import build_services

    return build_services(db_path=tmp_path / "t.db", artifacts_root=tmp_path / "art")


def _header_labels(table) -> list[str]:
    return [
        table.horizontalHeaderItem(c).text()
        for c in range(table.columnCount())
        if table.horizontalHeaderItem(c) is not None
    ]


def test_history_headers_vietnamese(qtbot, services) -> None:
    from imageguard.ui.screens.history import HistoryScreen

    _, history = services
    screen = HistoryScreen(history)
    qtbot.addWidget(screen)
    labels = _header_labels(screen.table)
    assert labels[:4] == [
        strings.HIST_COL_TIME,
        strings.HIST_COL_IMAGE,
        strings.HIST_COL_VERDICT,
        strings.HIST_COL_METHODS,
    ]
    assert "Verdict" not in labels
    assert "Methods" not in labels


def test_history_column_resize_modes(qtbot, services) -> None:
    from imageguard.ui.screens.history import HistoryScreen

    _, history = services
    screen = HistoryScreen(history)
    qtbot.addWidget(screen)
    hdr = screen.table.horizontalHeader()
    assert hdr.sectionResizeMode(0) == QHeaderView.ResizeMode.ResizeToContents
    assert hdr.sectionResizeMode(1) == QHeaderView.ResizeMode.Stretch
    assert hdr.sectionResizeMode(2) == QHeaderView.ResizeMode.ResizeToContents
    assert hdr.sectionResizeMode(3) == QHeaderView.ResizeMode.ResizeToContents


def test_history_verdict_tooltip_and_width(qtbot, services) -> None:
    from imageguard.ui.screens.history import HistoryScreen

    _, history = services
    _seed_one_row(history)
    screen = HistoryScreen(history)
    qtbot.addWidget(screen)
    screen.refresh()

    verdict_text = strings.VERDICT_LABEL["suspicious_indication"]
    item = screen.table.item(0, 2)
    assert item is not None
    assert item.text() == verdict_text
    assert item.toolTip() == verdict_text

    fm = QFontMetrics(screen.table.font())
    min_w = fm.horizontalAdvance(verdict_text)
    for width in (1280, 800):
        screen.resize(width, 600)
        screen.show()
        qtbot.wait(100)
        assert screen.table.columnWidth(2) >= min_w


def test_history_table_row_counts(qtbot, services) -> None:
    from imageguard.ui.screens.history import HistoryScreen

    _, history = services
    empty = HistoryScreen(history)
    qtbot.addWidget(empty)
    assert empty.table.rowCount() == 0
    assert not empty.lbl_empty.isHidden()

    rows = [
        {
            "id": f"row-{i:03d}",
            "created_at": "2026-11-12 08:40:00.000000",
            "verdict": "no_indication",
            "image_name": f"img_{i}.png",
            "n_methods": 3,
        }
        for i in range(100)
    ]

    class _ListOnly:
        def list(self, limit: int = 100) -> list[dict]:
            return rows[:limit]

    many = HistoryScreen(_ListOnly())  # type: ignore[arg-type]
    qtbot.addWidget(many)
    many.refresh()
    assert many.table.rowCount() == 100
    assert not many.lbl_empty.isVisible()


def test_stats_headers_and_resize(qtbot, services) -> None:
    from imageguard.ui.screens.stats import StatsScreen

    _, history = services
    _seed_one_row(history)
    screen = StatsScreen(history)
    qtbot.addWidget(screen)
    assert _header_labels(screen.table) == [
        strings.STATS_COL_VERDICT,
        strings.STATS_COL_COUNT,
    ]
    hdr = screen.table.horizontalHeader()
    assert hdr.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert hdr.sectionResizeMode(1) == QHeaderView.ResizeMode.ResizeToContents
    label = strings.VERDICT_LABEL["suspicious_indication"]
    cell = screen.table.item(0, 0)
    assert cell is not None
    assert cell.toolTip() == label
