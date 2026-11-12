"""Stats & Settings screens — offscreen Qt, không cần màn hình thật."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QTreeWidget

import imageguard.ui.screens.stats as stats_module
from imageguard.ui.screens.settings import SettingsScreen
from imageguard.ui.screens.stats import StatsScreen


class _StatsStub:
    """HistoryService-compatible stub: chỉ cần ``stats()``."""

    def __init__(self, payload) -> None:
        self.payload = payload

    def stats(self) -> dict:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_stats_rows_rendered(qtbot) -> None:
    stub = _StatsStub({
        "n_analyses": 3,
        "avg_duration_ms": 12.5,
        "label": "thống kê phiên — không phải benchmark",
        "by_verdict": {"suspicious_indication": 1, "no_indication": 2},
    })

    screen = StatsScreen(stub)
    qtbot.addWidget(screen)

    assert screen.table.rowCount() == 2
    assert "3 phân tích" in screen.lbl_head.text()
    assert screen.lbl_label.text() == "thống kê phiên — không phải benchmark"
    # verdict cell dùng label tiếng Việt, số lượng render thành str
    assert screen.table.item(0, 0).text() == "Có dấu hiệu nghi vấn"
    assert screen.table.item(0, 1).text() == "1"


def test_stats_error_path_shows_message_box(qtbot, monkeypatch) -> None:
    stub = _StatsStub(RuntimeError("boom"))
    warnings: list[tuple[str, str]] = []

    class _FakeMsgBox:
        @staticmethod
        def warning(parent, title, text):
            warnings.append((title, text))

    monkeypatch.setattr(stats_module, "QMessageBox", _FakeMsgBox)

    screen = StatsScreen(stub)
    qtbot.addWidget(screen)

    assert len(warnings) == 1
    title, text = warnings[0]
    assert title == "Lỗi cơ sở dữ liệu"
    assert "RuntimeError" in text
    # error path: không crash, bảng rỗng, headline giữ nguyên
    assert screen.table.rowCount() == 0
    assert screen.lbl_head.text() == ""


def test_settings_renders_sections_and_values(qtbot, ts) -> None:
    screen = SettingsScreen(ts)
    qtbot.addWidget(screen)

    tree = screen.findChild(QTreeWidget)
    assert tree is not None
    assert tree.columnCount() == 3  # Key | Value | Nguồn

    sections = [k for k, v in ts.raw.items() if isinstance(v, dict)]
    assert tree.topLevelItemCount() == len(sections)
    assert tree.topLevelItemCount() > 0
    # mỗi section có ít nhất một entry {value, source} được render
    assert any(
        tree.topLevelItem(i).childCount() > 0
        for i in range(tree.topLevelItemCount())
    )
