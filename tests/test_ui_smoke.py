"""UI smoke tests — offscreen QPA, không cần màn hình thật."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")

from imageguard.ui import strings


@pytest.fixture()
def services(tmp_path):
    from imageguard.ui.app import build_services

    return build_services(db_path=tmp_path / "t.db",
                          artifacts_root=tmp_path / "art")


def test_app_shell(qtbot, services):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    assert w.tabs.count() == 3
    texts = [w.tabs.tabText(i) for i in range(3)]
    assert texts == [strings.TAB_ANALYZE, strings.TAB_HISTORY,
                     strings.TAB_SETTINGS]
    # Disclaimer cố định luôn hiển thị
    found = w.findChildren(type(w.centralWidget().layout().itemAt(1).widget()))
    assert any(strings.DISCLAIMER[:20] in d.findChildren(
        __import__("PySide6.QtWidgets", fromlist=["QLabel"]).QLabel)[0].text()
        for d in found if d.findChildren(
        __import__("PySide6.QtWidgets", fromlist=["QLabel"]).QLabel))


def test_analyze_open_preview(qtbot, services, valid_png):
    from imageguard.ui.main_window import MainWindow

    analysis, _ = services
    w = MainWindow(analysis, _)
    qtbot.addWidget(w)
    w.analyze.load_path(valid_png)
    assert w.analyze.btn_run.isEnabled()
    assert "256" in w.analyze.lbl_meta.text()
    assert w.analyze.canvas._base is not None


def test_analyze_rejects_bad_file(qtbot, services, tmp_path):
    from imageguard.ui.main_window import MainWindow

    analysis, _ = services
    w = MainWindow(analysis, _)
    qtbot.addWidget(w)
    bad = tmp_path / "fake.png"
    bad.write_bytes(b"not an image at all")
    w.analyze.load_path(bad)
    assert not w.analyze.btn_run.isEnabled()
    assert w.analyze.lbl_err.text() != ""


def test_history_screen_empty(qtbot, services):
    from imageguard.ui.screens.history import HistoryScreen

    _, history = services
    h = HistoryScreen(history)
    qtbot.addWidget(h)
    assert h.lbl_empty.isVisible() or h.table.rowCount() == 0


def test_strings_no_forbidden_wording():
    """Integrity: mọi UI text không được overclaim."""
    import inspect

    banned = ["chắc chắn", "100%", "xác suất giả mạo", "kết luận giả mạo"]
    for name, val in inspect.getmembers(strings):
        if name.isupper() and isinstance(val, str):
            for b in banned:
                assert b not in val.lower(), f"{name} chứa '{b}'"
