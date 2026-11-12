"""UI accessibility — accessibleName và thông báo trạng thái ngắn."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from imageguard.ui import strings


@pytest.fixture()
def services(tmp_path):
    from imageguard.ui.app import build_services

    return build_services(db_path=tmp_path / "t.db",
                          artifacts_root=tmp_path / "art")


def _assert_named(widget, expected: str) -> None:
    assert widget.accessibleName() == expected


def test_analyze_interactive_accessible_names(qtbot, services):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    a = w.analyze
    _assert_named(a.btn_open, strings.A11Y_ANALYZE_OPEN)
    _assert_named(a.btn_run, strings.A11Y_ANALYZE_RUN)
    _assert_named(a.btn_cancel, strings.A11Y_ANALYZE_CANCEL)
    _assert_named(a.progress, strings.A11Y_ANALYZE_PROGRESS)
    _assert_named(a.overlay_box, strings.A11Y_ANALYZE_OVERLAY)
    _assert_named(a.canvas, strings.A11Y_CANVAS)
    assert a.canvas.focusPolicy() == Qt.FocusPolicy.StrongFocus


def test_enhance_blend_readout_and_a11y(qtbot, services):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    e = w.enhance
    _assert_named(e.slider_blend, strings.A11Y_ENHANCE_BLEND)
    _assert_named(e.lbl_blend_value, strings.A11Y_ENHANCE_BLEND_VALUE)
    assert e.lbl_blend_value.text() == strings.FMT_BLEND_VALUE.format(value=100)
    e._set_blend_readout(42)
    assert e.lbl_blend_value.text() == strings.FMT_BLEND_VALUE.format(value=42)


def test_history_no_selection_status(qtbot, services):
    from imageguard.ui.screens.history import HistoryScreen

    _, history = services
    h = HistoryScreen(history)
    qtbot.addWidget(h)
    _assert_named(h.btn_reopen, strings.A11Y_HISTORY_REOPEN)
    h.table.setCurrentCell(-1, -1)
    h._reopen()
    assert h.lbl_status.text() == strings.MSG_HISTORY_NO_SELECTION
    h._export()
    assert h.lbl_status.text() == strings.MSG_HISTORY_NO_SELECTION
    h._delete()
    assert h.lbl_status.text() == strings.MSG_HISTORY_NO_SELECTION


def test_analyze_busy_drop_status(qtbot, services, valid_png):
    from PySide6.QtCore import QMimeData, QPoint, QUrl
    from PySide6.QtGui import QDropEvent

    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    w.analyze.load_path(valid_png)
    w.analyze._worker = object()
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(valid_png))])
    evt = QDropEvent(
        QPoint(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    w.analyze.dropEvent(evt)
    assert w.analyze.lbl_status.text() == strings.MSG_BUSY_CANNOT_OPEN


def test_verdict_banner_signals_prefix(qtbot, services):
    from imageguard.core.enums import Verdict
    from imageguard.ui.widgets.verdict_panel import VerdictBanner

    banner = VerdictBanner()
    qtbot.addWidget(banner)
    _assert_named(banner, strings.A11Y_VERDICT_BANNER)
    banner.set_verdict(Verdict.INCONCLUSIVE, ["ELA (JPEG error levels)"])
    assert strings.VERDICT_SIGNALS_PREFIX in banner._sub.text()
    assert "Signals:" not in banner._sub.text()
