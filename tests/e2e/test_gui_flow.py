"""End-to-end GUI flow — real AnalysisWorker, offscreen QPA.

Chạy worker thật (không mock service/worker/engine); chỉ patch hộp thoại
blocking. Mọi chờ đều qua qtbot.waitUntil.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QFileDialog, QMessageBox

from imageguard.evaluation.synthforge import generate_set
from imageguard.ui import strings
from imageguard.ui.screens.history import AnalysisDetailDialog
from imageguard.ui.widgets.verdict_panel import MethodCard

ALL_CHECKS = ("chk_kp", "chk_bl", "chk_ela", "chk_noise", "chk_nloc",
              "chk_freq", "chk_spl")


def _check_all(screen) -> None:
    for name in ALL_CHECKS:
        getattr(screen, name).setChecked(True)


@pytest.fixture(scope="module")
def synth_384(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth384")
    generate_set(n=1, seed=7, out_dir=d, size=(384, 384), tier="easy")
    return d / "easy_0000_F.png"


@pytest.fixture(scope="module")
def synth_768(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth768")
    generate_set(n=1, seed=11, out_dir=d, size=(768, 768), tier="easy")
    return d / "easy_0000_F.png"


@pytest.fixture()
def services(tmp_path):
    from imageguard.ui.app import build_services

    return build_services(db_path=tmp_path / "t.db",
                          artifacts_root=tmp_path / "art")


@pytest.fixture()
def window(qtbot, services):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    w.show()
    return w


def _wait_idle(qtbot, screen, timeout=30000):
    qtbot.waitUntil(
        lambda: screen._worker is None
        and not screen.btn_cancel.isEnabled()
        and screen.btn_run.isEnabled(),
        timeout=timeout,
    )


def _drop_event(path):
    """Build QDropEvent; trả (evt, mime) — mime phải sống lâu hơn evt."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    evt = QDropEvent(
        QPoint(50, 50),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    return evt, mime


def test_happy_path_export_history_reopen_delete(
    qtbot, window, synth_384, tmp_path, monkeypatch
):
    a = window.analyze
    _check_all(a)
    a.load_path(synth_384)
    assert a.btn_run.isEnabled()

    a.btn_run.click()
    qtbot.waitUntil(
        lambda: a._worker is None and a.banner.isVisible()
        and a.btn_export.isVisible(),
        timeout=30000,
    )

    assert len(a.cards_inner.findChildren(MethodCard)) >= 7
    assert a.overlay_box.isVisible()
    assert a.overlay_box.count() >= 2
    report = a._report
    assert report is not None
    assert a.banner._label.text() == strings.VERDICT_LABEL[
        report.overall_verdict.value]

    md_path = tmp_path / "bao_cao.md"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        lambda *args, **kwargs: (str(md_path), ""),
    )
    monkeypatch.setattr(
        QMessageBox, "information",
        lambda *args, **kwargs: QMessageBox.StandardButton.Ok,
    )
    a.btn_export.click()
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert content.strip()
    assert report.report_id in content

    # History tab refresh là lúc bảng được nạp.
    window.tabs.setCurrentIndex(2)
    qtbot.waitUntil(lambda: window.history.table.rowCount() == 1, timeout=10000)
    assert window.history.table.item(0, 1).text() == synth_384.name
    analysis_id = window.history.table.item(0, 4).text()

    window.history.table.setCurrentCell(0, 1)
    opened = []

    def record_exec(self):
        opened.append(self._aid)

    monkeypatch.setattr(AnalysisDetailDialog, "exec", record_exec)
    window.history.btn_reopen.click()
    assert opened == [analysis_id]

    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    window.history.btn_delete.click()
    qtbot.waitUntil(
        lambda: window.history.table.rowCount() == 0
        and window.history.lbl_empty.isVisible(),
        timeout=10000,
    )

    # Xoá khỏi history không được xoá kết quả đang mở.
    window.tabs.setCurrentIndex(0)
    assert window.isVisible()
    assert a.banner.isVisible()
    assert a._report is report
    window.tabs.setCurrentIndex(2)
    assert window.history.table.rowCount() == 0


def test_corrupt_file_rejected(qtbot, window, tmp_path):
    a = window.analyze
    bad = tmp_path / "hong.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n khong phai anh")
    a.load_path(bad)
    assert not a.btn_run.isEnabled()
    assert a.lbl_err.text() != ""
    assert a.canvas._base is None
    assert window.isVisible()


def test_cancel_during_run(qtbot, window, synth_768):
    a = window.analyze
    _check_all(a)
    a.load_path(synth_768)
    a.btn_run.click()
    qtbot.waitUntil(lambda: a.progress.value() > 0, timeout=10000)
    a.btn_cancel.click()
    _wait_idle(qtbot, a)
    assert not a.btn_cancel.isEnabled()


def test_run_twice(qtbot, window, synth_384):
    a = window.analyze
    _check_all(a)
    a.load_path(synth_384)
    for _ in range(2):
        a.btn_run.click()
        qtbot.waitUntil(
            lambda: a._worker is None and a.banner.isVisible()
            and a.btn_export.isVisible(),
            timeout=30000,
        )
    window.tabs.setCurrentIndex(2)
    qtbot.waitUntil(lambda: window.history.table.rowCount() == 2, timeout=10000)


def test_drop_while_running_ignored(qtbot, window, synth_768, synth_384):
    a = window.analyze
    _check_all(a)
    a.load_path(synth_768)
    running_path = a._path
    running_name = a.lbl_file.text()
    a.btn_run.click()
    qtbot.waitUntil(lambda: a.progress.value() > 0, timeout=10000)

    evt, mime = _drop_event(synth_384)
    a.dropEvent(evt)
    assert mime.hasUrls()

    assert a._path == running_path
    assert a.lbl_file.text() == running_name
    _wait_idle(qtbot, a)
