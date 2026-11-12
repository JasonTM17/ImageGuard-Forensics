"""UI smoke tests — offscreen QPA, không cần màn hình thật."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from imageguard.ui import strings

_REMOVED_MAIN_DISCLAIMER_SNIPPET = "Công cụ phát hiện dấu hiệu nghi ngờ"


def _main_window_label_texts(window) -> list[str]:
    from PySide6.QtWidgets import QLabel

    return [lbl.text() for lbl in window.findChildren(QLabel)]


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
    assert w.tabs.count() == 5
    texts = [w.tabs.tabText(i) for i in range(5)]
    assert texts == [
        strings.TAB_ANALYZE,
        strings.TAB_ENHANCE,
        strings.TAB_HISTORY,
        "Thống kê",
        strings.TAB_SETTINGS,
    ]
    # Không còn thanh disclaimer cố định ngoài tab
    central_lay = w.centralWidget().layout()
    assert central_lay.count() == 1
    assert w.tabs is central_lay.itemAt(0).widget()
    assert not any(
        _REMOVED_MAIN_DISCLAIMER_SNIPPET in t for t in _main_window_label_texts(w)
    )

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


def test_canvas_and_window_drag_drop(qtbot, services, valid_png):
    from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
    from PySide6.QtGui import QDropEvent

    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(valid_png))])
    evt = QDropEvent(
        QPoint(50, 50),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )

    # Thử thả trực tiếp lên canvas
    w.analyze.canvas.dropEvent(evt)
    assert w.analyze._path == valid_png
    assert w.analyze.btn_run.isEnabled()


def test_load_path_ignored_while_running_and_bad_file_clears_canvas(qtbot, services, valid_png, tmp_path):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    w.analyze.load_path(valid_png)
    assert w.analyze.canvas._base is not None

    w.analyze._worker = object()  # mô phỏng đang chạy
    w.analyze.load_path(tmp_path / "other.png")
    assert w.analyze._path == valid_png
    w.analyze._worker = None

    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    w.analyze.load_path(bad)
    assert w.analyze._path is None
    assert w.analyze.canvas._base is None


def _drop(urls):
    """Build QDropEvent; trả (evt, mime) — mime phải sống lâu hơn evt
    vì QDropEvent không nhận sở hữu QMimeData."""
    from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
    from PySide6.QtGui import QDropEvent

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(u)) for u in urls])
    evt = QDropEvent(
        QPoint(50, 50),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    return evt, mime


def test_main_window_drop_directory_ignored(qtbot, services, tmp_path, monkeypatch):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)

    calls = []
    real_load = w.analyze.load_path

    def spy_load(path):
        calls.append(path)
        return real_load(path)

    monkeypatch.setattr(w.analyze, "load_path", spy_load)

    folder = tmp_path / "thu_muc"
    folder.mkdir()
    evt, mime = _drop([folder])
    assert mime.hasUrls()
    w.dropEvent(evt)

    assert calls == []
    assert w.analyze._path is None


def test_main_window_drop_multiple_urls_loads_only_first(
    qtbot, services, valid_png, tmp_path
):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)

    second = tmp_path / "second.png"
    second.write_bytes(valid_png.read_bytes())

    evt, mime = _drop([valid_png, second])
    assert mime.hasUrls()
    w.dropEvent(evt)

    assert w.analyze._path == valid_png
    assert w.analyze.btn_run.isEnabled()


def test_main_window_drop_non_image_file_shows_error(qtbot, services, tmp_path):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)

    bad = tmp_path / "ghi_chu.txt"
    bad.write_text("khong phai anh")

    evt, mime = _drop([bad])
    assert mime.hasUrls()
    w.dropEvent(evt)

    assert w.analyze._path is None
    assert not w.analyze.btn_run.isEnabled()
    assert w.analyze.lbl_err.text() != ""
    assert w.analyze.canvas._base is None


def test_run_without_methods_shows_error_and_no_worker(qtbot, services, valid_png):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    w.analyze.load_path(valid_png)
    for c in (w.analyze.chk_kp, w.analyze.chk_bl, w.analyze.chk_ela,
              w.analyze.chk_noise, w.analyze.chk_nloc, w.analyze.chk_freq,
              w.analyze.chk_spl):
        c.setChecked(False)

    before = w.analyze.progress.value()
    w.analyze._run()

    assert w.analyze.lbl_err.text() == (
        "Vui lòng chọn ít nhất một phương pháp phân tích.")
    assert w.analyze._worker is None
    assert w.analyze.progress.value() == before


def test_progress_format_reset_when_running_stops(qtbot, services, valid_png):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    w.analyze.load_path(valid_png)

    w.analyze._on_progress(0.4, "ela")
    assert "ELA" in w.analyze.progress.format()
    assert w.analyze.progress.format() != "%p%"

    w.analyze._set_running(False)
    assert w.analyze.progress.format() == "%p%"
