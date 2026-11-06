"""Enhance tab — real widgets, patch dialogs only, offscreen QPA."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog, QMessageBox

from imageguard.ui import strings
from imageguard.ui import workers as workers_mod


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


def _enhance(window):
    window.tabs.setCurrentIndex(1)
    return window.enhance


def _banner_text(e) -> str:
    from PySide6.QtWidgets import QLabel

    labs = e.banner.findChildren(QLabel)
    assert labs
    return labs[0].text()


def test_enhance_banner_visible(qtbot, window):
    e = _enhance(window)
    assert e.banner.isVisible()
    assert strings.ENHANCE_BANNER in _banner_text(e)


def test_open_image_does_not_mutate_source_bytes(qtbot, window, valid_png):
    e = _enhance(window)
    before = valid_png.read_bytes()
    e.load_path(valid_png)
    assert valid_png.read_bytes() == before


def test_corrupt_file_shows_error_run_disabled_no_export(qtbot, window, tmp_path):
    e = _enhance(window)
    bad = tmp_path / "hong.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n khong phai anh")
    e.load_path(bad)
    assert e.lbl_err.text() != ""
    assert not e.btn_run.isEnabled()
    assert e.btn_export.isHidden()


def test_cancel_discards_result_export_hidden(
    qtbot, window, valid_png, monkeypatch,
):
    e = _enhance(window)
    real_derive = workers_mod.derive

    def slow_derive(image, kind, params):
        time.sleep(0.6)
        return real_derive(image, kind, params)

    monkeypatch.setattr(workers_mod, "derive", slow_derive)

    e.load_path(valid_png)
    e.btn_run.click()
    qtbot.waitUntil(lambda: e._worker is not None, timeout=3000)
    e.btn_cancel.click()
    qtbot.waitUntil(
        lambda: e._worker is None and not e.btn_cancel.isEnabled(),
        timeout=30000,
    )
    assert e.btn_export.isHidden()


def test_second_run_does_not_start_second_worker(
    qtbot, window, valid_png, monkeypatch,
):
    e = _enhance(window)
    real_derive = workers_mod.derive
    gate = threading.Event()

    def gated_derive(image, kind, params):
        gate.wait(timeout=30)
        return real_derive(image, kind, params)

    monkeypatch.setattr(workers_mod, "derive", gated_derive)

    starts: list = []
    real_start = e._pool.start

    def spy_start(runnable):
        starts.append(runnable)
        return real_start(runnable)

    monkeypatch.setattr(e._pool, "start", spy_start)

    e.load_path(valid_png)
    e.btn_run.click()
    qtbot.waitUntil(lambda: e._worker is not None, timeout=3000)
    assert len(starts) == 1

    e.btn_run.click()
    e.btn_run.click()
    assert len(starts) == 1

    gate.set()
    qtbot.waitUntil(
        lambda: e._worker is None and e.btn_export.isVisible(),
        timeout=30000,
    )


def test_export_writes_png_json_provenance(qtbot, window, valid_png, tmp_path, monkeypatch):
    e = _enhance(window)
    e.load_path(valid_png)
    e.btn_run.click()
    qtbot.waitUntil(lambda: e.btn_export.isVisible(), timeout=30000)

    out_png = tmp_path / "out_enhanced.png"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(out_png), ""),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *a, **k: QMessageBox.StandardButton.Ok,
    )
    e.btn_export.click()

    assert out_png.exists()
    json_path = out_png.with_suffix(".json")
    assert json_path.exists()
    text = json_path.read_text(encoding="utf-8")
    record = json.loads(text)
    assert record.get("parent_sha256")
    assert record.get("derived") is True
    assert re.search(r"[A-Za-z]:\\", text) is None
    assert str(tmp_path) not in text
    assert str(tmp_path.resolve()) not in text


def test_export_same_path_as_source_writes_nothing(qtbot, window, valid_png, monkeypatch):
    e = _enhance(window)
    before = valid_png.read_bytes()
    e.load_path(valid_png)
    e.btn_run.click()
    qtbot.waitUntil(lambda: e.btn_export.isVisible(), timeout=30000)

    warned = []

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(valid_png), ""),
    )

    def record_warn(*a, **k):
        warned.append(True)
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "warning", record_warn)
    e.btn_export.click()

    assert warned
    assert valid_png.read_bytes() == before


def test_enhance_run_does_not_call_analysis_service(
    qtbot, window, valid_png, monkeypatch,
):
    e = _enhance(window)
    svc = window.analyze._svc
    spy = MagicMock(wraps=svc.analyze)
    monkeypatch.setattr(svc, "analyze", spy)

    e.load_path(valid_png)
    e.btn_run.click()
    qtbot.waitUntil(lambda: e.btn_export.isVisible(), timeout=30000)

    spy.assert_not_called()
