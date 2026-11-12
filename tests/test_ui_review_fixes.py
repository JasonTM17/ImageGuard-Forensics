"""Offscreen tests for UX review fixes (report D:\\ig-test\\ag\\report.md)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialogButtonBox, QMessageBox, QScrollArea

from imageguard.ui import strings


@pytest.fixture()
def services(tmp_path: Path):
    from imageguard.ui.app import build_services

    return build_services(db_path=tmp_path / "t.db", artifacts_root=tmp_path / "art")


def test_canvas_keyboard_zoom_and_reset(qtbot, services, valid_png: Path) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    canvas = w.analyze.canvas
    w.analyze.load_path(valid_png)
    qtbot.waitUntil(lambda: canvas._base is not None, timeout=5000)
    canvas.setFocus()
    before = canvas.transform().m11()
    qtbot.keyClick(canvas, Qt.Key.Key_Plus)
    qtbot.waitUntil(lambda: canvas.transform().m11() > before, timeout=2000)
    after_zoom = canvas.transform().m11()
    qtbot.keyClick(canvas, Qt.Key.Key_Minus)
    qtbot.waitUntil(lambda: canvas.transform().m11() < after_zoom, timeout=2000)
    fitted = canvas.transform().m11()
    canvas.scale(3.0, 3.0)
    assert canvas.transform().m11() > fitted
    qtbot.keyClick(canvas, Qt.Key.Key_Home)
    qtbot.waitUntil(
        lambda: abs(canvas.transform().m11() - fitted) < 0.05,
        timeout=2000,
    )
    assert strings.A11Y_CANVAS_DESC in canvas.accessibleDescription()


def test_enhance_export_sidecar_only_overwrite_message(
    qtbot, services, valid_png: Path, tmp_path: Path,
) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    e = w.enhance
    e.load_path(valid_png)
    e._derived = e._doc.array.copy()
    e._run_kind = "unsharp"
    e._run_params = {"ksize": 3, "sigma": 1.0, "amount": 1.0}
    dest = tmp_path / "out.png"
    sidecar = dest.with_suffix(".json")
    sidecar.write_text("{}", encoding="utf-8")
    captured: list[str] = []

    def fake_question(parent, title, text):
        captured.append(text)
        return QMessageBox.StandardButton.Yes

    with patch(
        "imageguard.ui.screens.enhance.QMessageBox.information",
        return_value=QMessageBox.StandardButton.Ok,
    ), patch(
        "imageguard.ui.screens.enhance.QMessageBox.question",
        side_effect=fake_question,
    ), patch(
        "imageguard.ui.screens.enhance.QFileDialog.getSaveFileName",
        return_value=(str(dest), ""),
    ), patch(
        "imageguard.enhancement.derive.write_sibling",
        return_value=sidecar,
    ):
        e._export()

    assert captured
    assert "siêu dữ liệu" in captured[0]
    assert strings.MSG_EXPORT_OVERWRITE_SIDECAR.split("\n")[0] in captured[0]


def test_analyze_cancel_status_feedback(
    qtbot, services, valid_png: Path, monkeypatch,
) -> None:
    from imageguard.services import analysis_service as svc_mod
    from imageguard.ui.main_window import MainWindow

    real_analyze = svc_mod.AnalysisService.analyze

    def slow_analyze(self, req):
        for _ in range(80):
            if req.cancel is not None and req.cancel.is_set():
                break
            time.sleep(0.05)
        return real_analyze(self, req)

    monkeypatch.setattr(svc_mod.AnalysisService, "analyze", slow_analyze)

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    a = w.analyze
    a.load_path(valid_png)
    for c in (a.chk_kp, a.chk_bl, a.chk_noise, a.chk_nloc, a.chk_freq, a.chk_spl):
        c.setChecked(False)
    a.chk_ela.setChecked(True)
    a.btn_run.click()
    qtbot.waitUntil(lambda: a._worker is not None, timeout=5000)
    a.btn_cancel.click()
    assert a.lbl_status.text() == strings.STATUS_ANALYZE_CANCELLING
    qtbot.waitUntil(
        lambda: a._worker is None and not a.btn_cancel.isEnabled(),
        timeout=30000,
    )
    assert a.lbl_status.text() == strings.STATUS_ANALYZE_CANCELLED


def test_enhance_right_panel_scroll_area(qtbot, services) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    scrolls = w.enhance.findChildren(QScrollArea)
    assert any(sa.widget() is not None and sa.widgetResizable() for sa in scrolls)


def test_analyze_cards_minimum_height(qtbot, services) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    assert w.analyze.cards_area.minimumHeight() >= 180


def test_history_detail_dialog_close_and_labels(qtbot, services) -> None:
    from imageguard.ui.screens.history import AnalysisDetailDialog

    detail = {
        "id": "r1",
        "image_name": "a.png",
        "created_at": "2026-11-12T08:00:00",
        "verdict": "no_indication",
        "duration_ms": 12,
        "config_hash": "abc",
        "methods": [{
            "method": "copy_move_keypoint",
            "verdict": "no_indication",
            "score": 0.1,
            "score_definition": "demo",
        }],
        "artifacts": [{"kind": "mask", "method": "copy_move_keypoint", "rel_path": "x.png"}],
    }
    dlg = AnalysisDetailDialog(detail, SimpleNamespace(store=SimpleNamespace()), None)
    qtbot.addWidget(dlg)
    scrolls = dlg.findChildren(QScrollArea)
    assert scrolls
    box = dlg.findChild(QDialogButtonBox)
    assert box is not None
    close_btn = box.button(QDialogButtonBox.StandardButton.Close)
    assert close_btn is not None
    assert close_btn.text() == strings.BTN_CLOSE
    assert strings.METHOD_LABEL["copy_move_keypoint"] in dlg.combo.itemText(1)


def test_history_action_buttons_disabled_without_selection(qtbot, services) -> None:
    from imageguard.ui.screens.history import HistoryScreen

    _, history = services
    h = HistoryScreen(history)
    qtbot.addWidget(h)
    assert not h.btn_reopen.isEnabled()
    assert not h.btn_export.isEnabled()
    assert not h.btn_delete.isEnabled()


def test_strings_terminology_consistency() -> None:
    assert strings.OVERLAY_NONE == "Không dùng lớp phủ"
    assert strings.SETTINGS_TREE_HDR_KEY == "Khóa"
    assert "verdict" not in strings.A11Y_STATS_TABLE.lower()


def test_enhance_spinbox_accessible_names(qtbot, services) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    e = w.enhance
    _ksize = e._fields[("unsharp", "ksize")][1]
    assert "kernel" in _ksize.accessibleName().lower()


def test_stats_empty_label_visible(qtbot, services) -> None:
    from imageguard.ui.screens.stats import StatsScreen

    _, history = services
    s = StatsScreen(history)
    qtbot.addWidget(s)
    s.refresh()
    assert not s.lbl_empty.isHidden()
