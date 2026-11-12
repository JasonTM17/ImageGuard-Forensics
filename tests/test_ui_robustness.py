"""Offscreen UI robustness — hostile/empty inputs across tabs and widgets."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QWidget

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ForensicReport

_LONG = "z" * 520
_UNICODE_RTL = "\u202e\u0627\u0644\u0639\u0631\u0628\u064a\u0629_\u1ea3nh_" + "ñ" * 40
# Đường dẫn pytest-of-Admin dài — giữ leaf ngắn trên đĩa; chuỗi 500+ qua setText/lỗi.
_DISK_LEAF = "ảnh_" + ("x" * 80) + ".png"


@pytest.fixture()
def services(tmp_path: Path):
    from imageguard.ui.app import build_services

    return build_services(db_path=tmp_path / "t.db", artifacts_root=tmp_path / "art")


def _png_path(tmp_path: Path, name: str, shape: tuple[int, ...]) -> Path:
    arr = np.random.default_rng(3).integers(0, 255, shape, dtype=np.uint8)
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(cv2.imencode(".png", arr)[1].tobytes())
    return p


def _assert_word_wrap_on(widget: QWidget, *attrs: str) -> None:
    for name in attrs:
        lbl = getattr(widget, name)
        assert isinstance(lbl, QLabel)
        assert lbl.wordWrap() is True


def test_all_tabs_empty_history_and_stats(qtbot, services) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    w.show()
    for i in range(w.tabs.count()):
        w.tabs.setCurrentIndex(i)
        qtbot.wait(30)
    assert w.history.table.rowCount() == 0
    assert w.stats.table.rowCount() == 0
    assert not w.history.lbl_empty.isHidden()
    _assert_word_wrap_on(w.history, "lbl_status")


def test_analyze_long_unicode_filename(qtbot, services, tmp_path: Path) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, _ = services
    w = MainWindow(analysis, _)
    qtbot.addWidget(w)
    img = _png_path(tmp_path, _DISK_LEAF, (96, 96, 3))
    w.analyze.load_path(img)
    assert w.analyze.btn_run.isEnabled()
    _assert_word_wrap_on(w.analyze, "lbl_file", "lbl_meta")
    hostile = f"{_LONG}{_UNICODE_RTL}\u202e.png"
    w.analyze.lbl_file.setText(hostile)
    qtbot.wait(20)
    assert w.analyze.lbl_file.wordWrap() is True
    assert len(w.analyze.lbl_file.text()) >= len(_LONG)


def test_analyze_long_error_strings(qtbot, services, tmp_path: Path) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, _ = services
    w = MainWindow(analysis, _)
    qtbot.addWidget(w)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not-image")
    w.analyze.load_path(bad)
    assert not w.analyze.btn_run.isEnabled()
    _assert_word_wrap_on(w.analyze, "lbl_err")
    w.analyze._on_failed(f"{_LONG} failure detail")
    assert w.analyze._worker is None
    assert len(w.analyze.lbl_err.text()) >= len(_LONG)


def test_analyze_one_pixel_image(qtbot, services, tmp_path: Path) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, _ = services
    w = MainWindow(analysis, _)
    qtbot.addWidget(w)
    tiny = _png_path(tmp_path, "1x1.png", (1, 1, 3))
    w.analyze.load_path(tiny)
    assert w.analyze._path is None
    assert not w.analyze.btn_run.isEnabled()
    assert w.analyze.lbl_err.text() != ""
    assert w.analyze.canvas._base is None


def test_analyze_zero_byte_file_does_not_crash(qtbot, services, tmp_path: Path) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, _ = services
    w = MainWindow(analysis, _)
    qtbot.addWidget(w)
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    w.analyze.load_path(empty)
    assert not w.analyze.btn_run.isEnabled()


def test_method_card_long_limits_and_empty_evidence(qtbot) -> None:
    from imageguard.ui.widgets.verdict_panel import MethodCard

    lims = [_LONG[:200] + f" #{i}" for i in range(6)]
    result = AlgorithmResult(
        method=MethodName.ELA,
        verdict=Verdict.INCONCLUSIVE,
        score=0.0,
        score_definition="test metric",
        evidence={},
        limitations=lims,
    )
    card = MethodCard(result)
    qtbot.addWidget(card)
    card.show()
    labels = card.findChildren(QLabel)
    assert labels
    wrapped = [lbl for lbl in labels if lbl.wordWrap()]
    assert len(wrapped) >= 2
    assert any(_LONG[:80] in lbl.text() for lbl in labels)


def test_analyze_on_result_missing_mask_heatmap_evidence(qtbot, services, tmp_path: Path) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, _ = services
    w = MainWindow(analysis, _)
    qtbot.addWidget(w)
    img = _png_path(tmp_path, "hostile.png", (128, 128, 3))
    w.analyze.load_path(img)
    report = ForensicReport(
        report_id="rid-long-" + "a" * 32,
        image_sha256="x" * 64,
        image_name=_UNICODE_RTL,
        image_width=128,
        image_height=128,
        created_at="2026-11-12T00:00:00Z",
        tool_version="0.0.0",
        overall_verdict=Verdict.NO_INDICATION,
        methods=[
            AlgorithmResult(
                method=MethodName.ELA,
                verdict=Verdict.NO_INDICATION,
                evidence={"pairs_image": None},
            ),
            AlgorithmResult(
                method=MethodName.NOISE_INCONSISTENCY,
                verdict=Verdict.INCONCLUSIVE,
                limitations=[_LONG],
            ),
        ],
    )
    w.analyze._on_result(report)
    assert not w.analyze.banner.isHidden()
    assert w.analyze.overlay_box.count() == 1  # chỉ "Không overlay"
    w.analyze.overlay_box.setCurrentIndex(0)
    assert w.analyze.canvas._overlay is None


def test_analyze_overlay_combo_mask_none_no_crash(qtbot, services, tmp_path: Path) -> None:
    from imageguard.ui.screens.analyze import AnalyzeScreen

    analysis, _ = services
    screen = AnalyzeScreen(analysis)
    qtbot.addWidget(screen)
    img = _png_path(tmp_path, "ov.png", (8, 8, 3))
    screen.load_path(img)
    stub = SimpleNamespace(mask=None, heatmap=None, evidence={})
    screen.overlay_box.blockSignals(True)
    screen.overlay_box.addItem("hostile mask None", ("mask", stub))
    screen.overlay_box.blockSignals(False)
    screen.overlay_box.setCurrentIndex(1)
    assert screen.canvas._overlay is None


def test_canvas_set_image_extreme_sizes(qtbot) -> None:
    from imageguard.ui.widgets.image_canvas import ImageCanvas

    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    one = np.zeros((1, 1, 3), dtype=np.uint8)
    canvas.set_image(one)
    assert canvas._base is not None
    canvas.clear_image()
    assert canvas._base is None


def test_rapid_cancel_clicks_no_crash(qtbot, services, valid_png: Path) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, _ = services
    w = MainWindow(analysis, _)
    qtbot.addWidget(w)
    w.analyze.load_path(valid_png)
    w.analyze._set_running(True)
    w.analyze._worker = SimpleNamespace(cancel=lambda: None)
    for _ in range(25):
        w.analyze.btn_cancel.click()
    assert w.analyze._worker is not None


def test_tab_switch_while_analysis_running(qtbot, services, valid_png: Path) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, _ = services
    w = MainWindow(analysis, _)
    qtbot.addWidget(w)
    a = w.analyze
    a.chk_kp.setChecked(False)
    a.chk_bl.setChecked(False)
    a.chk_noise.setChecked(False)
    a.chk_nloc.setChecked(False)
    a.chk_freq.setChecked(False)
    a.chk_spl.setChecked(False)
    a.chk_ela.setChecked(True)
    a.load_path(valid_png)
    a.btn_run.click()
    qtbot.waitUntil(lambda: a._worker is not None, timeout=5000)
    for idx in (1, 2, 3, 4, 0):
        w.tabs.setCurrentIndex(idx)
        qtbot.wait(50)
    qtbot.waitUntil(lambda: a._worker is None, timeout=120000)
    assert not a.btn_cancel.isEnabled() or a.progress.value() >= 0


def test_history_long_unicode_rows(qtbot, services) -> None:
    from imageguard.ui.screens.history import HistoryScreen

    long_img = _LONG + _UNICODE_RTL

    class _Stub:
        def list(self, limit: int = 100):
            return [{
                "id": "id-" + "b" * 40,
                "created_at": "2026-11-12 08:00:00.000000",
                "verdict": "inconclusive",
                "image_name": long_img,
                "n_methods": 7,
            }]

    screen = HistoryScreen(_Stub())  # type: ignore[arg-type]
    qtbot.addWidget(screen)
    screen.resize(640, 480)
    screen.show()
    qtbot.wait(50)
    item = screen.table.item(0, 1)
    assert item is not None
    assert item.text() == long_img
    assert item.toolTip() == long_img


def test_stats_empty_verdict_map(qtbot, services) -> None:
    from imageguard.ui.screens.stats import StatsScreen

    class _EmptyStats:
        def stats(self) -> dict:
            return {
                "n_analyses": 0,
                "avg_duration_ms": 0,
                "label": "",
                "by_verdict": {},
            }

    _, history = services
    _ = history
    screen = StatsScreen(_EmptyStats())  # type: ignore[arg-type]
    qtbot.addWidget(screen)
    screen.refresh()
    assert screen.table.rowCount() == 0
    assert "0 phân tích" in screen.lbl_head.text()


def test_enhance_long_filename_and_bad_file(qtbot, tmp_path: Path) -> None:
    from imageguard.ui.screens.enhance import EnhanceScreen

    e = EnhanceScreen()
    qtbot.addWidget(e)
    path = _png_path(tmp_path, _DISK_LEAF, (64, 64, 3))
    e.load_path(path)
    assert e.btn_run.isEnabled()
    _assert_word_wrap_on(e, "lbl_file", "lbl_meta")
    e.lbl_file.setText(_LONG + _UNICODE_RTL)

    bad = tmp_path / "bad_enhance.png"
    bad.write_bytes(b"xx")
    e.load_path(bad)
    assert e.lbl_err.text() != ""
    _assert_word_wrap_on(e, "lbl_err")


def test_verdict_banner_very_long_signals(qtbot) -> None:
    from imageguard.ui.widgets.verdict_panel import VerdictBanner

    banner = VerdictBanner()
    qtbot.addWidget(banner)
    signals = [_LONG[i : i + 80] for i in range(0, 400, 80)]
    banner.set_verdict(Verdict.SUSPICIOUS_INDICATION, signals)
    _assert_word_wrap_on(banner, "_sub")
    banner.show()


def test_cooperative_cancel_worker_survives_rapid_cancel(qtbot, services, tmp_path: Path) -> None:
    from imageguard.services.analysis_service import AnalysisRequest
    from imageguard.ui.workers import AnalysisWorker

    analysis, _ = services
    img = _png_path(tmp_path, "cancel.png", (128, 128, 3))
    cancel = threading.Event()
    req = AnalysisRequest(path=img, methods=["ela"], cancel=cancel)
    worker = AnalysisWorker(analysis, req)
    for _ in range(10):
        worker.cancel()
    assert cancel.is_set()
