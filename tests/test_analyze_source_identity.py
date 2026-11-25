"""Analyze-screen source identity — report chỉ được hiển thị khi khớp ảnh preview.

Bug gốc: preview ảnh A; file cùng path bị thay bằng B; service trả report B
(``image_sha256`` của B); ``_on_result`` hiển thị report B trên ``_doc``/canvas
A — check geometry không phát hiện thay pixel cùng kích thước. Invariant:
verdict/overlay chỉ hiển thị khi ``report.image_sha256`` khớp ``_doc.sha256``.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")

from imageguard.core.errors import ErrorCode, ErrorResult
from imageguard.services.analysis_service import AnalysisRequest
from imageguard.ui import strings


@pytest.fixture()
def services(tmp_path: Path):
    from imageguard.ui.app import build_services

    return build_services(db_path=tmp_path / "t.db", artifacts_root=tmp_path / "art")


def _write_png(path: Path, hw: tuple[int, int], value: int) -> Path:
    arr = np.full((hw[0], hw[1], 3), value, dtype=np.uint8)
    path.write_bytes(cv2.imencode(".png", arr)[1].tobytes())
    return path


def _analyze_now(screen):
    """Phân tích lại file trên đĩa — bắt chước service đọc source mới."""
    return screen._svc.analyze(
        AnalysisRequest(screen._path, methods=[], include_exif=False))


def _assert_rejected(screen) -> None:
    assert screen._report is None
    assert screen._doc is None
    assert screen._path is None
    assert screen.canvas._base is None
    assert screen.banner.isHidden()
    assert screen.btn_export.isHidden()
    assert not screen.btn_run.isEnabled()
    assert not screen.btn_send_enhance.isEnabled()
    assert screen.lbl_err.text() == strings.ERR_SOURCE_CHANGED_REOPEN


def test_result_rejected_same_geometry_replacement(qtbot, services, tmp_path: Path) -> None:
    from imageguard.ui.screens.analyze import AnalyzeScreen

    analysis, _ = services
    screen = AnalyzeScreen(analysis)
    qtbot.addWidget(screen)

    p = _write_png(tmp_path / "swap.png", (64, 80), 30)
    screen.load_path(p)
    assert screen._doc is not None

    _write_png(p, (64, 80), 220)  # thay pixel, cùng geometry, cùng path
    report = _analyze_now(screen)
    assert report.image_sha256 != screen._doc.sha256

    screen._on_result(report)
    _assert_rejected(screen)


def test_result_rejected_different_geometry_replacement(qtbot, services, tmp_path: Path) -> None:
    from imageguard.ui.screens.analyze import AnalyzeScreen

    analysis, _ = services
    screen = AnalyzeScreen(analysis)
    qtbot.addWidget(screen)

    p = _write_png(tmp_path / "swap2.png", (64, 80), 30)
    screen.load_path(p)
    assert screen._doc is not None

    _write_png(p, (96, 72), 200)  # thay cả geometry lẫn pixel
    report = _analyze_now(screen)
    assert report.image_sha256 != screen._doc.sha256

    screen._on_result(report)
    _assert_rejected(screen)


def test_reload_after_rejection_recovers(qtbot, services, tmp_path: Path) -> None:
    from imageguard.ui.screens.analyze import AnalyzeScreen

    analysis, _ = services
    screen = AnalyzeScreen(analysis)
    qtbot.addWidget(screen)

    p = _write_png(tmp_path / "swap3.png", (64, 80), 30)
    screen.load_path(p)
    _write_png(p, (64, 80), 220)
    screen._on_result(_analyze_now(screen))
    _assert_rejected(screen)

    screen.load_path(p)  # mở lại — doc giờ là B trên đĩa
    assert screen._doc is not None
    assert screen.btn_run.isEnabled()
    assert screen.btn_send_enhance.isEnabled()
    report2 = _analyze_now(screen)
    screen._on_result(report2)
    assert screen._report is report2
    assert screen._doc is not None
    assert screen._path == p
    assert not screen.banner.isHidden()


def test_unchanged_source_happy_flow(qtbot, services, tmp_path: Path) -> None:
    from imageguard.ui.screens.analyze import AnalyzeScreen

    analysis, _ = services
    screen = AnalyzeScreen(analysis)
    qtbot.addWidget(screen)

    p = _write_png(tmp_path / "same.png", (64, 80), 30)
    screen.load_path(p)
    for chk in (screen.chk_kp, screen.chk_bl, screen.chk_noise, screen.chk_nloc,
                screen.chk_freq, screen.chk_spl, screen.chk_df):
        chk.setChecked(False)
    screen.chk_ela.setChecked(True)
    screen.btn_run.click()
    qtbot.waitUntil(lambda: screen._worker is None, timeout=30_000)
    assert screen._report is not None
    assert screen._doc is not None
    assert screen._report.image_sha256 == screen._doc.sha256
    assert not screen.banner.isHidden()
    assert not screen.btn_export.isHidden()


def test_deepfake_card_explains_center_only_analysis(qtbot, services, tmp_path, monkeypatch) -> None:
    from PySide6.QtWidgets import QLabel

    from imageguard.ingest.session_reference import SessionReferenceStore
    from imageguard.ui.screens.analyze import AnalyzeScreen
    from imageguard.ui.widgets.verdict_panel import MethodCard

    monkeypatch.setattr(
        "imageguard.ui.screens.analyze.SessionReferenceStore",
        lambda: SessionReferenceStore(tmp_path / "reference"),
    )
    analysis, _ = services
    screen = AnalyzeScreen(analysis)
    qtbot.addWidget(screen)
    screen.load_path(_write_png(tmp_path / "wide.png", (192, 448), 120))
    screen._apply_method_preset(["deepfake"])
    screen.btn_run.click()
    qtbot.waitUntil(lambda: screen._worker is None, timeout=30_000)

    assert screen._report is not None
    labels = [
        label.text().lower()
        for card in screen.cards_inner.findChildren(MethodCard)
        for label in card.findChildren(QLabel)
    ]
    assert any("chưa định vị được vùng mặt" in text and "vùng trung tâm" in text for text in labels)


def test_error_result_keeps_source_for_retry(qtbot, services, tmp_path: Path) -> None:
    from imageguard.ui.screens.analyze import AnalyzeScreen

    analysis, _ = services
    screen = AnalyzeScreen(analysis)
    qtbot.addWidget(screen)

    p = _write_png(tmp_path / "ok.png", (64, 80), 30)
    screen.load_path(p)
    screen._on_result(ErrorResult(ErrorCode.DECODE_FAILED, "boom"))
    assert screen._report is None
    assert screen._doc is not None  # nguồn giữ nguyên — user retry được
    assert screen._path == p
    assert screen.btn_run.isEnabled()
    assert "DECODE_FAILED" in screen.lbl_err.text()


def test_load_path_error_redacts_host_path(qtbot, services, tmp_path: Path, monkeypatch) -> None:
    from imageguard.core import io as core_io
    from imageguard.ui.screens.analyze import AnalyzeScreen

    analysis, _ = services
    screen = AnalyzeScreen(analysis)
    qtbot.addWidget(screen)

    def fail_open(_path, _cfg=None):
        return ErrorResult(
            ErrorCode.FILE_UNREADABLE,
            "Không đọc được file: C:/Users/Jane Doe/secret.png",
        )

    monkeypatch.setattr(core_io, "safe_open", fail_open)
    screen.load_path(tmp_path / "input.png")
    assert "secret.png" in screen.lbl_err.text()
    assert "Jane Doe" not in screen.lbl_err.text()
    assert "C:/Users" not in screen.lbl_err.text()
