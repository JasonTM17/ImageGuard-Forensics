"""Tests for UI stale view state regressions."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")

if TYPE_CHECKING:
    from pathlib import Path


def test_canvas_scene_rect_after_set_image(qtbot) -> None:
    from imageguard.ui.widgets.image_canvas import ImageCanvas

    canvas = ImageCanvas()
    qtbot.addWidget(canvas)

    img = np.zeros((600, 800, 3), dtype=np.uint8)
    canvas.set_image(img)

    rect = canvas.sceneRect()
    assert rect.width() >= 800
    assert rect.height() >= 600

    canvas.clear_image()
    assert canvas._base is None


def test_history_detail_non_image_clears_base(qtbot) -> None:
    from imageguard.ui.screens.history import AnalysisDetailDialog

    class DummyStore:
        def read(self, rel: str) -> bytes:
            if rel.endswith(".png"):
                arr = np.zeros((100, 100, 3), dtype=np.uint8)
                return cv2.imencode(".png", arr)[1].tobytes()
            return b'{"key": "value"}'

    class DummyHistoryService:
        def __init__(self) -> None:
            self.store = DummyStore()

    svc = DummyHistoryService()
    detail = {
        "id": "test-id",
        "image_name": "sample.png",
        "created_at": "2026-11-06T00:00:00",
        "verdict": "suspicious",
        "methods": [],
        "duration_ms": 15,
        "config_hash": "cfg123",
        "artifacts": [
            {"kind": "mask", "method": "ela", "rel_path": "art/mask.png"},
            {"kind": "report", "method": "", "rel_path": "art/report.json"},
        ],
    }
    dlg = AnalysisDetailDialog(detail, svc)
    qtbot.addWidget(dlg)

    assert dlg.combo.count() == 3

    dlg.combo.setCurrentIndex(1)
    assert dlg.canvas._base is not None

    dlg.combo.setCurrentIndex(2)
    assert dlg.canvas._base is None


def test_analyze_overlay_mismatch_and_clear(qtbot, tmp_path: Path) -> None:
    from imageguard.ui.app import build_services
    from imageguard.ui.screens.analyze import AnalyzeScreen

    analysis_svc, _ = build_services(
        db_path=tmp_path / "t.db",
        artifacts_root=tmp_path / "art",
    )
    screen = AnalyzeScreen(analysis_svc)
    qtbot.addWidget(screen)

    img_path = tmp_path / "test_100x100.png"
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path.write_bytes(cv2.imencode(".png", arr)[1].tobytes())

    screen.load_path(img_path)
    assert screen._doc is not None
    assert screen._doc.array.shape[:2] == (100, 100)

    m_valid = SimpleNamespace(mask=np.zeros((100, 100), dtype=np.uint8))
    m_bad = SimpleNamespace(mask=np.zeros((50, 50), dtype=np.uint8))

    screen.overlay_box.addItem("Mask hợp lệ", ("mask", m_valid))
    screen.overlay_box.addItem("Mask lệch kích thước", ("mask", m_bad))

    # 1. Valid choice sets overlay
    screen.overlay_box.setCurrentIndex(1)
    assert screen.canvas._overlay is not None

    # 2. Bad choice clears overlay and sets lbl_err
    screen.overlay_box.setCurrentIndex(2)
    assert screen.canvas._overlay is None
    assert screen.lbl_err.text() != ""

    # 3. "Không overlay" clears lbl_err
    screen.overlay_box.setCurrentIndex(0)
    assert screen.canvas._overlay is None
    assert screen.lbl_err.text() == ""


def test_analyze_load_path_error_clears_doc(qtbot, tmp_path: Path) -> None:
    from imageguard.ui.app import build_services
    from imageguard.ui.screens.analyze import AnalyzeScreen

    analysis_svc, _ = build_services(
        db_path=tmp_path / "t.db",
        artifacts_root=tmp_path / "art",
    )
    screen = AnalyzeScreen(analysis_svc)
    qtbot.addWidget(screen)

    valid_png = tmp_path / "valid.png"
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    valid_png.write_bytes(cv2.imencode(".png", arr)[1].tobytes())

    screen.load_path(valid_png)
    assert screen._doc is not None
    assert screen._path == valid_png

    garbage_png = tmp_path / "corrupt.png"
    garbage_png.write_bytes(b"not a valid image content")

    screen.load_path(garbage_png)
    assert screen._path is None
    assert screen._doc is None
