"""Two-camera layout is explicit and survives saving; wide photos remain single."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from imageguard.core.io import CAPTURE_LAYOUT_KEY, safe_open
from imageguard.core.models import ImageDocument
from imageguard.ingest import camera


def test_dual_capture_preserves_exact_split_and_pixels_after_rename(tmp_path, monkeypatch) -> None:
    left = np.full((80, 100, 3), (10, 20, 30), dtype=np.uint8)
    right = np.full((80, 150, 3), (150, 160, 170), dtype=np.uint8)
    monkeypatch.setattr(
        camera, "_read_frame_from_device", lambda idx, _: [left, right][idx].copy()
    )
    path = camera.capture_dual_device_frames(0, 1, tmp_path)
    renamed = path.rename(tmp_path / "renamed.png")
    doc = safe_open(renamed)

    assert isinstance(doc, ImageDocument)
    assert doc.capture_layout == "dual_side_by_side"
    assert doc.capture_split_x == 100
    np.testing.assert_array_equal(doc.array, np.hstack([left, right]))


def test_unmarked_panorama_is_single_even_with_dual_camera_filename(tmp_path) -> None:
    frame = np.full((80, 300, 3), 110, dtype=np.uint8)
    path = tmp_path / "camera_dual_dev0_dev1.png"
    assert cv2.imwrite(str(path), frame)
    doc = safe_open(path)

    assert isinstance(doc, ImageDocument)
    assert doc.capture_layout == "single"
    assert doc.capture_split_x is None


@pytest.mark.parametrize("change", [
    {"split_x": True}, {"split_x": 0}, {"split_x": 240},
    {"width": 241}, {"version": 2}, {"layout": "unknown"},
])
def test_invalid_layout_marker_does_not_activate_dual(tmp_path: Path, change: dict) -> None:
    marker = {
        "version": 1, "layout": "dual_side_by_side", "split_x": 100,
        "width": 240, "height": 80,
    }
    marker.update(change)
    info = PngImagePlugin.PngInfo()
    info.add_text(CAPTURE_LAYOUT_KEY, json.dumps(marker))
    path = tmp_path / "marked.png"
    Image.new("RGB", (240, 80), (50, 60, 70)).save(path, pnginfo=info)
    doc = safe_open(path)

    assert isinstance(doc, ImageDocument)
    assert doc.capture_layout == "single"
    assert doc.capture_split_x is None


def test_exif_rotation_discards_stale_vertical_split(tmp_path: Path) -> None:
    marker = {
        "version": 1, "layout": "dual_side_by_side", "split_x": 100,
        "width": 240, "height": 80,
    }
    info = PngImagePlugin.PngInfo()
    info.add_text(CAPTURE_LAYOUT_KEY, json.dumps(marker))
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "rotated.png"
    Image.new("RGB", (240, 80), (50, 60, 70)).save(path, pnginfo=info, exif=exif)
    doc = safe_open(path)

    assert isinstance(doc, ImageDocument)
    assert (doc.width, doc.height) == (80, 240)
    assert doc.capture_layout == "single"
    assert doc.capture_split_x is None


def test_preview_receives_explicit_unequal_dual_layout(tmp_path, monkeypatch) -> None:
    from imageguard.ui.widgets import camera_dialog

    frame = np.full((80, 250, 3), 120, dtype=np.uint8)
    path = camera._save_lossless_png(
        frame, tmp_path, "dual", capture_layout="dual_side_by_side", capture_split_x=100
    )
    seen = {}

    def capture_format(arr, **kwargs):
        seen.update(kwargs)
        np.testing.assert_array_equal(arr, frame)
        return "explicit dual preview"

    monkeypatch.setattr(camera_dialog, "format_deepfake_preview", capture_format)
    assert camera_dialog._pillar_text_from_file(path) == "explicit dual preview"
    assert seen == {"capture_layout": "dual_side_by_side", "capture_split_x": 100}


def test_wide_camera_diagnostics_do_not_infer_two_sensors() -> None:
    from imageguard.ui.widgets.camera_dialog import analyze_camera_frame_ipkit

    frame = np.full((80, 300, 3), 120, dtype=np.uint8)
    text = analyze_camera_frame_ipkit(frame)

    assert "Chẩn đoán cảm biến:" in text
    assert "2 Camera" not in text


def test_analysis_explains_why_dual_reference_is_unavailable(qtbot, tmp_path) -> None:
    from imageguard.ingest.session_reference import SessionReferenceStore
    from imageguard.services.analysis_service import AnalysisService
    from imageguard.ui import strings
    from imageguard.ui.screens.analyze import AnalyzeScreen

    dual = camera._save_lossless_png(
        np.zeros((80, 200, 3), np.uint8), tmp_path, "dual",
        capture_layout="dual_side_by_side", capture_split_x=90,
    )
    screen = AnalyzeScreen(AnalysisService())
    screen._ref_store = SessionReferenceStore(root=tmp_path / "reference")
    qtbot.addWidget(screen)
    screen.load_path(dual)

    assert not screen.btn_set_ref.isEnabled()
    assert screen.lbl_ref.text() == strings.ANALYZE_REFERENCE_DUAL_UNSUPPORTED
