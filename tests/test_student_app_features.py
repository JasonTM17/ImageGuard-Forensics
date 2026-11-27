"""Tests cho các tính năng hoàn thiện app đồ án sinh viên: Demo Samples, IPKit Lab, Presets & Metrics."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.io import safe_open
from imageguard.core.models import AlgorithmResult
from imageguard.enhancement.derive import derive
from imageguard.ui.widgets.camera_dialog import (
    CameraCaptureDialog,
    analyze_camera_frame_ipkit,
)
from imageguard.ui.widgets.sample_dialog import (
    SampleGeneratorDialog,
    generate_demo_sample,
)
from imageguard.ui.widgets.verdict_panel import MethodCard


@pytest.fixture()
def services(tmp_path: Path):
    from imageguard.ui.app import build_services

    return build_services(db_path=tmp_path / "t.db", artifacts_root=tmp_path / "art")


@pytest.mark.parametrize("kind", ["copymove", "splicing", "deepfake", "clean"])
def test_generate_demo_samples_valid_png(tmp_path: Path, kind: str) -> None:
    path, methods = generate_demo_sample(kind, out_dir=tmp_path)
    assert path.is_file()
    assert len(methods) >= 3
    doc = safe_open(path, load_thresholds())
    assert doc.width >= 256 and doc.height >= 256


def test_sample_generator_dialog_flow(qtbot, tmp_path: Path) -> None:
    dlg = SampleGeneratorDialog(out_dir=tmp_path)
    qtbot.addWidget(dlg)
    dlg.rad_deepfake.setChecked(True)
    dlg.create_now()
    assert dlg.generated_path is not None and dlg.generated_path.is_file()
    assert "deepfake" in dlg.recommended_methods


def test_camera_enhance_operators_and_no_lab_clutter(qtbot, services) -> None:
    from imageguard.ui.main_window import MainWindow

    rng = np.random.default_rng(11)
    single = rng.integers(25, 220, (96, 128, 3), dtype=np.uint8)
    out_single, meta_single = derive(
        single, "cam_calibrate", {"gamma": 0.85, "clip_limit": 0.04, "boost_k": 1.3}
    )
    assert out_single.dtype == np.uint8 and out_single.shape == single.shape
    assert meta_single["dual_matched"] is False

    dual = np.hstack(
        [
            rng.integers(15, 95, (128, 160, 3), dtype=np.uint8),
            rng.integers(120, 235, (128, 160, 3), dtype=np.uint8),
        ]
    )
    out_dual, meta_dual = derive(
        dual, "cam_calibrate", {"gamma": 0.85, "clip_limit": 0.04, "boost_k": 1.3}
    )
    assert out_dual.dtype == np.uint8 and out_dual.shape == dual.shape
    assert meta_dual["dual_matched"] is True

    out_freq, meta_freq = derive(
        single, "freq_sharpen", {"d0": 35, "strength": 1.4, "median_k": 3}
    )
    assert out_freq.dtype == np.uint8 and out_freq.shape == single.shape
    assert meta_freq["kind"] == "freq_sharpen"

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    assert not hasattr(w.analyze, "btn_ipkit_lab")
    assert not hasattr(w.enhance, "btn_ipkit_lab")
    assert w.enhance.combo_op.findData("cam_calibrate") >= 0
    assert w.enhance.combo_op.findData("freq_sharpen") >= 0


def test_camera_frame_ipkit_diagnostics_single_and_dual() -> None:
    rng = np.random.default_rng(21)
    single = rng.integers(30, 220, (120, 160, 3), dtype=np.uint8)
    s_txt = analyze_camera_frame_ipkit(single, is_dual=False)
    assert "Chẩn đoán cảm biến:" in s_txt and "Ngưỡng tách nền:" in s_txt

    dual = np.hstack(
        [
            rng.integers(20, 120, (120, 160, 3), dtype=np.uint8),
            rng.integers(100, 235, (120, 160, 3), dtype=np.uint8),
        ]
    )
    d_txt = analyze_camera_frame_ipkit(dual, is_dual=True)
    assert "Ảnh ghép 2 Camera" in d_txt and "sau hiệu chuẩn:" in d_txt


def test_analyze_presets_and_send_to_enhance(qtbot, services, valid_png: Path) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    a = w.analyze
    assert not a.btn_send_enhance.isEnabled()

    a.btn_preset_all.click()
    assert len(a._methods()) == 9

    a.btn_preset_camera.click()
    assert set(a._methods()) == {"ela", "noise_local", "freq", "deepfake", "deepfake_ml"}
    a._set_running(True)
    assert not a.chk_dfml.isEnabled()
    a._set_running(False)
    assert a.chk_dfml.isEnabled()

    a.btn_preset_default.click()
    assert set(a._methods()) == {"keypoint", "block", "ela"}

    a.load_path(valid_png)
    assert a.btn_send_enhance.isEnabled()
    a.btn_send_enhance.click()

    assert w.tabs.currentWidget() is w.enhance
    assert w.enhance._path == valid_png
    assert w.enhance.combo_op.currentData() == "super_res"


def test_enhance_displays_quality_metrics(qtbot, services, valid_png: Path) -> None:
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    e = w.enhance
    e.load_path(valid_png)
    e._cancel_event.clear()
    derived = np.clip(e._doc.array.astype(np.int16) + 5, 0, 255).astype(np.uint8)
    e._on_result((derived, {}))

    assert not e.lbl_metrics.isHidden()
    txt = e.lbl_metrics.text()
    assert "PSNR=" in txt
    assert "SSIM=" in txt
    assert "Laplacian" in txt


def test_camera_dialog_probe_ports_updates_usb_spin(qtbot, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "imageguard.ui.widgets.camera_dialog.probe_camera_indices",
        lambda max_index=4: [0, 2],
    )
    dlg = CameraCaptureDialog(out_dir=tmp_path)
    qtbot.addWidget(dlg)
    dlg.probe_ports()
    qtbot.waitUntil(lambda: dlg._worker is None)
    assert dlg.spin_usb.value() == 2
    assert "[0, 2]" in dlg.lbl_probe_info.text()


def test_method_card_shows_duration_and_evidence(qtbot) -> None:
    res = AlgorithmResult(
        method=MethodName.DEEPFAKE_CUES,
        verdict=Verdict.NO_INDICATION,
        score=0.22,
        score_definition="4 trụ cột DIP",
        evidence={"variant": "dip_4pillar_deepfake_cues", "roi_source": "skin_haar_dip", "lbp_chi2": 0.0123},
        limitations=["signal khám phá"],
        duration_ms=42,
    )
    card = MethodCard(res)
    qtbot.addWidget(card)
    from PySide6.QtWidgets import QLabel

    blob = "\n".join(lbl.text() for lbl in card.findChildren(QLabel))
    assert "42 ms" in blob
    assert "roi_source=skin_haar_dip" in blob
    assert "lbp_chi2=0.0123" in blob
