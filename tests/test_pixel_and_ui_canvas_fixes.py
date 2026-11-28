"""Targeted regression tests for pixel/channel contracts, heatmap overlays,
ImageCanvas interactions, splicing empty-ELA handling, frozen paths, DB cleanup,
and Windows non-ASCII path handling."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import (
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QKeyEvent,
    QWheelEvent,
)

import imageguard.forensics.auxiliary.splicing as splicing_mod
from imageguard import cli
from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.io import safe_open
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.core.paths import data_root
from imageguard.evaluation.robustness import _DISTORTIONS, run_battery, save_battery
from imageguard.evaluation.synthforge import generate_set as generate_synth_set
from imageguard.forensics.auxiliary.splicing import SplicingCues
from imageguard.forensics.keypoint.pipeline import KeypointCopyMove
from imageguard.forensics.localize.heatmap import (
    normalize_map,
    overlay_heatmap,
    overlay_mask,
    render_heatmap,
)
from imageguard.ingest.camera import DEFAULT_CAPTURE_DIR
from imageguard.ipkit import _util, edges
from imageguard.services.analysis_service import AnalysisRequest, AnalysisService
from imageguard.storage.artifacts import ArtifactStore
from imageguard.storage.db import make_session_factory
from imageguard.storage.models import MetaKV
from imageguard.storage.repository import AnalysisRepository, ImageRepository
from imageguard.ui.widgets.image_canvas import (
    ImageCanvas,
    heat_to_qpixmap,
    ndarray_to_qpixmap,
)
from imageguard.ui.widgets.sample_dialog import DEFAULT_DEMO_DIR


def test_apply_per_channel_preserves_float32_and_edges_3d() -> None:
    bgr = np.zeros((24, 24, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255
    bgr[20, 20, 0] = 0  # isolated dark impulse -> positive Laplacian 1020
    bgr[:, 12:, 1] = 220
    bgr[4, 4, 2] = 255  # isolated bright impulse -> negative Laplacian -1020

    lap_3d = edges.laplacian(bgr)
    assert lap_3d.dtype == np.float32
    assert lap_3d.shape == (24, 24, 3)
    assert float(lap_3d.min()) < -255.0
    assert float(lap_3d.max()) > 255.0
    for c in range(3):
        np.testing.assert_allclose(lap_3d[:, :, c], edges.laplacian(bgr[:, :, c]))

    gx3, gy3 = edges.prewitt(bgr)
    assert gx3.dtype == np.float32 and gx3.shape == (24, 24, 3)
    assert gy3.dtype == np.float32 and gy3.shape == (24, 24, 3)
    for c in range(3):
        gx_c, gy_c = edges.prewitt(bgr[:, :, c])
        np.testing.assert_allclose(gx3[:, :, c], gx_c)
        np.testing.assert_allclose(gy3[:, :, c], gy_c)

    empty_ch = np.zeros((8, 8, 0), dtype=np.uint8)
    assert _util.apply_per_channel(empty_ch, lambda ch: ch).shape == (8, 8, 0)

    with pytest.raises(ValueError, match="odd"):
        edges.convolve2d(bgr[:, :, 0], np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="odd|non-empty"):
        edges.convolve2d(bgr[:, :, 0], np.empty((0, 0), dtype=np.float32))
    with pytest.raises(ValueError, match="non-empty 2D"):
        edges.convolve2d(bgr, np.ones((3, 3), dtype=np.float32))


def test_ndarray_to_qpixmap_all_shapes_dtypes_and_heat(qtbot) -> None:
    gray = np.full((16, 20), 123, dtype=np.uint8)
    pm_gray = ndarray_to_qpixmap(gray)
    assert (pm_gray.width(), pm_gray.height()) == (20, 16)
    c_gray = pm_gray.toImage().pixelColor(10, 8)
    assert (c_gray.red(), c_gray.green(), c_gray.blue()) == (123, 123, 123)

    gray1 = gray[:, :, None]
    pm_g1 = ndarray_to_qpixmap(gray1)
    c_g1 = pm_g1.toImage().pixelColor(5, 4)
    assert (c_g1.red(), c_g1.green(), c_g1.blue()) == (123, 123, 123)

    bgr = np.zeros((16, 20, 3), dtype=np.uint8)
    bgr[:, :] = (10, 40, 200)
    pm_bgr = ndarray_to_qpixmap(bgr)
    c_bgr = pm_bgr.toImage().pixelColor(2, 3)
    assert (c_bgr.red(), c_bgr.green(), c_bgr.blue()) == (200, 40, 10)

    bgra_opaque = np.zeros((16, 20, 4), dtype=np.uint8)
    bgra_opaque[:, :] = (15, 60, 220, 255)
    pm_opaque = ndarray_to_qpixmap(bgra_opaque)
    for x, y in ((0, 0), (1, 0), (5, 7)):
        c = pm_opaque.toImage().pixelColor(x, y)
        assert (c.red(), c.green(), c.blue(), c.alpha()) == (220, 60, 15, 255)

    bgra_alpha = np.zeros((16, 20, 4), dtype=np.uint8)
    bgra_alpha[:, :] = (15, 60, 220, 128)
    pm_alpha = ndarray_to_qpixmap(bgra_alpha)
    c_alpha = pm_alpha.toImage().pixelColor(2, 3)
    assert c_alpha.alpha() == 128
    assert c_alpha.red() >= 218 and c_alpha.green() == 60 and c_alpha.blue() <= 17

    strided = bgr[::2, ::2, :]
    assert not strided.flags["C_CONTIGUOUS"]
    pm_strided = ndarray_to_qpixmap(strided)
    assert (pm_strided.width(), pm_strided.height()) == (10, 8)

    weird_f = np.array([[np.nan, np.inf], [-5.0, 300.0]], dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        pm_f = ndarray_to_qpixmap(weird_f)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    img_f = pm_f.toImage()
    assert img_f.pixelColor(0, 0).red() == 0
    assert img_f.pixelColor(1, 0).red() == 255
    assert img_f.pixelColor(0, 1).red() == 0
    assert img_f.pixelColor(1, 1).red() == 255

    assert ndarray_to_qpixmap(np.zeros((0, 0, 3), dtype=np.uint8)).isNull()
    with pytest.raises(ValueError, match="Unsupported image shape"):
        ndarray_to_qpixmap(np.zeros((8, 8, 5), dtype=np.uint8))

    pm_heat = heat_to_qpixmap(np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(6, 8), (18, 24))
    assert (pm_heat.width(), pm_heat.height()) == (24, 18)


def test_image_canvas_interactions_zoom_pan_overlay_and_drop(qtbot, tmp_path: Path) -> None:
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(240, 200)
    canvas.show()

    canvas.set_overlay(None)
    wheel_up = QWheelEvent(
        QPointF(20, 20),
        QPointF(20, 20),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    canvas.wheelEvent(wheel_up)

    bgr = np.full((64, 80, 3), 100, dtype=np.uint8)
    canvas.set_image(bgr)
    assert canvas._base is not None

    pm_ov = heat_to_qpixmap(np.ones((64, 80), dtype=np.float32), (64, 80))
    canvas.set_overlay(pm_ov, alpha=0.5)
    first_ov = canvas._overlay
    assert first_ov is not None and first_ov.opacity() == pytest.approx(0.5)
    canvas.set_overlay(pm_ov, alpha=0.25)
    assert canvas._overlay is not None and canvas._overlay is not first_ov
    canvas.set_overlay(None)
    assert canvas._overlay is None

    m0 = canvas.transform().m11()
    canvas.wheelEvent(wheel_up)
    assert canvas.transform().m11() > m0

    for key in (
        Qt.Key.Key_Plus,
        Qt.Key.Key_Equal,
        Qt.Key.Key_Minus,
        Qt.Key.Key_Left,
        Qt.Key.Key_Right,
        Qt.Key.Key_Up,
        Qt.Key.Key_Down,
        Qt.Key.Key_Home,
        Qt.Key.Key_A,
    ):
        ev = QKeyEvent(QKeyEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
        canvas.keyPressEvent(ev)

    img_path = tmp_path / "drop_sample.png"
    img_path.write_bytes(cv2.imencode(".png", bgr)[1].tobytes())
    dropped: list[Path] = []
    canvas.image_dropped.connect(dropped.append)

    url_mime = QMimeData()
    url_mime.setUrls([QUrl.fromLocalFile(str(img_path))])
    text_mime = QMimeData()
    text_mime.setText("plain-text")

    drag_enter_ok = QDragEnterEvent(
        QPointF(10, 10), Qt.DropAction.CopyAction, url_mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    drag_enter_ok.ignore()
    drag_enter_no = QDragEnterEvent(
        QPointF(10, 10), Qt.DropAction.CopyAction, text_mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    drag_enter_no.ignore()
    canvas.dragEnterEvent(drag_enter_ok)
    canvas.dragEnterEvent(drag_enter_no)
    assert drag_enter_ok.isAccepted()

    drag_move_ok = QDragMoveEvent(
        QPointF(10, 10), Qt.DropAction.CopyAction, url_mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    drag_move_ok.ignore()
    drag_move_no = QDragMoveEvent(
        QPointF(10, 10), Qt.DropAction.CopyAction, text_mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    drag_move_no.ignore()
    canvas.dragMoveEvent(drag_move_ok)
    canvas.dragMoveEvent(drag_move_no)
    assert drag_move_ok.isAccepted()

    drop_ok = QDropEvent(
        QPointF(10, 10), Qt.DropAction.CopyAction, url_mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    canvas.dropEvent(drop_ok)
    assert dropped == [img_path]

    drop_text = QDropEvent(
        QPointF(10, 10), Qt.DropAction.CopyAction, text_mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    canvas.dropEvent(drop_text)
    assert dropped == [img_path]

    missing_mime = QMimeData()
    missing_mime.setUrls([QUrl.fromLocalFile(str(tmp_path / "missing.png"))])
    drop_missing = QDropEvent(
        QPointF(10, 10), Qt.DropAction.CopyAction, missing_mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    canvas.dropEvent(drop_missing)
    assert len(dropped) == 1

    canvas.clear_image()
    assert canvas._base is None


def test_heatmap_overlay_and_mask_multichannel_and_resize_contracts() -> None:
    bgr = np.full((32, 40, 3), 80, dtype=np.uint8)
    heat_3d1 = np.full((16, 20, 1), 0.5, dtype=np.float32)
    heat_3d1[0, 0, 0] = np.nan
    heat_3d1[1, 1, 0] = np.inf
    heat_3d3 = np.dstack([heat_3d1[:, :, 0]] * 3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        ov1 = overlay_heatmap(bgr, heat_3d1, alpha=0.5)
        ov3 = overlay_heatmap(bgr, heat_3d3, alpha=0.5)
        hm3 = render_heatmap(heat_3d3)
        ov_gray = overlay_heatmap(bgr[:, :, 0], heat_3d1[:, :, 0])
        bgra = np.dstack([bgr, np.full((32, 40), 255, dtype=np.uint8)])
        ov_bgra = overlay_heatmap(bgra, heat_3d1)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert ov1.shape == (32, 40, 3) and ov1.dtype == np.uint8
    assert ov3.shape == (32, 40, 3) and ov3.dtype == np.uint8
    assert hm3.shape == (16, 20, 3) and hm3.dtype == np.uint8
    assert ov_gray.shape == (32, 40, 3)
    assert ov_bgra.shape == (32, 40, 3)
    assert normalize_map(np.array([-1.0, np.nan, -np.inf], dtype=np.float32)).max() == 0.0

    small_mask = np.zeros((16, 20, 1), dtype=np.uint8)
    small_mask[4:12, 5:15, 0] = 255
    masked = overlay_mask(bgr, small_mask, color=(0, 0, 255), alpha=0.5)
    assert masked.shape == (32, 40, 3)
    assert masked[0, 0].tolist() == [80, 80, 80]
    assert masked[16, 20, 2] > masked[16, 20, 0]

    masked_gray = overlay_mask(bgr[:, :, 0], small_mask[:, :, 0], color=(0, 255, 0), alpha=0.5)
    assert masked_gray.shape == (32, 40, 3) and masked_gray[16, 20, 1] > masked_gray[16, 20, 0]

    with pytest.raises(ValueError, match="2D or 3D"):
        render_heatmap(np.zeros((2, 2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="image"):
        overlay_mask(np.zeros((2, 2, 2, 2), dtype=np.uint8), small_mask)


def test_analysis_service_heatmap_artifact_sanitizes_nan_and_preserves_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = load_thresholds()
    img_path = tmp_path / "mẫu_báo_cáo.png"
    img = np.random.default_rng(42).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    img_path.write_bytes(cv2.imencode(".png", img)[1].tobytes())

    heat = np.zeros((64, 64), dtype=np.float32)
    heat[0, 0] = 0.2
    heat[0, 1] = 1.5
    heat[0, 2] = -0.4
    heat[0, 3] = np.nan
    heat[0, 4] = np.inf
    heat[0, 5] = -np.inf

    class _HeatStub:
        name = MethodName.COPY_MOVE_KEYPOINT.value

        def analyze(self, doc: ImageDocument, _cfg, _cancel=None) -> AlgorithmResult:
            return AlgorithmResult(
                method=MethodName.COPY_MOVE_KEYPOINT,
                verdict=Verdict.NO_INDICATION,
                score=0.2,
                score_definition="stub heat",
                heatmap=heat,
            )

    monkeypatch.setattr(
        "imageguard.services.analysis_service.load_method",
        lambda _spec: _HeatStub(),
    )
    sf = make_session_factory(tmp_path / "db.sqlite")
    store = ArtifactStore(tmp_path / "artifacts")
    try:
        svc = AnalysisService(cfg, ImageRepository(sf), AnalysisRepository(sf), store)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            rep = svc.analyze(
                AnalysisRequest(path=img_path, methods=["keypoint"], include_exif=False)
            )
        assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
        heat_file = next((tmp_path / "artifacts" / rep.report_id).glob("*_heat.png"))
        decoded = cv2.imdecode(
            np.frombuffer(heat_file.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED
        )
        assert decoded[0, 0] == 51
        assert decoded[0, 1] == 255
        assert decoded[0, 2] == 0
        assert decoded[0, 3] == 0
        assert decoded[0, 4] == 255
        assert decoded[0, 5] == 0
    finally:
        bind = sf.kw.get("bind")
        if bind is not None:
            bind.dispose()


def test_splicing_cues_zero_ela_and_zero_overlap_emit_no_phantom_mask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = load_thresholds()
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    jpg_path = tmp_path / "clean.jpg"
    jpg_path.write_bytes(
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    )
    doc = safe_open(jpg_path, cfg)
    assert isinstance(doc, ImageDocument)

    zero_ela = np.zeros((128, 128), dtype=np.uint8)
    hot_ela = np.zeros((128, 128), dtype=np.uint8)
    hot_ela[:10, :10] = 255
    noise_mask = np.zeros((128, 128), dtype=np.uint8)
    noise_mask[40:50, 40:50] = 255

    def _stub_noise(_self, _doc, _cfg, _cancel=None):
        return AlgorithmResult(
            method=MethodName.NOISE_INCONSISTENCY,
            verdict=Verdict.NO_INDICATION,
            score=0.0,
            score_definition="noise",
            mask=noise_mask,
        )

    def _make_ela_stub(ela_mask: np.ndarray):
        def _stub_ela(_self, _doc, _cfg, _cancel=None):
            return AlgorithmResult(
                method=MethodName.ELA,
                verdict=Verdict.NO_INDICATION,
                score=0.0,
                score_definition="ela",
                mask=ela_mask,
            )

        return _stub_ela

    monkeypatch.setattr(splicing_mod.NoiseMap, "analyze", _stub_noise)
    monkeypatch.setattr(splicing_mod.ELA, "analyze", _make_ela_stub(zero_ela))
    r_zero = SplicingCues().analyze(doc, cfg)
    assert r_zero.verdict == Verdict.NO_INDICATION
    assert r_zero.score == 0.0
    assert r_zero.evidence["ela_hot_px"] == 0
    assert r_zero.evidence["overlap_px"] == 0
    assert r_zero.mask is None

    monkeypatch.setattr(splicing_mod.ELA, "analyze", _make_ela_stub(hot_ela))
    r_disjoint = SplicingCues().analyze(doc, cfg)
    assert r_disjoint.verdict == Verdict.NO_INDICATION
    assert r_disjoint.score == 0.0
    assert r_disjoint.evidence["ela_hot_px"] == 100
    assert r_disjoint.evidence["overlap_px"] == 0
    assert r_disjoint.mask is None


def test_frozen_paths_db_mismatch_cleanup_and_cli_persist_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    expected_root = data_root()
    assert DEFAULT_CAPTURE_DIR == expected_root / "captures"
    assert DEFAULT_DEMO_DIR == expected_root / "demo_samples"

    db_path = tmp_path / "mismatch.db"
    sf = make_session_factory(db_path)
    with sf() as s:
        row = s.get(MetaKV, "schema_version")
        assert row is not None
        row.value = "999"
        s.commit()
    bind = sf.kw.get("bind")
    assert bind is not None
    bind.dispose()

    with pytest.raises(RuntimeError, match="schema_version"):
        make_session_factory(db_path)
    db_path.unlink()

    custom_root = tmp_path / "vùng_dữ_liệu"
    img_path = tmp_path / "cli_sample.png"
    img = np.full((64, 64, 3), 140, dtype=np.uint8)
    img_path.write_bytes(cv2.imencode(".png", img)[1].tobytes())

    monkeypatch.setattr("imageguard.core.paths.data_root", lambda: custom_root)
    monkeypatch.setattr(
        sys, "argv", ["imageguard", "analyze", str(img_path), "--methods", "ela"]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["report_id"]
    assert (custom_root / "imageguard.db").is_file()
    assert (custom_root / "artifacts" / payload["report_id"]).is_dir()


def test_unicode_windows_paths_robustness_synthforge_and_ipkit_demo(tmp_path: Path) -> None:
    uni_dir = tmp_path / "thư_mục_tiếng_Việt_ảnh"
    manifest = generate_synth_set(
        1, seed=7, out_dir=uni_dir / "synth", size=(128, 128), tier="easy"
    )
    assert manifest.is_file()
    forged_path = uni_dir / "synth" / "easy_0000_F.png"
    assert forged_path.is_file()

    cfg = load_thresholds()
    missing_uni = uni_dir / "không_tồn_tại.png"
    res = run_battery(
        [forged_path, missing_uni],
        [KeypointCopyMove()],
        cfg,
        {"none": _DISTORTIONS["none"], "jpeg_90": _DISTORTIONS["jpeg_90"]},
    )
    assert res["copy_move_keypoint"]["none"]["n"] == 1
    assert res["copy_move_keypoint"]["jpeg_90"]["n"] == 1

    saved = save_battery(res, uni_dir / "báo_cáo" / "robustness.json")
    assert saved.is_file()
    assert json.loads(saved.read_text(encoding="utf-8"))["copy_move_keypoint"]["none"]["n"] == 1

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "ipkit_demo.py"
    spec = importlib.util.spec_from_file_location("ipkit_demo", script_path)
    assert spec is not None and spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    sample = np.full((48, 64, 3), 175, dtype=np.uint8)
    sample[20:40, 16:48] = (40, 90, 210)
    sample[28:44, 32:58] = (220, 210, 60)
    out_demo_dir = demo.validate_out_dir(uni_dir / "demo_out")
    written = demo.write_png_safe(out_demo_dir, "xuất_ảnh.png", sample)
    assert written.is_file()
    loaded = demo.load_input_image(written)
    assert loaded.shape == (48, 64, 3)
    assert int(loaded[10, 10, 0]) == 175
    with pytest.raises(ValueError, match="unsafe"):
        demo.resolve_output_file(out_demo_dir, "")
    with pytest.raises(ValueError, match="unsafe"):
        demo.resolve_output_file(out_demo_dir, ".")

    # Subprocess CLI under cp1252 stdout with Vietnamese --out and --image paths
    cli_out = uni_dir / "đầu_ra_demo"
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    proc = subprocess.run(
        [sys.executable, str(script_path), "--out", str(cli_out), "--image", str(written)],
        env=env,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    for name in demo.CHAPTER_OUTPUT_NAMES:
        assert (cli_out / name).is_file()

    # Non-ASCII path safety on scripts/make_report_figures.py helpers
    fig_script = Path(__file__).resolve().parents[1] / "scripts" / "make_report_figures.py"
    fig_spec = importlib.util.spec_from_file_location("make_report_figures_uni_test", fig_script)
    assert fig_spec is not None and fig_spec.loader is not None
    fig_mod = importlib.util.module_from_spec(fig_spec)
    sys.modules[fig_spec.name] = fig_mod
    try:
        fig_spec.loader.exec_module(fig_mod)
    finally:
        sys.modules.pop(fig_spec.name, None)

    fig_dir = uni_dir / "hình_báo_cáo"
    fig_out = fig_dir / "ảnh_kiểm_tra.png"
    banner = np.full((120, 200, 3), 240, dtype=np.uint8)
    banner[:30, :100] = 20
    banner[-30:, :100] = 20
    written_bytes = fig_mod.write_png_limited(banner, fig_out)
    assert fig_out.is_file() and fig_out.stat().st_size == written_bytes
    fig_mod._verify_gui_screenshot_png(fig_out)

    records: list[object] = []
    fig_mod.make_pipeline_figures(fig_dir, records)
    assert len(records) == 4
    for rec in records:
        assert (fig_dir / rec.filename).is_file()
