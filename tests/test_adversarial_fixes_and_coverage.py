"""Adversarial regression tests for fixes and low-coverage edge paths."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import http.server
import inspect
import shutil
import socket
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import cv2
import numpy as np
import pytest
from sqlalchemy.exc import DatabaseError

from imageguard.core import config as cfg_mod
from imageguard.core import paths
from imageguard.core.enums import Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.evaluation import robustness
from imageguard.forensics.localize.heatmap import (
    overlay_heatmap,
    overlay_mask,
    render_heatmap,
)
from imageguard.ingest import camera as camera_mod
from imageguard.ingest import commands as ingest_commands
from imageguard.ingest import http_pull
from imageguard.ipkit import edges
from imageguard.storage import artifacts as artifacts_mod
from imageguard.storage import db as db_mod


def _write_png(path: Path, arr: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    path.write_bytes(buf.tobytes())
    return path


def test_db_corrupt_file_disposes_engine_and_allows_unlink(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not a valid sqlite database file at all")

    with pytest.raises(DatabaseError):
        db_mod.make_session_factory(corrupt)

    # Disposing on early WAL/create_all failure must release Windows file locks.
    corrupt.unlink()
    assert not corrupt.exists()


def test_db_url_special_chars_and_frozen_dynamic_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qtbot
) -> None:
    pytest.importorskip("PySide6")
    from imageguard.ui.widgets.camera_dialog import CameraCaptureDialog
    from imageguard.ui.widgets.sample_dialog import (
        SampleGeneratorDialog,
        generate_demo_sample,
    )

    # '#' and '%' in path must not be interpreted as SQLite URL fragment/escape.
    special_db = tmp_path / "db#1%20frag.sqlite"
    factory = db_mod.make_session_factory(special_db)
    try:
        assert special_db.is_file()
        assert not (tmp_path / "db").exists()
    finally:
        factory.kw["bind"].dispose()

    # Dynamic FROZEN=True resolution even when modules were imported earlier.
    bundle_dir = tmp_path / "bundle_ảnh"
    app_dir = tmp_path / "app_ảnh"
    (bundle_dir / "configs").mkdir(parents=True)
    app_dir.mkdir(parents=True)
    repo_cfg = Path(__file__).resolve().parents[1] / "configs" / "thresholds.yaml"
    frozen_cfg = bundle_dir / "configs" / "thresholds.yaml"
    shutil.copy2(repo_cfg, frozen_cfg)
    frozen_text = frozen_cfg.read_text(encoding="utf-8")
    assert "value: 2048" in frozen_text
    frozen_cfg.write_text(
        frozen_text.replace("value: 2048", "value: 2047", 1),
        encoding="utf-8",
    )
    exe_path = app_dir / "ImageGuard.exe"
    exe_path.write_bytes(b"")

    monkeypatch.setattr(paths, "FROZEN", True)
    monkeypatch.setattr(paths.sys, "executable", str(exe_path))
    monkeypatch.setattr(paths.sys, "_MEIPASS", str(bundle_dir), raising=False)
    monkeypatch.delattr(paths.sys, "__compiled__", raising=False)

    expected_data = app_dir / "data"
    assert paths.data_root() == expected_data
    assert paths.config_path() == frozen_cfg
    assert db_mod.DEFAULT_DB != expected_data / "imageguard.db"

    loaded_cfg = cfg_mod.load_thresholds()
    assert loaded_cfg.limit("work_max_side", 0) == 2047
    assert loaded_cfg.snapshot_hash() != cfg_mod.load_thresholds(repo_cfg).snapshot_hash()

    store = artifacts_mod.ArtifactStore()
    assert store.root == expected_data / "artifacts"

    frozen_factory = db_mod.make_session_factory()
    try:
        assert (expected_data / "imageguard.db").is_file()
    finally:
        frozen_factory.kw["bind"].dispose()

    demo_path, _ = generate_demo_sample("clean")
    assert demo_path.parent == expected_data / "demo_samples"
    assert demo_path.is_file()

    seen_out_dirs: list[Path] = []

    class _FakeCap:
        def isOpened(self) -> bool:
            return True

        def read(self):
            return True, np.full((16, 16, 3), 80, dtype=np.uint8)

        def release(self) -> None:
            return

    monkeypatch.setattr(camera_mod, "_open_capture", lambda _idx: _FakeCap())
    single_cap = camera_mod.capture_device_frame(0, warmup_frames=1)
    assert single_cap.parent == expected_data / "captures"
    dual_cap = camera_mod.capture_dual_device_frames(0, 1, warmup_frames=1)
    assert dual_cap.parent == expected_data / "captures"

    def fake_pull_due(_cfg, out_dir: Path, **_kw):
        seen_out_dirs.append(Path(out_dir))
        out = Path(out_dir) / "iot.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\xff\xd8\xffok")
        return [{"status": "written", "basename": "iot.jpg"}]

    monkeypatch.setattr(camera_mod, "pull_due", fake_pull_due)
    iot_cap = camera_mod.capture_iot_http_frame("http://camera.example/snap.jpg")
    assert iot_cap == expected_data / "captures" / "iot.jpg"
    assert seen_out_dirs == [expected_data / "captures"]

    cam_dlg = CameraCaptureDialog()
    qtbot.addWidget(cam_dlg)
    assert cam_dlg._out_dir == expected_data / "captures"

    smp_dlg = SampleGeneratorDialog()
    qtbot.addWidget(smp_dlg)
    assert smp_dlg._out_dir == expected_data / "demo_samples"


def test_run_ingest_disposes_engine_on_setup_and_scan_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    watch = tmp_path / "watch"
    watch.mkdir()
    cfg = tmp_path / "ingest.yaml"
    cfg.write_text("methods: [ela]\ndata_dir: data\n", encoding="utf-8")

    # Failure AFTER make_session_factory succeeds (FolderIngestor constructor raises)
    def boom_init(*_a, **_kw):
        raise RuntimeError("ingestor init boom")

    monkeypatch.setattr(ingest_commands, "FolderIngestor", boom_init)
    rc1 = ingest_commands.run_ingest("once", watch, cfg)
    assert rc1 == 1
    db_file = tmp_path / "data" / "imageguard.db"
    assert db_file.is_file()
    db_file.unlink()
    assert not db_file.exists()

    # Failure during scan_once
    class _FailingIngestor:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def scan_once(self, _dir: Path):
            raise RuntimeError("scan failed boom")

    monkeypatch.setattr(ingest_commands, "FolderIngestor", _FailingIngestor)
    rc2 = ingest_commands.run_ingest("once", watch, cfg)
    assert rc2 == 1
    assert db_file.is_file()
    db_file.unlink()
    assert not db_file.exists()


def test_heatmap_empty_nonuint8_bool_nan_and_zero_channel_contracts() -> None:
    empty_2d = np.zeros((0, 0), dtype=np.float32)
    empty_3d = np.zeros((0, 0, 3), dtype=np.uint8)
    empty_mask = np.zeros((0, 0), dtype=np.uint8)

    hm_empty = render_heatmap(empty_2d)
    assert hm_empty.shape == (0, 0, 3) and hm_empty.dtype == np.uint8

    ov_hm_empty = overlay_heatmap(empty_3d, empty_2d)
    assert ov_hm_empty.shape == (0, 0, 3) and ov_hm_empty.dtype == np.uint8

    ov_mask_empty = overlay_mask(empty_3d, empty_mask)
    assert ov_mask_empty is not None
    assert ov_mask_empty.shape == (0, 0, 3) and ov_mask_empty.dtype == np.uint8

    base_u8 = np.full((8, 8, 3), 42, dtype=np.uint8)
    assert np.array_equal(overlay_heatmap(base_u8, empty_2d), base_u8)
    assert np.array_equal(overlay_mask(base_u8, empty_mask), base_u8)

    # Non-uint8 base inputs (int32, float32 with NaN/Inf, bool)
    heat = np.ones((8, 8), dtype=np.float32)
    int32_base = np.full((8, 8), 100, dtype=np.int32)
    out_i32 = overlay_heatmap(int32_base, heat)
    assert out_i32.shape == (8, 8, 3) and out_i32.dtype == np.uint8

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    f32_base = np.full((8, 8, 3), 100.0, dtype=np.float32)
    out_f32_mask = overlay_mask(f32_base, mask)
    assert out_f32_mask.shape == (8, 8, 3) and out_f32_mask.dtype == np.uint8

    bool_base = np.eye(8, dtype=bool)
    out_bool = overlay_mask(bool_base, mask)
    assert out_bool.shape == (8, 8, 3) and out_bool.dtype == np.uint8

    nan_base = np.array([[np.nan, np.inf], [-np.inf, 128.0]], dtype=np.float32)
    out_nan = overlay_heatmap(nan_base, np.ones((2, 2), dtype=np.float32))
    assert out_nan.shape == (2, 2, 3) and out_nan.dtype == np.uint8

    with pytest.raises(ValueError):
        render_heatmap(np.zeros((4, 4, 0), dtype=np.float32))
    with pytest.raises(ValueError):
        overlay_mask(np.zeros((4, 4, 0), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))


def test_image_canvas_wheel_horizontal_no_zoom_and_dtype_shape_guards(qtbot) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    from imageguard.ui.widgets.image_canvas import (
        ImageCanvas,
        heat_to_qpixmap,
        ndarray_to_qpixmap,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Unsupported image dtype"):
            ndarray_to_qpixmap(np.zeros((4, 4, 3), dtype=np.complex64))
    assert not any("Complex" in str(w.message) for w in caught)

    with pytest.raises(ValueError, match="Unsupported image dtype"):
        ndarray_to_qpixmap(np.array([["a", "b"]], dtype=object))

    bool_pm = ndarray_to_qpixmap(np.array([[True, False]], dtype=bool))
    assert not bool_pm.isNull()
    qimg = bool_pm.toImage()
    assert qimg.pixelColor(0, 0).red() == 255
    assert qimg.pixelColor(1, 0).red() == 0

    assert heat_to_qpixmap(np.zeros((0, 0), dtype=np.float32), (8, 8)).isNull()
    assert heat_to_qpixmap(np.ones((8, 8), dtype=np.float32), (0, 8)).isNull()
    assert heat_to_qpixmap(np.ones((8, 8), dtype=np.float32), (8, 0)).isNull()

    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(np.full((120, 160, 3), 90, dtype=np.uint8))
    assert canvas._base is not None

    before_m11 = canvas.transform().m11()
    horiz_ev = QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(0, 0),
        QPoint(120, 0),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    canvas.wheelEvent(horiz_ev)
    assert canvas.transform().m11() == pytest.approx(before_m11)

    zoom_in_ev = QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    canvas.wheelEvent(zoom_in_ev)
    after_in_m11 = canvas.transform().m11()
    assert after_in_m11 == pytest.approx(before_m11 * 1.25)

    zoom_out_ev = QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    canvas.wheelEvent(zoom_out_ev)
    assert canvas.transform().m11() == pytest.approx(after_in_m11 * 0.8)

    # Empty image restores placeholder rather than attaching dead empty pixmap
    canvas.set_image(np.zeros((0, 0, 3), dtype=np.uint8))
    assert canvas._base is None


def test_enhance_metrics_analyze_history_overlays_and_a11y(qtbot, tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QWidget

    from imageguard.ui.a11y import apply_a11y
    from imageguard.ui.app import build_services
    from imageguard.ui.screens.analyze import AnalyzeScreen
    from imageguard.ui.screens.enhance import _compute_quality_metrics
    from imageguard.ui.screens.history import AnalysisDetailDialog

    # _compute_quality_metrics across (H, W, 1), 2D, 3-ch, and 4-ch BGRA
    for shape in ((32, 32, 1), (32, 32), (32, 32, 3), (32, 32, 4)):
        a = np.full(shape, 80, dtype=np.uint8)
        b = np.full(shape, 95, dtype=np.uint8)
        txt = _compute_quality_metrics(a, b)
        assert "PSNR=" in txt and "SSIM=" in txt

    w = QWidget()
    qtbot.addWidget(w)
    with pytest.raises(ValueError, match="non-empty"):
        apply_a11y(w, "   ")

    # AnalyzeScreen._apply_overlay handles 2D doc and (H, W, 1) mask while rejecting (H, W, 3)
    analysis_svc, _ = build_services(
        db_path=tmp_path / "an.db",
        artifacts_root=tmp_path / "art",
    )
    try:
        screen = AnalyzeScreen(analysis_svc)
        qtbot.addWidget(screen)
        img2d = np.full((64, 64), 120, dtype=np.uint8)
        screen._doc = ImageDocument(
            array=img2d,
            width=64,
            height=64,
            channels=1,
            image_format="PNG",
            sha256="0" * 64,
            file_bytes=4096,
            file_name="gray.png",
        )
        screen.canvas.set_image(img2d)

        m_1ch = SimpleNamespace(mask=np.ones((64, 64, 1), dtype=np.uint8))
        m_3ch_bad = SimpleNamespace(mask=np.ones((64, 64, 3), dtype=np.uint8))
        screen.overlay_box.addItem("Mask 1ch", ("mask", m_1ch))
        screen.overlay_box.addItem("Mask 3ch bad", ("mask", m_3ch_bad))

        screen.overlay_box.setCurrentIndex(1)
        assert screen.canvas._overlay is not None
        assert screen.lbl_err.text() == ""

        screen.overlay_box.setCurrentIndex(2)
        assert screen.canvas._overlay is None
        assert screen.lbl_err.text() != ""
    finally:
        analysis_svc.images._sf.kw["bind"].dispose()

    # AnalysisDetailDialog._show_art renders 2D gray, 4-ch BGRA, and handles invalid bytes
    gray_png = cv2.imencode(".png", np.full((24, 24), 150, dtype=np.uint8))[1].tobytes()
    bgra_png = cv2.imencode(".png", np.full((24, 24, 4), 200, dtype=np.uint8))[1].tobytes()

    class _DummyStore:
        def read(self, rel: str) -> bytes:
            if rel == "a/gray.png":
                return gray_png
            if rel == "a/bgra.png":
                return bgra_png
            return b"corrupt-bytes"

    svc = SimpleNamespace(store=_DummyStore())
    detail = {
        "id": "aid-1",
        "image_name": "x.png",
        "created_at": "2026-11-22T00:00:00",
        "verdict": "inconclusive",
        "methods": [],
        "duration_ms": 10,
        "config_hash": "cfg",
        "artifacts": [
            {"kind": "mask", "method": "ela", "rel_path": "a/gray.png"},
            {"kind": "heatmap", "method": "ela", "rel_path": "a/bgra.png"},
            {"kind": "ela", "method": "ela", "rel_path": "a/bad.png"},
        ],
    }
    dlg = AnalysisDetailDialog(detail, svc)
    qtbot.addWidget(dlg)

    dlg.combo.setCurrentIndex(1)
    assert dlg.canvas._base is not None
    assert dlg.lbl_art_err.text() == ""

    dlg.combo.setCurrentIndex(2)
    assert dlg.canvas._base is not None
    assert dlg.lbl_art_err.text() == ""

    dlg.combo.setCurrentIndex(3)
    assert dlg.canvas._base is None
    assert "bad.png" in dlg.lbl_art_err.text()


def test_ipkit_edges_annotations_and_robustness_edge_branches(tmp_path: Path) -> None:
    sobel_ret = inspect.get_annotations(edges.sobel, eval_str=True)["return"]
    prewitt_ret = inspect.get_annotations(edges.prewitt, eval_str=True)["return"]
    assert sobel_ret == tuple[np.ndarray, np.ndarray]
    assert prewitt_ret == tuple[np.ndarray, np.ndarray]

    img2d = np.arange(64, dtype=np.uint8).reshape(8, 8)
    gx, gy = edges.sobel(img2d)
    assert gx.shape == (8, 8) and gx.dtype == np.float32
    assert gy.shape == (8, 8) and gy.dtype == np.float32
    px, py = edges.prewitt(np.stack([img2d] * 3, axis=2))
    assert px.shape == (8, 8, 3) and py.shape == (8, 8, 3)

    # Robustness edge branches under Unicode path: missing, corrupt, tiny rejected, INCONCLUSIVE
    u_dir = tmp_path / "kiểm_thử_ảnh"
    u_dir.mkdir()
    missing = u_dir / "missing.png"
    corrupt = u_dir / "corrupt.png"
    corrupt.write_bytes(b"invalid-png-data")
    tiny = _write_png(u_dir / "tiny.png", np.zeros((2, 2, 3), dtype=np.uint8))
    valid = _write_png(
        u_dir / "valid_ảnh.png",
        np.random.default_rng(7).integers(0, 255, (96, 96, 3), dtype=np.uint8),
    )

    class _DummyMethod:
        name = "ela"

        def analyze(self, _doc, _cfg):
            return AlgorithmResult(
                method="ela",
                verdict=Verdict.INCONCLUSIVE,
                score=0.25,
                score_definition="inconclusive test",
                duration_ms=1,
            )

    cfg = cfg_mod.load_thresholds()
    res_default = robustness.run_battery([missing, corrupt, valid], [_DummyMethod()], cfg)
    for dname in ("none", "jpeg_90", "jpeg_70", "rotate_5", "scale_0.9"):
        assert dname in res_default["ela"]
        assert res_default["ela"][dname]["n"] == 1
        assert res_default["ela"][dname]["n_inconclusive"] == 1
        assert res_default["ela"][dname]["detection_rate"] == 0.0

    # Tiny image gets rejected by safe_open -> skipped gracefully
    res_tiny = robustness.run_battery(
        [tiny],
        [_DummyMethod()],
        cfg,
        distortions={"noop": lambda arr: arr},
    )
    assert res_tiny["ela"]["noop"]["n"] == 0
    assert res_tiny["ela"]["noop"]["detection_rate"] is None


def test_http_pull_disabled_ipv6_ssrf_http_error_and_retry_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with http_pull._LAST_ATTEMPTS_LOCK:
        state = getattr(http_pull.pull_due, "_last_attempts", None)
        if state is not None:
            state.clear()

    # Disabled config short-circuits immediately
    assert http_pull.pull_due({"enabled": False}, tmp_path) == []

    # IPv4-mapped IPv6 metadata address & loopback must both be blocked cleanly
    for v6_addr in ("::ffff:169.254.169.254", "::ffff:127.0.0.1", "::1"):
        monkeypatch.setattr(
            http_pull.socket,
            "getaddrinfo",
            lambda host, port, type, addr=v6_addr: [
                (http_pull.socket.AF_INET6, http_pull.socket.SOCK_STREAM, 6, "", (addr, port, 0, 0))
            ],
        )
        res = http_pull.pull_due(
            {
                "enabled": True,
                "urls": ["http://cam.example/shot.jpg"],
                "hosts": ["cam.example"],
                "interval_s": 5,
                "timeout_s": 2,
                "max_retries": 0,
                "allow_loopback": False,
                "allow_link_local": False,
            },
            tmp_path,
            now=10.0,
        )
        assert res[0]["status"] == "refused"

    # Retry exhaustion on repeated timeouts -> request_error
    attempts = 0

    def always_timeout(self, _request, timeout=None):
        nonlocal attempts
        attempts += 1
        raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr(
        http_pull.socket,
        "getaddrinfo",
        lambda host, port, type: [
            (http_pull.socket.AF_INET, http_pull.socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))
        ],
    )
    monkeypatch.setattr("urllib.request.OpenerDirector.open", always_timeout)
    with http_pull._LAST_ATTEMPTS_LOCK:
        state = getattr(http_pull.pull_due, "_last_attempts", None)
        if state is not None:
            state.clear()
    res_retry = http_pull.pull_due(
        {
            "enabled": True,
            "urls": ["http://cam.example/shot.jpg"],
            "hosts": ["cam.example"],
            "interval_s": 5,
            "timeout_s": 1,
            "max_retries": 1,
            "allow_loopback": True,
            "allow_link_local": False,
        },
        tmp_path,
        now=50.0,
    )
    assert attempts == 2
    assert res_retry == [{"status": "refused", "reason": "timeout"}]

    # Real HTTP 404 server refusal without leaking files
    class _NotFoundHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, _fmt: str, *_args: object) -> None:
            return

    @contextmanager
    def _serve_404() -> Iterator[str]:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NotFoundHandler)
        srv.daemon_threads = True
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            yield f"http://{srv.server_address[0]}:{srv.server_address[1]}"
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=2)

    monkeypatch.undo()
    with _serve_404() as base:
        with http_pull._LAST_ATTEMPTS_LOCK:
            state = getattr(http_pull.pull_due, "_last_attempts", None)
            if state is not None:
                state.clear()
        res_404 = http_pull.pull_due(
            {
                "enabled": True,
                "urls": [f"{base}/missing.jpg"],
                "hosts": ["127.0.0.1"],
                "interval_s": 5,
                "timeout_s": 2,
                "max_retries": 0,
                "allow_loopback": True,
                "allow_link_local": False,
            },
            tmp_path,
            now=100.0,
        )
    assert res_404 == [{"status": "refused", "reason": "http_status"}]
    assert list(tmp_path.iterdir()) == []


def test_http_pull_supports_chunked_transfer_and_keepalive_responses(
    tmp_path: Path,
) -> None:
    jpg_bytes = cv2.imencode(
        ".jpg", np.full((16, 16, 3), 130, dtype=np.uint8)
    )[1].tobytes()

    class _ChunkedHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            mid = len(jpg_bytes) // 2
            for part in (jpg_bytes[:mid], jpg_bytes[mid:]):
                self.wfile.write(f"{len(part):x}\r\n".encode("ascii") + part + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")

        def log_message(self, _fmt: str, *_args: object) -> None:
            return

    chunked_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ChunkedHandler)
    chunked_srv.daemon_threads = True
    chunked_th = threading.Thread(target=chunked_srv.serve_forever, daemon=True)
    chunked_th.start()
    try:
        chunked_url = (
            f"http://127.0.0.1:{chunked_srv.server_address[1]}/chunked.jpg"
        )
        res_chunked = http_pull.pull_due(
            {
                "enabled": True,
                "urls": [chunked_url],
                "hosts": ["127.0.0.1"],
                "interval_s": 5,
                "timeout_s": 2,
                "max_retries": 0,
                "allow_loopback": True,
            },
            tmp_path,
            now=200.0,
        )
        assert res_chunked == [{"status": "written", "basename": "chunked.jpg"}]
        assert (tmp_path / "chunked.jpg").read_bytes() == jpg_bytes
    finally:
        chunked_srv.shutdown()
        chunked_srv.server_close()
        chunked_th.join(timeout=2)

    # HTTP/1.1 keep-alive server that keeps TCP socket open after Content-Length bytes
    ready: list[int] = []
    stop_ev = threading.Event()

    def _keepalive_server() -> None:
        srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv_sock.bind(("127.0.0.1", 0))
        srv_sock.listen(1)
        ready.append(srv_sock.getsockname()[1])
        try:
            conn, _addr = srv_sock.accept()
            with conn:
                conn.recv(4096)
                header = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpg_bytes)}\r\n"
                    "Connection: keep-alive\r\n\r\n"
                ).encode("ascii")
                conn.sendall(header + jpg_bytes)
                stop_ev.wait(timeout=5)
        finally:
            srv_sock.close()

    ka_th = threading.Thread(target=_keepalive_server, daemon=True)
    ka_th.start()
    while not ready:
        pass
    try:
        res_ka = http_pull.pull_due(
            {
                "enabled": True,
                "urls": [f"http://127.0.0.1:{ready[0]}/keepalive.jpg"],
                "hosts": ["127.0.0.1"],
                "interval_s": 5,
                "timeout_s": 0.5,
                "max_retries": 0,
                "allow_loopback": True,
            },
            tmp_path,
            now=300.0,
        )
        assert res_ka == [{"status": "written", "basename": "keepalive.jpg"}]
        assert (tmp_path / "keepalive.jpg").read_bytes() == jpg_bytes
    finally:
        stop_ev.set()
        ka_th.join(timeout=2)
