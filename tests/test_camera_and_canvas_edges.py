"""Edge tests cho ingest/camera, core/metadata nested EXIF và ui/widgets/image_canvas."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image

from imageguard.core.metadata import KEY_TAGS, extract_exif
from imageguard.ingest import camera as camera_mod


class _FakeCap:
    def __init__(self, *, opened: bool = True, frames=None) -> None:
        self._opened = opened
        self._frames = list(frames or [])
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        return self._frames.pop(0)

    def release(self) -> None:
        self.released = True


def test_camera_win32_dshow_open_and_fallback(monkeypatch) -> None:
    dshow_ok = _FakeCap(opened=True)
    calls: list[tuple] = []

    def cap_factory(*args):
        calls.append(args)
        if len(args) == 2:
            return dshow_ok
        return _FakeCap(opened=False)

    monkeypatch.setattr(camera_mod.sys, "platform", "win32")
    monkeypatch.setattr(camera_mod.cv2, "CAP_DSHOW", 700, raising=False)
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", cap_factory)

    got = camera_mod._open_capture(0)
    assert got is dshow_ok
    assert calls == [(0, 700)]

    # When DSHOW cap is not opened, it must release DSHOW cap and fallback to VideoCapture(idx).
    dshow_closed = _FakeCap(opened=False)
    fallback_cap = _FakeCap(opened=True)
    calls.clear()

    def cap_fallback(*args):
        calls.append(args)
        return dshow_closed if len(args) == 2 else fallback_cap

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", cap_fallback)
    got2 = camera_mod._open_capture(1)
    assert got2 is fallback_cap
    assert dshow_closed.released is True
    assert calls == [(1, 700), (1,)]


def test_camera_invalid_frames_imencode_failure_and_iot_validation(
    tmp_path: Path, monkeypatch
) -> None:
    # All warmup frames invalid (False, 2D gray, float32) -> RuntimeError & release()
    bad_frames = [
        (False, None),
        (True, np.zeros((10, 10), dtype=np.uint8)),
        (True, np.zeros((10, 10, 3), dtype=np.float32)),
    ]
    cap = _FakeCap(opened=True, frames=bad_frames)
    monkeypatch.setattr(camera_mod, "_open_capture", lambda _idx: cap)

    with pytest.raises(RuntimeError, match="Không đọc được khung hình hợp lệ"):
        camera_mod.capture_device_frame(0, out_dir=tmp_path, warmup_frames=3)
    assert cap.released is True

    # imencode failure raises RuntimeError
    good_cap = _FakeCap(
        opened=True, frames=[(True, np.zeros((20, 20, 3), dtype=np.uint8))]
    )
    monkeypatch.setattr(camera_mod, "_open_capture", lambda _idx: good_cap)
    monkeypatch.setattr(camera_mod.cv2, "imencode", lambda *_a, **_kw: (False, None))
    with pytest.raises(RuntimeError, match="Không mã hóa được"):
        camera_mod.capture_device_frame(0, out_dir=tmp_path, warmup_frames=1)

    # IoT HTTP validation errors & empty pull_due return
    with pytest.raises(ValueError, match="URL camera"):
        camera_mod.capture_iot_http_frame("   ", out_dir=tmp_path)
    with pytest.raises(ValueError, match="timeout"):
        camera_mod.capture_iot_http_frame("http://example.com/a.jpg", out_dir=tmp_path, timeout=0)
    with pytest.raises(ValueError, match="timeout"):
        camera_mod.capture_iot_http_frame("http://example.com/a.jpg", out_dir=tmp_path, timeout=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hostname"):
        camera_mod.capture_iot_http_frame("http:///shot.jpg", out_dir=tmp_path)

    monkeypatch.setattr(camera_mod, "pull_due", lambda *_a, **_kw: [])
    with pytest.raises(RuntimeError, match="Không nhận được phản hồi"):
        camera_mod.capture_iot_http_frame("http://example.com/a.jpg", out_dir=tmp_path)


def test_metadata_extract_exif_corrupt_and_nested_ifd() -> None:
    # Corrupt bytes -> all KEY_TAGS missing, no exception raised.
    bad = extract_exif(b"not-an-image-at-all")
    assert bad["present"] == {}
    assert bad["missing"] == list(KEY_TAGS)
    assert bad["tag_count"] == 0

    # JPEG with Make in IFD0 and DateTimeOriginal inside nested ExifIFD (0x8769).
    im = Image.new("RGB", (32, 32), (100, 120, 140))
    exif = Image.Exif()
    exif[0x010F] = "CanonTest"  # Make
    exif[0x8769] = {0x9003: "2026:11:22 10:00:00"}  # ExifOffset -> DateTimeOriginal
    buf = BytesIO()
    im.save(buf, format="JPEG", exif=exif.tobytes())

    info = extract_exif(buf.getvalue())
    assert info["present"].get("Make") == "CanonTest"
    assert info["present"].get("DateTimeOriginal") == "2026:11:22 10:00:00"
    assert "DateTimeOriginal" not in info["missing"]


def test_image_canvas_interactions_heat_resize_and_drop(qtbot, tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
    from PySide6.QtGui import (
        QDragEnterEvent,
        QDragMoveEvent,
        QDropEvent,
        QKeyEvent,
        QWheelEvent,
    )

    from imageguard.ui.widgets.image_canvas import ImageCanvas, heat_to_qpixmap

    # heat_to_qpixmap resizes heatmap to target base_shape
    pm = heat_to_qpixmap(np.ones((16, 16), dtype=np.float32), (48, 64))
    assert (pm.width(), pm.height()) == (64, 48)

    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(300, 200)

    # Zoom step before image loaded is a safe no-op.
    canvas._zoom_step(True)

    bgr = np.full((64, 80, 3), 120, dtype=np.uint8)
    canvas.set_image(bgr)

    # Wheel zoom & key press zoom/pan/home
    wheel = QWheelEvent(
        QPointF(50, 50),
        QPointF(50, 50),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    canvas.wheelEvent(wheel)
    for key in (
        Qt.Key.Key_Plus,
        Qt.Key.Key_Minus,
        Qt.Key.Key_Left,
        Qt.Key.Key_Right,
        Qt.Key.Key_Up,
        Qt.Key.Key_Down,
        Qt.Key.Key_Home,
    ):
        ev = QKeyEvent(QKeyEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
        canvas.keyPressEvent(ev)
        assert ev.isAccepted()

    # Drag & drop valid image path emits image_dropped
    img_path = tmp_path / "drop.png"
    Image.fromarray(bgr).save(img_path, format="PNG")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(img_path))])

    dropped: list[Path] = []
    canvas.image_dropped.connect(dropped.append)

    enter_ev = QDragEnterEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    canvas.dragEnterEvent(enter_ev)
    assert enter_ev.isAccepted()

    move_ev = QDragMoveEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    canvas.dragMoveEvent(move_ev)
    assert move_ev.isAccepted()

    drop_ev = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    canvas.dropEvent(drop_ev)
    assert dropped == [img_path]


def test_enhance_screen_params_validation_and_blend_slider(qtbot, tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from imageguard.ui.screens.enhance import EnhanceScreen

    screen = EnhanceScreen()
    qtbot.addWidget(screen)

    # Verify parameter extraction for clahe, cam_calibrate, freq_sharpen
    for kind, expected_keys in (
        ("clahe", {"clip_limit", "tile_grid_size"}),
        ("cam_calibrate", {"gamma", "clip_limit", "boost_k"}),
        ("freq_sharpen", {"d0", "strength", "median_k"}),
    ):
        idx = screen.combo_op.findData(kind)
        assert idx >= 0
        screen.combo_op.setCurrentIndex(idx)
        params = screen._current_params()
        assert set(params.keys()) == expected_keys
        assert screen._validate_params(kind, params) == ""

    # Even-kernel validation errors
    assert "lẻ" in screen._validate_params("unsharp", {"ksize": 4, "sigma": 1.0, "amount": 1.0})
    assert "lẻ" in screen._validate_params("freq_sharpen", {"d0": 35, "strength": 1.0, "median_k": 4})
    assert "lẻ" in screen._validate_params(
        "nlm",
        {"h": 3.0, "hColor": 3.0, "templateWindowSize": 4, "searchWindowSize": 21},
    )

    # Load an image, simulate result, and test blend slider updates overlay opacity without re-running
    img_path = tmp_path / "enh.png"
    bgr = np.full((64, 64, 3), 100, dtype=np.uint8)
    Image.fromarray(bgr).save(img_path, format="PNG")
    screen.load_path(img_path)

    # Blend change before result is safe no-op
    screen._on_blend_changed(50)
    assert "50" in screen.lbl_blend_value.text()

    # Simulate valid derived result
    screen._cancel_event.clear()
    derived = np.full((64, 64, 3), 200, dtype=np.uint8)
    screen._on_result((derived, {}))
    assert screen.slider_blend.isEnabled()
    assert screen.canvas._overlay is not None

    screen.slider_blend.setValue(35)
    assert "35" in screen.lbl_blend_value.text()
    assert screen.canvas._overlay.opacity() == pytest.approx(0.35)
