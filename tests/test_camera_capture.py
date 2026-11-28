"""Tests cho camera capture (Webcam, Camera USB ngoài, IoT HTTP) và UI dialog."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.io import safe_open
from imageguard.ingest import camera as camera_mod


class _FakeCapture:
    def __init__(self, *, opened: bool = True, frame: np.ndarray | None = None) -> None:
        self._opened = opened
        self._frame = frame
        self.released = False
        self.reads = 0

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        self.reads += 1
        if self._frame is None:
            return False, None
        return True, self._frame.copy()

    def release(self) -> None:
        self.released = True


def test_capture_device_frame_saves_lossless_png_and_releases(tmp_path: Path, monkeypatch) -> None:
    frame = np.random.default_rng(7).integers(0, 255, (96, 128, 3), dtype=np.uint8)
    fake = _FakeCapture(opened=True, frame=frame)
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", lambda _idx: fake)

    out = camera_mod.capture_device_frame(device_index=0, out_dir=tmp_path, warmup_frames=2)

    assert fake.released is True
    assert fake.reads == 2
    assert out.is_file()
    assert out.suffix == ".png"
    doc = safe_open(out)
    assert not hasattr(doc, "code")
    assert (doc.width, doc.height) == (128, 96)
    np.testing.assert_array_equal(doc.array, frame)


def test_capture_device_frame_unopened_releases_and_raises(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeCapture(opened=False, frame=None)
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", lambda _idx: fake)

    with pytest.raises(RuntimeError, match="Không mở được thiết bị camera"):
        camera_mod.capture_device_frame(device_index=1, out_dir=tmp_path)
    assert fake.released is True


def test_capture_device_frame_rejects_invalid_index(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="device_index"):
        camera_mod.capture_device_frame(device_index=-1, out_dir=tmp_path)
    with pytest.raises(TypeError, match="device_index"):
        camera_mod.capture_device_frame(device_index=True, out_dir=tmp_path)  # type: ignore[arg-type]


def test_capture_iot_http_frame_delegates_to_pull_due(tmp_path: Path, monkeypatch) -> None:
    seen: dict = {}

    def fake_pull_due(cfg, dest_dir, **kwargs):
        seen["cfg"] = cfg
        seen["dest_dir"] = dest_dir
        out_file = dest_dir / "shot.jpg"
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(b"\xff\xd8\xff\xe0dummy")
        return [{"status": "written", "basename": "shot.jpg"}]

    monkeypatch.setattr(camera_mod, "pull_due", fake_pull_due)

    path = camera_mod.capture_iot_http_frame("http://192.168.1.55:8080/shot.jpg", out_dir=tmp_path)

    assert path == tmp_path / "shot.jpg"
    assert path.is_file()
    assert seen["cfg"]["hosts"] == ["192.168.1.55"]
    assert seen["cfg"]["urls"] == ["http://192.168.1.55:8080/shot.jpg"]


def test_capture_iot_http_frame_refusal_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        camera_mod,
        "pull_due",
        lambda *_a, **_kw: [{"status": "refused", "reason": "loopback_blocked"}],
    )
    with pytest.raises(RuntimeError, match="loopback_blocked"):
        camera_mod.capture_iot_http_frame("http://127.0.0.1/shot.jpg", out_dir=tmp_path)


def test_probe_camera_indices_releases_each(monkeypatch) -> None:
    opened_flags = {0: True, 1: False, 2: True}
    created: list[_FakeCapture] = []

    def make_cap(idx: int):
        cap = _FakeCapture(opened=opened_flags.get(idx, False))
        created.append(cap)
        return cap

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", make_cap)
    found = camera_mod.probe_camera_indices(max_index=3)

    assert found == [0, 2]
    assert all(c.released for c in created)


def test_capture_dual_device_frames_combines_side_by_side(tmp_path: Path, monkeypatch) -> None:
    frame0 = np.full((80, 100, 3), 50, dtype=np.uint8)
    frame1 = np.full((160, 200, 3), 180, dtype=np.uint8)
    by_idx = {
        0: _FakeCapture(opened=True, frame=frame0),
        1: _FakeCapture(opened=True, frame=frame1),
    }
    monkeypatch.setattr(
        camera_mod, "_open_capture", lambda idx: by_idx[int(idx)]
    )

    out = camera_mod.capture_dual_device_frames(0, 1, out_dir=tmp_path, warmup_frames=1)
    assert out.is_file()
    doc = safe_open(out)
    assert not hasattr(doc, "code")
    # frame1 (160x200) co theo chieu cao frame0 (80) -> (80x100) -> ghep ngang thanh 200x80
    assert (doc.width, doc.height) == (200, 80)
    assert by_idx[0].released and by_idx[1].released

    with pytest.raises(ValueError, match="phải khác nhau"):
        camera_mod.capture_dual_device_frames(0, 0, out_dir=tmp_path)


def test_ui_camera_capture_loads_analyze_and_enhance(qtbot, tmp_path: Path, monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QDialog

    from imageguard.ui.app import build_services
    from imageguard.ui.main_window import MainWindow
    from imageguard.ui.screens import analyze as analyze_mod
    from imageguard.ui.screens import enhance as enhance_mod

    img = np.random.default_rng(11).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    cap_path = tmp_path / "webcam.png"
    cap_path.write_bytes(cv2.imencode(".png", img)[1].tobytes())

    class _FakeDlg:
        def __init__(self, parent=None) -> None:
            self.captured_path = cap_path

        def exec(self) -> int:
            return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(analyze_mod, "CameraCaptureDialog", _FakeDlg)
    monkeypatch.setattr(enhance_mod, "CameraCaptureDialog", _FakeDlg)

    analysis, history = build_services(
        db_path=tmp_path / "t.db", artifacts_root=tmp_path / "art"
    )
    win = MainWindow(analysis, history)
    qtbot.addWidget(win)

    assert win.analyze.chk_df.isChecked() is False
    win.analyze._capture_camera()
    assert win.analyze._path == cap_path
    assert win.analyze.chk_df.isChecked() is True
    assert win.analyze.chk_dfml.isChecked() is True
    assert win.analyze.chk_dfscreen.isChecked() is True
    assert win.analyze.chk_dfens.isChecked() is True
    methods = set(win.analyze._methods())
    assert {"deepfake", "deepfake_ml", "deepfake_screen", "deepfake_ensemble"} <= methods
    assert {"keypoint", "block"} <= methods

    win.enhance._capture_camera()
    assert win.enhance._path == cap_path


def test_camera_capture_dialog_modes_and_error_handling(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from imageguard.ui.widgets import camera_dialog as dlg_mod

    dlg = dlg_mod.CameraCaptureDialog(out_dir=tmp_path)
    qtbot.addWidget(dlg)

    # Mặc định chọn webcam: spin USB và URL IoT đều bị khóa.
    assert dlg.rad_webcam.isChecked()
    assert not dlg.spin_usb.isEnabled()
    assert not dlg.edit_url.isEnabled()

    # Chuyển sang USB ngoài -> bật spin_usb; sang Dual -> giữ spin_usb; sang IoT -> bật edit_url.
    dlg.rad_usb.setChecked(True)
    assert dlg.spin_usb.isEnabled()
    assert not dlg.edit_url.isEnabled()

    dlg.rad_dual.setChecked(True)
    assert dlg.spin_usb.isEnabled()
    assert not dlg.edit_url.isEnabled()

    dlg.rad_iot.setChecked(True)
    assert not dlg.spin_usb.isEnabled()
    assert dlg.edit_url.isEnabled()

    # Lỗi khi chụp hiển thị lên lbl_err và không accept dialog.
    def fail_iot(_url: str, _out: Path) -> Path:
        raise RuntimeError(
            "http connection failed http://user:pass@192.168.1.99/shot.jpg "
            "opening C:/Users/Jane Doe/secret.png"
        )

    monkeypatch.setattr(dlg_mod, "capture_iot_http_frame", fail_iot)
    dlg.edit_url.setText("http://192.168.1.99/shot.jpg")
    dlg.capture_now()
    qtbot.waitUntil(lambda: dlg._worker is None)
    assert dlg.captured_path is None
    assert "http connection failed" in dlg.lbl_err.text()
    assert "<url>" in dlg.lbl_err.text()
    assert "secret.png" in dlg.lbl_err.text()
    assert "user:pass" not in dlg.lbl_err.text()
    assert "192.168.1.99" not in dlg.lbl_err.text()
    assert "Jane Doe" not in dlg.lbl_err.text()

    # Thành công với USB ngoài (index 2) -> gán captured_path.
    expected = tmp_path / "usb2.png"
    calls: list[tuple[int, Path]] = []

    def ok_device(idx: int, out_dir: Path) -> Path:
        calls.append((idx, out_dir))
        cv2.imwrite(str(expected), np.zeros((48, 64, 3), dtype=np.uint8))
        return expected

    monkeypatch.setattr(dlg_mod, "capture_device_frame", ok_device)
    dlg.rad_usb.setChecked(True)
    dlg.chk_burst.setChecked(False)
    dlg.spin_usb.setValue(2)
    dlg.capture_now()
    qtbot.waitUntil(lambda: dlg._worker is None)
    assert calls == [(2, tmp_path)]
    assert dlg.captured_path == expected
    assert dlg.lbl_err.text() == ""

    # Dual mode -> gọi capture_dual_device_frames(0, 2, tmp_path)
    dual_calls: list[tuple[int, int, Path]] = []
    expected_dual = tmp_path / "dual.png"

    def ok_dual(idx_a: int, idx_b: int, out_dir: Path) -> Path:
        dual_calls.append((idx_a, idx_b, out_dir))
        cv2.imwrite(str(expected_dual), np.zeros((48, 128, 3), dtype=np.uint8))
        return expected_dual

    monkeypatch.setattr(dlg_mod, "capture_dual_device_frames", ok_dual)
    dlg.show()
    dlg.rad_dual.setChecked(True)
    dlg.spin_usb.setValue(2)
    dlg.capture_now()
    qtbot.waitUntil(lambda: dlg._worker is None)
    assert dual_calls == [(0, 2, tmp_path)]
    assert dlg.captured_path == expected_dual


def test_burst_checkbox_routes_webcam_to_burst_helper(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from imageguard.ui.widgets import camera_dialog as dlg_mod

    dlg = dlg_mod.CameraCaptureDialog(out_dir=tmp_path)
    assert dlg.chk_burst.isChecked()
    expected = tmp_path / "burst.png"
    calls: list[int] = []

    def fake_burst(_self, device_index: int, out_dir: Path) -> Path:
        calls.append(device_index)
        cv2.imwrite(str(expected), np.zeros((48, 64, 3), dtype=np.uint8))
        return expected

    monkeypatch.setattr(dlg_mod.CameraCaptureDialog, "_capture_burst", fake_burst)
    assert dlg.chk_burst.toolTip() == dlg_mod.strings.CAMERA_BURST_HINT
    dlg.capture_now()
    qtbot.waitUntil(lambda: dlg._worker is None)
    assert calls == [0]
    assert dlg.captured_path == expected

    dlg.rad_dual.setChecked(True)
    assert not dlg.chk_burst.isChecked()
    assert not dlg.chk_burst.isEnabled()


def test_capture_burst_helper_returns_primary_from_module(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from imageguard.ingest import camera_burst
    from imageguard.ui.widgets import camera_dialog as dlg_mod

    expected = tmp_path / "primary.png"
    cv2.imwrite(str(expected), np.zeros((64, 64, 3), dtype=np.uint8))
    manifest = tmp_path / "primary.burst.json"
    calls: list[int] = []

    def fake_burst(device_index: int, out_dir: Path, *, frame_count: int = 5, warmup_frames: int = 2):
        calls.append(frame_count)
        assert device_index == 1
        assert out_dir == tmp_path
        return expected, manifest

    monkeypatch.setattr(camera_burst, "capture_device_burst", fake_burst)
    dlg = dlg_mod.CameraCaptureDialog(out_dir=tmp_path)
    assert dlg._capture_burst(1, tmp_path) == expected
    assert calls == [5]
