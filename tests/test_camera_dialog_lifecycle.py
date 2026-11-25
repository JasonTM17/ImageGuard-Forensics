"""Controlled device waits exercise the actual Qt dialog slots."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt, QTimer

from imageguard.ui.widgets import camera_dialog as camera


@pytest.fixture
def device(qtbot, tmp_path):
    path = tmp_path / "camera.png"
    Image.fromarray(np.full((80, 96, 3), 100, np.uint8)).save(path)
    dialog = camera.CameraCaptureDialog(out_dir=tmp_path)
    dialog._burst_enabled = False
    dialog.chk_burst.setChecked(False)
    dialog.lbl_probe_info.setText("")
    qtbot.addWidget(dialog)
    dialog.show()
    entered, release = threading.Event(), threading.Event()
    threads = []

    def wait():
        threads.append(threading.get_ident())
        entered.set()
        release.wait(1)

    yield dialog, path, entered, release, threads, wait
    release.set()
    qtbot.waitUntil(lambda: getattr(dialog, "_worker", None) is None, timeout=3000)


@pytest.mark.parametrize("operation", ["probe", "preview", "capture"])
def test_camera_io_keeps_gui_responsive_and_prevents_overlap(qtbot, monkeypatch, device, operation):
    dialog, path, entered, release, threads, wait = device
    gui_thread = threading.get_ident()

    def slow_probe(**_kwargs):
        wait()
        return [0, 2]

    def slow_capture(_index, _directory):
        wait()
        return path

    monkeypatch.setattr(camera, "probe_camera_indices", slow_probe)
    monkeypatch.setattr(camera, "capture_device_frame", slow_capture)
    heartbeat = []
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    getattr(dialog, {"probe": "probe_ports", "preview": "preview_now", "capture": "capture_now"}[operation])()
    try:
        qtbot.waitUntil(entered.is_set)
        assert len(threads) == 1
        assert threads[0] != gui_thread
        qtbot.waitUntil(lambda: bool(heartbeat))
        assert not release.is_set()
        assert not dialog.btn_probe.isEnabled()
        assert not dialog.btn_capture.isEnabled()
        assert not dialog.btn_preview.isEnabled()
        assert dialog.btn_cancel.isEnabled()
        dialog.probe_ports()
        dialog.capture_now()
        assert len(threads) == 1
    finally:
        release.set()
    qtbot.waitUntil(lambda: dialog._worker is None)


@pytest.mark.parametrize("operation", ["probe", "preview", "capture"])
def test_close_during_device_wait_discards_late_results(qtbot, monkeypatch, device, operation):
    dialog, path, entered, release, _threads, wait = device

    def slow_probe(**_kwargs):
        wait()
        return [0, 2]

    def slow_capture(_index, _directory):
        wait()
        return path

    monkeypatch.setattr(camera, "probe_camera_indices", slow_probe)
    monkeypatch.setattr(camera, "capture_device_frame", slow_capture)
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    getattr(dialog, {"probe": "probe_ports", "preview": "preview_now", "capture": "capture_now"}[operation])()
    qtbot.waitUntil(entered.is_set)
    dialog.reject()
    assert not dialog.isVisible()
    release.set()
    qtbot.waitUntil(lambda: getattr(dialog, "_worker", None) is None)
    assert accepted == []
    assert dialog.captured_path is None
    assert dialog.lbl_probe_info.text() == ""
    if operation == "capture":
        assert not path.exists()


def test_probe_failure_is_displayed_and_retry_works(qtbot, monkeypatch, device):
    dialog, *_ = device

    def fail(**_kwargs):
        raise RuntimeError("camera driver unavailable")

    monkeypatch.setattr(camera, "probe_camera_indices", fail)
    dialog.probe_ports()
    qtbot.waitUntil(lambda: "camera driver unavailable" in dialog.lbl_err.text())
    qtbot.waitUntil(lambda: dialog.btn_probe.isEnabled())
    monkeypatch.setattr(camera, "probe_camera_indices", lambda **_kwargs: [0, 2])
    dialog.probe_ports()
    qtbot.waitUntil(lambda: "[0, 2]" in dialog.lbl_probe_info.text())
    assert dialog.lbl_err.text() == ""
    assert dialog.spin_usb.value() == 2


def test_preview_diagnostic_failure_is_displayed(qtbot, monkeypatch, device):
    dialog, path, *_ = device
    monkeypatch.setattr(camera, "capture_device_frame", lambda *_args: path)

    def fail(*_args, **_kwargs):
        raise RuntimeError("preview diagnostics unavailable")

    monkeypatch.setattr(camera, "analyze_camera_frame_ipkit", fail)
    dialog.preview_now()
    qtbot.waitUntil(lambda: "preview diagnostics unavailable" in dialog.lbl_err.text())
    qtbot.waitUntil(lambda: dialog.btn_preview.isEnabled())


def test_invalid_preview_is_reported(qtbot, monkeypatch, device):
    dialog, path, *_ = device
    path.write_bytes(b"unreadable image")
    monkeypatch.setattr(camera, "capture_device_frame", lambda *_args: path)
    dialog.preview_now()
    qtbot.waitUntil(lambda: bool(dialog.lbl_err.text()))
    qtbot.waitUntil(lambda: dialog.btn_preview.isEnabled())


def test_capture_uses_source_values_from_dispatch(qtbot, monkeypatch, device):
    dialog, path, entered, release, threads, wait = device
    requests = []

    def slow_iot(url, directory):
        requests.append((url, directory))
        wait()
        return path

    monkeypatch.setattr(camera, "capture_iot_http_frame", slow_iot)
    dialog.rad_iot.setChecked(True)
    dialog.edit_url.setText("http://camera.test/first.jpg")
    accepted_threads = []
    dialog.accepted.connect(lambda: accepted_threads.append(threading.get_ident()))
    dialog.capture_now()
    try:
        qtbot.waitUntil(entered.is_set)
        dialog.edit_url.setText("http://camera.test/second.jpg")
        assert requests == [("http://camera.test/first.jpg", path.parent)]
    finally:
        release.set()
    qtbot.waitUntil(lambda: dialog._worker is None)
    assert dialog.captured_path == path
    qtbot.mouseClick(dialog.btn_accept_frame, Qt.MouseButton.LeftButton)
    assert accepted_threads == [threading.get_ident()]
    assert threads[0] != accepted_threads[0]


def test_reopened_dialog_ignores_cancelled_capture_and_allows_new_probe(qtbot, monkeypatch, device):
    dialog, path, entered, release, _threads, wait = device

    def slow_capture(*_args):
        wait()
        return path

    monkeypatch.setattr(camera, "capture_device_frame", slow_capture)
    dialog.capture_now()
    qtbot.waitUntil(entered.is_set)
    dialog.close()
    dialog.show()
    release.set()
    qtbot.waitUntil(lambda: dialog._worker is None)
    assert dialog.isVisible()
    assert dialog.captured_path is None
    assert dialog.btn_capture.isEnabled()
    monkeypatch.setattr(camera, "probe_camera_indices", lambda **_kwargs: [1])
    dialog.probe_ports()
    qtbot.waitUntil(lambda: dialog._worker is None)
    assert "[1]" in dialog.lbl_probe_info.text()


@pytest.mark.parametrize("operation", ["probe", "capture"])
def test_message_free_device_error_cannot_signal_success(qtbot, monkeypatch, device, operation):
    dialog, *_ = device

    def fail(*_args, **_kwargs):
        raise RuntimeError()

    monkeypatch.setattr(camera, "probe_camera_indices", fail)
    monkeypatch.setattr(camera, "capture_device_frame", fail)
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    getattr(dialog, "probe_ports" if operation == "probe" else "capture_now")()
    qtbot.waitUntil(lambda: dialog._worker is None)
    assert accepted == []
    assert dialog.isVisible()
    assert dialog.captured_path is None
    assert "RuntimeError" in dialog.lbl_err.text()
    assert dialog.btn_capture.isEnabled()


def test_deepfake_formatting_runs_off_gui_thread(qtbot, monkeypatch, device):
    dialog, path, *_ = device
    gui_thread = threading.get_ident()
    seen = []
    monkeypatch.setattr(camera, "capture_device_frame", lambda *_args: path)

    def fake_format(frame):
        seen.append(threading.get_ident())
        return "deepfake text"

    monkeypatch.setattr(camera, "format_deepfake_preview", fake_format)
    dialog.capture_now()
    qtbot.waitUntil(lambda: dialog.captured_path == path, timeout=5000)
    qtbot.waitUntil(lambda: dialog.lbl_probe_info.text() == "deepfake text", timeout=5000)
    assert seen and seen[0] != gui_thread


def test_discard_capture_artifacts_removes_unaccepted_burst(tmp_path):
    names = [f"burst_{i}.png" for i in range(4)]
    payloads = {name: f"png-{name}".encode() for name in names}
    import hashlib
    import json

    sha_map = {}
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
        sha_map[name] = hashlib.sha256(payload).hexdigest()
    primary = tmp_path / names[2]
    manifest = tmp_path / f"{primary.stem}.burst.json"
    manifest.write_text(
        json.dumps({"frames": names, "sha256": sha_map}), encoding="utf-8"
    )

    camera.discard_capture_artifacts(primary, tmp_path)

    assert not manifest.exists()
    assert all(not (tmp_path / name).exists() for name in names)


def test_discard_capture_artifacts_keeps_frame_missing_hash(tmp_path):
    import json

    primary = tmp_path / "primary.png"
    other = tmp_path / "other.png"
    primary.write_bytes(b"primary")
    other.write_bytes(b"other")
    manifest = tmp_path / "primary.burst.json"
    manifest.write_text(
        json.dumps({"frames": ["other.png", "primary.png"], "sha256": {}}),
        encoding="utf-8",
    )

    camera.discard_capture_artifacts(primary, tmp_path)

    assert not primary.exists()
    assert not manifest.exists()
    assert other.exists()
