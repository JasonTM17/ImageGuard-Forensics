"""Actual Qt workers stay bounded across cancellation, shutdown and repeat runs."""

from __future__ import annotations

import gc
import threading
import weakref
from queue import Queue

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox

from imageguard.forensics.auxiliary.deepfake_model import save_model
from imageguard.ingest.session_reference import SessionReferenceStore
from imageguard.ui import strings
from imageguard.ui.app import build_services
from imageguard.ui.main_window import MainWindow
from imageguard.ui.widgets.verdict_panel import MethodCard
from imageguard.ui.workers import AnalysisWorker
from tests.test_deepfake_model_ui import ui_model  # noqa: F401 - shared pytest fixture


@pytest.fixture()
def window(qtbot, tmp_path, monkeypatch):
    reference = SessionReferenceStore(root=tmp_path / "reference")
    monkeypatch.setattr("imageguard.ui.screens.analyze.SessionReferenceStore", lambda: reference)
    service, history = build_services(db_path=tmp_path / "history.db", artifacts_root=tmp_path / "artifacts")
    widget = MainWindow(service, history)
    qtbot.addWidget(widget)
    widget.show()
    return widget


def test_retry_after_cancel_clears_old_status(qtbot, window, valid_png, monkeypatch):
    screen = window.analyze
    original = screen._svc.analyze
    started = threading.Event()
    release = threading.Event()

    def delayed(req):
        started.set()
        assert release.wait(5), "test did not release the worker"
        return original(req)

    monkeypatch.setattr(screen._svc, "analyze", delayed)
    screen.load_path(valid_png)
    screen._apply_method_preset(["ela"])
    try:
        screen.btn_run.click()
        qtbot.waitUntil(started.is_set, timeout=5000)
        screen.btn_cancel.click()
        release.set()
        qtbot.waitUntil(lambda: screen._worker is None, timeout=10000)
        assert screen.lbl_status.text() == strings.STATUS_ANALYZE_CANCELLED

        # Reuse the same image, a common retry: load_path must not be needed to
        # clear the last run's cancellation message.
        started.clear()
        release.clear()
        screen.btn_run.click()
        qtbot.waitUntil(started.is_set, timeout=5000)
        assert not screen.lbl_status.text()
        release.set()
        qtbot.waitUntil(lambda: screen._worker is None, timeout=10000)
        assert screen._report is not None
        assert not screen.lbl_status.text()
        assert not any(result.evidence.get("error") == "cancelled" for result in screen._report.methods)
    finally:
        release.set()
        assert screen._pool.waitForDone(10000)


def test_repeated_cancel_image_model_history_and_export_cycle(
    qtbot, window, valid_png, ui_model, tmp_path, monkeypatch,  # noqa: F811 - injected fixture
):
    """Twelve real persisted runs, with bounded objects and no stale export."""
    model_path = tmp_path / "model.json"
    save_model(model_path, ui_model)
    second_path = tmp_path / "another.png"
    pixels = np.random.default_rng(91).integers(0, 255, (192, 320, 3), dtype=np.uint8)
    second_path.write_bytes(cv2.imencode(".png", pixels)[1].tobytes())
    paths = (valid_png, second_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: (str(model_path), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: QMessageBox.StandardButton.Ok)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: warnings.append(_args[-1]))

    screen = window.analyze
    original = screen._svc.analyze
    gates = Queue()

    def delayed(req):
        started, release = gates.get(timeout=2)
        started.set()
        assert release.wait(5), "test did not release the worker"
        return original(req)

    monkeypatch.setattr(screen._svc, "analyze", delayed)
    screen._apply_method_preset(["ela", "deepfake"])
    report_ids = set()
    image_refs = []
    worker_refs = []
    release = threading.Event()
    try:
        for iteration in range(12):
            image_path = paths[iteration % 2]
            screen.load_path(image_path)
            if iteration % 3 == 0:
                screen.btn_deepfake_model.click()
            elif iteration % 3 == 2:
                screen.btn_deepfake_model_clear.click()
            assert screen._report is None
            assert screen.cards_layout.count() == 0
            assert screen.btn_export.isHidden()
            expected_hash = screen._doc.sha256
            selected_model = screen._deepfake_model_sha256
            image_refs.append(weakref.ref(screen._doc.array))
            started, release = threading.Event(), threading.Event()
            gates.put((started, release))
            screen.btn_run.click()
            worker_refs.append(weakref.ref(screen._worker))
            qtbot.waitUntil(started.is_set, timeout=5000)

            # A second run or image/model switch must not enqueue another job
            # or change the source associated with the pending report.
            screen._run()
            screen.load_path(paths[(iteration + 1) % 2])
            screen._clear_deepfake_model()
            assert screen._path == image_path
            assert screen._deepfake_model_sha256 == selected_model
            assert screen._pool.activeThreadCount() == 1
            cancelled = iteration % 3 == 1
            if cancelled:
                screen.btn_cancel.click()
            release.set()
            qtbot.waitUntil(lambda: screen._worker is None, timeout=10000)

            report = screen._report
            assert report is not None and report.image_sha256 == expected_hash
            assert report.report_id not in report_ids
            report_ids.add(report.report_id)
            assert screen.btn_run.isEnabled()
            assert screen.lbl_status.text() == (strings.STATUS_ANALYZE_CANCELLED if cancelled else "")
            assert any(result.evidence.get("error") == "cancelled" for result in report.methods) is cancelled
            method_count = len(report.methods)
            qtbot.waitUntil(
                lambda count=method_count: len(screen.cards_inner.findChildren(MethodCard)) == count,
                timeout=2000,
            )
            if not cancelled:
                if screen.overlay_box.count() > 1:
                    screen.overlay_box.setCurrentIndex(1)
                export_path = tmp_path / f"report_{iteration}.md"
                monkeypatch.setattr(
                    QFileDialog, "getSaveFileName", lambda *_args, p=export_path: (str(p), ""),
                )
                screen.btn_export.click()
                assert not warnings
                exported = export_path.read_text(encoding="utf-8")
                assert report.report_id in exported and image_path.name in exported

            window.tabs.setCurrentWidget(window.history)
            assert window.history.table.rowCount() == iteration + 1
            saved = window.history._svc.get(report.report_id)
            assert saved is not None and saved["image_name"] == image_path.name
            window.tabs.setCurrentWidget(screen)

        assert {row["id"] for row in window.history._svc.list()} == report_ids
        # Drop the last report/image and flush deferred Qt card deletions. No
        # earlier image buffer or completed QRunnable should be retained.
        screen.load_path(valid_png)
        qtbot.waitUntil(lambda: not screen.cards_inner.findChildren(MethodCard), timeout=2000)
        assert screen._pool.waitForDone(10000)
        report = None
        gc.collect()
        assert all(reference() is None for reference in image_refs)
        assert all(reference() is None for reference in worker_refs)
    finally:
        release.set()
        assert screen._pool.waitForDone(10000)


@pytest.mark.parametrize("kind", ["analysis", "enhancement"])
def test_close_requests_cancel_and_waits_without_blocking_gui(
    qtbot, window, valid_png, monkeypatch, kind,
):
    started = threading.Event()
    release = threading.Event()
    if kind == "analysis":
        screen = window.analyze
        original = screen._svc.analyze

        def delayed(req):
            started.set()
            assert release.wait(5), "test did not release the worker"
            return original(req)

        monkeypatch.setattr(screen._svc, "analyze", delayed)
        screen._apply_method_preset(["ela"])
    else:
        from imageguard.ui import workers

        screen = window.enhance
        original = workers.derive

        def delayed(*args):
            started.set()
            assert release.wait(5), "test did not release the worker"
            return original(*args)

        monkeypatch.setattr(workers, "derive", delayed)

    screen.load_path(valid_png)
    try:
        screen.btn_run.click()
        qtbot.waitUntil(started.is_set, timeout=5000)
        assert not screen._cancel_event.is_set()
        assert screen._pool.activeThreadCount() == 1
        window.close()

        assert screen._cancel_event.is_set()
        assert window.isVisible(), "close must wait for the worker's safe exit"
        assert not window.tabs.isEnabled()
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        qtbot.waitUntil(lambda: bool(heartbeat), timeout=1000)
        assert screen._pool.activeThreadCount() == 1

        # Finish a simulated noninterruptible phase, then allow the real
        # analysis/operator and its result/finished signals to run.
        release.set()
        qtbot.waitUntil(lambda: not window.isVisible(), timeout=10000)
        assert screen._pool.activeThreadCount() == 0
        assert screen._worker is None
        if kind == "enhancement":
            assert screen._derived is None
        window.show()
        assert window.tabs.isEnabled()
        assert screen.btn_run.isEnabled()
    finally:
        release.set()
        assert screen._pool.waitForDone(10000)


def test_close_waits_for_runnable_after_result_already_reached_gui(
    qtbot, window, valid_png, monkeypatch,
):
    """Result delivery alone must not be mistaken for the pool being drained."""
    exiting = threading.Event()
    release = threading.Event()
    original = AnalysisWorker.run

    def delayed_exit(worker):
        original(worker)
        exiting.set()
        assert release.wait(5), "test did not release the runnable"

    monkeypatch.setattr(AnalysisWorker, "run", delayed_exit)
    screen = window.analyze
    screen.load_path(valid_png)
    screen._apply_method_preset(["ela"])
    try:
        screen.btn_run.click()
        qtbot.waitUntil(lambda: exiting.is_set() and screen._worker is None, timeout=10000)
        assert screen._report is not None
        assert screen._pool.activeThreadCount() == 1
        window.close()
        assert window.isVisible()
        assert not window.tabs.isEnabled()
        release.set()
        qtbot.waitUntil(lambda: not window.isVisible(), timeout=10000)
        assert screen._pool.activeThreadCount() == 0
    finally:
        release.set()
        assert screen._pool.waitForDone(10000)
