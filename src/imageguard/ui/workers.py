"""Worker bridge — QThreadPool + QRunnable + signals.

RULE CỨNG: worker KHÔNG chạm QWidget. Mọi kết quả đi qua signal; GUI thread
là nơi duy nhất update widget. Analysis chạy serialized (maxThreadCount=1)
để kiểm soát RAM; cancel = cooperative threading.Event.
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from imageguard.enhancement.derive import derive
from imageguard.services.analysis_service import AnalysisRequest, AnalysisService


class WorkerSignals(QObject):
    progress = Signal(float, str)
    result = Signal(object)   # ForensicReport | ErrorResult
    failed = Signal(str)


class AnalysisWorker(QRunnable):
    def __init__(self, service: AnalysisService, req: AnalysisRequest) -> None:
        super().__init__()
        self._svc = service
        self._req = req
        self.signals = WorkerSignals()
        if self._req.cancel is None:
            self._req.cancel = threading.Event()
        self._req.progress_cb = self._on_progress

    def _on_progress(self, frac: float, stage: str) -> None:
        self.signals.progress.emit(frac, stage)

    def cancel(self) -> None:
        if self._req.cancel is not None:
            self._req.cancel.set()

    def run(self) -> None:
        try:
            out = self._svc.analyze(self._req)
            self.signals.result.emit(out)
        except Exception as e:  # noqa: BLE001 - không bao giờ để exception chết thread
            self.signals.failed.emit(str(e))


class EnhanceSignals(QObject):
    result = Signal(object)   # (np.ndarray, dict) từ derive
    failed = Signal(str)
    finished = Signal()       # luôn emit — kể cả khi cancel


class EnhanceWorker(QRunnable):
    """Chạy một operator tăng cường — KHÔNG chạm QWidget, KHÔNG gọi
    AnalysisService.analyze. Cancel là cooperative threading.Event."""

    def __init__(
        self,
        image: np.ndarray,
        kind: str,
        params: dict,
        cancel: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self._image = image
        self._kind = kind
        self._params = dict(params)
        self._cancel = cancel if cancel is not None else threading.Event()
        self.signals = EnhanceSignals()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            out, meta = derive(self._image, self._kind, self._params)
            if not self._cancel.is_set():
                self.signals.result.emit((out, meta))
        except Exception as e:  # noqa: BLE001 - không bao giờ để exception chết thread
            if not self._cancel.is_set():
                self.signals.failed.emit(str(e))
        finally:
            self.signals.finished.emit()


def make_pool() -> QThreadPool:
    pool = QThreadPool()
    pool.setMaxThreadCount(1)  # serialize — một analysis tại một thời điểm
    return pool
