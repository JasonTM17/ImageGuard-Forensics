"""Worker bridge — QThreadPool + QRunnable + signals.

RULE CỨNG: worker KHÔNG chạm QWidget. Mọi kết quả đi qua signal; GUI thread
là nơi duy nhất update widget. Analysis chạy serialized (maxThreadCount=1)
để kiểm soát RAM; cancel = cooperative threading.Event.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

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
        except Exception as e:  # không bao giờ để exception chết thread
            self.signals.failed.emit(str(e))


def make_pool() -> QThreadPool:
    pool = QThreadPool()
    pool.setMaxThreadCount(1)  # serialize — một analysis tại một thời điểm
    return pool
