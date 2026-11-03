"""App entry — wire services + QApplication."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from imageguard.core.config import load_thresholds
from imageguard.services.analysis_service import AnalysisService
from imageguard.services.history_service import HistoryService
from imageguard.storage.artifacts import ArtifactStore
from imageguard.storage.db import make_session_factory
from imageguard.storage.repository import AnalysisRepository, ImageRepository
from imageguard.ui.main_window import MainWindow


def build_services(db_path=None, artifacts_root=None):
    cfg = load_thresholds()
    kwargs = {}
    if db_path is not None:
        kwargs["db_path"] = db_path
    sf = make_session_factory(**kwargs)
    store = ArtifactStore(artifacts_root) if artifacts_root else ArtifactStore()
    analyses = AnalysisRepository(sf)
    analysis = AnalysisService(cfg, ImageRepository(sf), analyses, store)
    history = HistoryService(analyses, store)
    return analysis, history


def main() -> int:
    app = QApplication(sys.argv)
    analysis, history = build_services()
    w = MainWindow(analysis, history)
    w.show()
    return app.exec()
