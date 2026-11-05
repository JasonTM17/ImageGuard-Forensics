"""App entry — wire services + QApplication."""

from __future__ import annotations

import sys
from pathlib import Path

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


def main(initial_image: Path | str | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    analysis, history = build_services()
    w = MainWindow(analysis, history)

    img_to_load = initial_image
    if img_to_load is None:
        for arg in sys.argv[1:]:
            if not arg.startswith("-"):
                cand = Path(arg)
                if cand.is_file():
                    img_to_load = cand
                    break

    if img_to_load is not None:
        p = Path(img_to_load)
        if p.is_file():
            w.analyze.load_path(p)

    w.show()
    return app.exec()
