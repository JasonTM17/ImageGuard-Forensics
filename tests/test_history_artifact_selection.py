"""The selected artifact must own the pixels shown in the history dialog."""

import numpy as np
import pytest

from imageguard.services.history_service import HistoryService
from imageguard.storage.artifacts import ArtifactStore
from imageguard.ui.screens.history import AnalysisDetailDialog


@pytest.mark.parametrize("failure", ["missing", "corrupt", "refused", "nonimage"])
def test_failed_artifact_selection_clears_previous_pixels_and_can_retry(qtbot, tmp_path, failure):
    store = ArtifactStore(tmp_path / "art")
    store.put_image("r1", "first.png", np.full((64, 80, 3), 30, np.uint8))
    store.put_image("r1", "last.png", np.full((70, 90, 3), 200, np.uint8))
    failing = {"missing": "r1/missing.png", "corrupt": "r1/corrupt.png",
               "refused": "../../outside.png", "nonimage": "r1/report.json"}[failure]
    store.put_bytes("r1", "corrupt.png", b"invalid-image")
    detail = {"id": "r1", "image_name": "fixture.png", "created_at": "2026-11-17",
              "verdict": "inconclusive", "duration_ms": 1, "config_hash": "x", "methods": [],
              "artifacts": [{"kind": "mask", "method": "copy_move_block", "rel_path": rel}
                            for rel in ("r1/first.png", failing, "r1/last.png")]}
    dialog = AnalysisDetailDialog(detail, HistoryService(None, store))
    qtbot.addWidget(dialog)
    dialog.combo.setCurrentIndex(1)
    assert dialog.canvas._base is not None
    assert dialog.canvas._base.pixmap().width() == 80
    dialog.combo.setCurrentIndex(2)
    assert dialog.canvas._base is None
    if failure != "nonimage":
        assert dialog.lbl_art_err.text()
    dialog.combo.setCurrentIndex(3)
    assert dialog.lbl_art_err.text() == ""
    assert dialog.canvas._base.pixmap().width() == 90
    dialog.combo.setCurrentIndex(0)
    assert dialog.canvas._base is None
