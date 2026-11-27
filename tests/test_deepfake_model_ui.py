"""Model selection and rendering contracts; synthetic rows are not accuracy data."""

from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog, QLabel

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ForensicReport, ImageDocument
from imageguard.evaluation.deepfake_training import fit_model
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues
from imageguard.forensics.auxiliary.deepfake_model import (
    EXTRACTOR_VARIANT,
    FEATURE_NAMES,
    config_digest,
    save_model,
)
from imageguard.forensics.auxiliary.learned_deepfake import LearnedDeepfakeCues
from imageguard.ingest.session_reference import SessionReferenceStore
from imageguard.services.analysis_service import AnalysisRequest, AnalysisService
from imageguard.ui import strings
from imageguard.ui.screens.analyze import AnalyzeScreen
from imageguard.ui.widgets.verdict_panel import MethodCard


@pytest.fixture(scope="module")
def ui_model():
    records = []
    for group in range(3):
        for label, level in (("real", 0.1), ("fake", 0.9)):
            sample_id = f"ui_{group}_{label}"
            records.append({
                "id": sample_id, "source_group": f"video_{group}",
                "label": label, "split": "dev",
                "sha256": hashlib.sha256(sample_id.encode()).hexdigest(),
                "features": dict.fromkeys(FEATURE_NAMES, level + group * 0.01),
            })
    model, _ = fit_model(
        records, config_sha256=config_digest(load_thresholds().raw),
        manifest_sha256="b" * 64, folds=3,
    )
    return model


@pytest.fixture()
def model_path(tmp_path, ui_model):
    path = tmp_path / "mô_hình_DIP.json"
    save_model(path, ui_model)
    return path


@pytest.fixture()
def screen(qtbot, monkeypatch, tmp_path):
    store = SessionReferenceStore(root=tmp_path / "reference")
    monkeypatch.setattr("imageguard.ui.screens.analyze.SessionReferenceStore", lambda: store)
    widget = AnalyzeScreen(AnalysisService())
    qtbot.addWidget(widget)
    return widget


def _choose(screen, monkeypatch, path):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: (str(path), ""))
    screen.btn_deepfake_model.click()


def _show_report(screen, image_path):
    screen.load_path(image_path)
    report = screen._svc.analyze(AnalysisRequest(
        path=image_path, methods=["deepfake"], include_exif=False,
        deepfake_model_path=screen._deepfake_model_path,
        deepfake_model_sha256=screen._deepfake_model_sha256,
    ))
    assert isinstance(report, ForensicReport)
    screen._on_result(report)
    assert screen._report is report
    assert not screen.btn_export.isHidden()
    assert screen.cards_layout.count() > 0
    return report


def _assert_results_cleared(screen):
    assert screen._report is None
    assert screen.banner.isHidden()
    assert screen.btn_export.isHidden()
    assert screen.overlay_box.isHidden()
    assert screen.cards_layout.count() == 0


def test_select_valid_model_invalidates_previous_result(
    screen, valid_png, model_path, ui_model, monkeypatch,
):
    _show_report(screen, valid_png)
    screen.chk_df.setChecked(False)
    _choose(screen, monkeypatch, model_path)

    assert screen._deepfake_model_path == model_path
    assert screen._deepfake_model_sha256 == ui_model["model_sha256"]
    assert model_path.name in screen.lbl_deepfake_model.text()
    assert str(model_path.parent) not in screen.lbl_deepfake_model.text()
    assert screen.chk_df.isChecked()
    assert screen.btn_deepfake_model_clear.isEnabled()
    assert not screen.lbl_err.text()
    assert screen._path == valid_png and screen.btn_run.isEnabled()
    _assert_results_cleared(screen)


@pytest.mark.parametrize("invalid_kind", ["malformed", "config_mismatch"])
def test_rejected_model_keeps_previous_selection_and_result(
    screen, valid_png, model_path, ui_model, tmp_path, monkeypatch, invalid_kind,
):
    _choose(screen, monkeypatch, model_path)
    previous = _show_report(screen, valid_png)
    invalid_path = tmp_path / "invalid.json"
    if invalid_kind == "malformed":
        invalid_path.write_text("{", encoding="utf-8")
    else:
        incompatible = copy.deepcopy(ui_model)
        incompatible["config_sha256"] = "0" * 64
        incompatible.pop("model_sha256")
        incompatible["model_sha256"] = config_digest(incompatible)
        save_model(invalid_path, incompatible)

    _choose(screen, monkeypatch, invalid_path)

    assert screen._deepfake_model_path == model_path
    assert screen._deepfake_model_sha256 == ui_model["model_sha256"]
    assert screen._report is previous
    assert not screen.btn_export.isHidden()
    assert screen.btn_deepfake_model_clear.isEnabled()
    assert screen.lbl_err.text().startswith("Không dùng được mô hình:")
    if invalid_kind == "config_mismatch":
        assert "configuration" in screen.lbl_err.text()
    assert str(tmp_path) not in screen.lbl_err.text()


def test_clear_model_invalidates_result_and_keeps_loaded_image(
    screen, valid_png, model_path, monkeypatch,
):
    _choose(screen, monkeypatch, model_path)
    _show_report(screen, valid_png)
    document = screen._doc
    screen.btn_deepfake_model_clear.click()

    assert screen._deepfake_model_path is None
    assert screen._deepfake_model_sha256 is None
    assert screen.lbl_deepfake_model.text() == strings.DEEPFAKE_MODEL_DEFAULT
    assert not screen.btn_deepfake_model_clear.isEnabled()
    assert screen._doc is document and screen._path == valid_png
    assert screen.btn_run.isEnabled()
    _assert_results_cleared(screen)


def test_running_worker_pins_model_and_locks_selection(
    screen, valid_png, model_path, ui_model, monkeypatch,
):
    screen.load_path(valid_png)
    _choose(screen, monkeypatch, model_path)
    screen._apply_method_preset(["deepfake"])
    pending = []
    screen._pool = SimpleNamespace(start=pending.append)
    screen.btn_run.click()

    assert len(pending) == 1
    worker = pending[0]
    assert worker._req.path == valid_png
    assert worker._req.methods == ["deepfake"]
    assert worker._req.deepfake_model_path == model_path
    assert worker._req.deepfake_model_sha256 == ui_model["model_sha256"]
    assert not screen.btn_deepfake_model.isEnabled()
    assert not screen.btn_deepfake_model_clear.isEnabled()

    dialogs = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: dialogs.append(True))
    screen.btn_deepfake_model.click()
    screen._pick_deepfake_model()
    screen._clear_deepfake_model()
    assert not dialogs
    assert screen._deepfake_model_path == model_path
    assert screen._deepfake_model_sha256 == ui_model["model_sha256"]

    # Complete the real worker synchronously after observing its pending state.
    worker.run()
    assert screen._worker is None
    assert screen.btn_deepfake_model.isEnabled()
    assert screen.btn_deepfake_model_clear.isEnabled()
    result = screen._report.methods[0]
    assert result.evidence["decision_source"] == "learned_dip"
    assert result.evidence["model_sha256"] == ui_model["model_sha256"]


@pytest.mark.parametrize("state", ["applied", "unavailable", "training_overlap"])
def test_card_explains_actual_model_result(qtbot, ui_model, monkeypatch, state):
    unavailable = state == "unavailable"
    base = AlgorithmResult(
        method=MethodName.DEEPFAKE_CUES,
        verdict=Verdict.INCONCLUSIVE if unavailable else Verdict.NO_INDICATION,
        score=None if unavailable else 0.2,
        score_definition="synthetic heuristic control for UI state",
        evidence={
            **dict.fromkeys(FEATURE_NAMES, 0.9), "variant": EXTRACTOR_VARIANT,
            "face_detected": not unavailable, "cue_unavailable": unavailable,
            "cue_flag": False, "unavailable_reason": "no_face" if unavailable else None,
        },
    )
    monkeypatch.setattr(DeepfakeCues, "analyze", lambda *_args: base)
    sha = ui_model["training"]["samples"][0]["sha256"] if state == "training_overlap" else "c" * 64
    doc = ImageDocument(np.zeros((96, 96, 3), np.uint8), 96, 96, 3, "png", sha, 1, "test.png")
    result = LearnedDeepfakeCues(ui_model).analyze(doc, load_thresholds())
    card = MethodCard(result)
    qtbot.addWidget(card)
    text = "\n".join(label.text() for label in card.findChildren(QLabel))

    if unavailable:
        assert strings.DEEPFAKE_MODEL_UNAVAILABLE in text
        assert strings.DEEPFAKE_MODEL_APPLIED not in text
        assert strings.DEEPFAKE_CUE_UNAVAILABLE in text
        assert strings.DEEPFAKE_CUE_FLAG_OFF not in text
        assert "Điểm dấu hiệu:" not in text
    else:
        assert strings.DEEPFAKE_MODEL_APPLIED in text
        assert strings.DEEPFAKE_MODEL_UNAVAILABLE not in text
        assert strings.DEEPFAKE_CUE_FLAG_ON in text
        assert "không phải xác suất" in text
        assert strings.DEEPFAKE_PREVIEW_SCORE.format(score=result.score) in text
    assert (strings.DEEPFAKE_MODEL_TRAINING_OVERLAP in text) is (state == "training_overlap")
