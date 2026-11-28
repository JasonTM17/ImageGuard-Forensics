"""User-facing deepfake screening path: method-set wiring + labeled CoC pair.

Drives shipped ``AnalysisService`` / registry methods. Does not reimplement
detectors or hard-code expected scores. Skip (not pass) when local ONNX
weights, onnxruntime, or labeled pilot images are absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.ui import strings
from imageguard.ui.method_presets import (
    ALL_ANALYZE_METHODS,
    CAMERA_AI_METHODS,
    DEEPFAKE_SCREENING_METHODS,
    DEFAULT_ANALYZE_METHODS,
)

FAKE_PATH = Path("data/deepfake-coc-pilot/case-03.png")
REAL_PATH = Path("data/deepfake-coc-pilot/case-01.png")

_SCREENING = {"deepfake_ml", "deepfake_screen", "deepfake_ensemble"}


def _screening_blocker() -> str | None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return "onnxruntime not installed"
    if not FAKE_PATH.is_file() or not REAL_PATH.is_file():
        return "labeled CoC pilot images not installed"
    from imageguard.forensics.ml import deepfake_genconvit as dgc
    from imageguard.forensics.ml import deepfake_screen_coc as dsc
    from imageguard.forensics.ml.face_detect import yunet_integrity_error

    if dsc.model_integrity_error() is not None:
        return "CoC ONNX missing or integrity mismatch"
    if dgc.model_integrity_error() is not None:
        return "GenConViT ONNX missing or integrity mismatch"
    if yunet_integrity_error() is not None:
        return "YuNet ONNX missing or integrity mismatch"
    return None


needs_screening = pytest.mark.skipif(
    _screening_blocker() is not None,
    reason=_screening_blocker() or "screening assets unavailable",
)


def test_camera_ai_preset_includes_suspicious_capable_detector() -> None:
    assert _SCREENING & set(CAMERA_AI_METHODS)
    assert set(DEEPFAKE_SCREENING_METHODS) <= set(CAMERA_AI_METHODS)
    assert "deepfake" in CAMERA_AI_METHODS
    assert set(CAMERA_AI_METHODS) <= set(ALL_ANALYZE_METHODS)
    assert "noise" in ALL_ANALYZE_METHODS
    assert "noise" not in CAMERA_AI_METHODS
    assert DEFAULT_ANALYZE_METHODS == ("keypoint", "block", "ela")


def test_demo_deepfake_sample_recommends_screening_methods() -> None:
    from imageguard.ui.widgets.sample_dialog import _DEMO_SPECS

    methods = set(_DEMO_SPECS["deepfake"][1])
    assert _SCREENING <= methods
    assert "deepfake" in methods
    assert "block" not in methods


def test_method_card_shows_screen_score_and_verdict(qtbot) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from imageguard.ui.widgets.verdict_panel import MethodCard

    res = AlgorithmResult(
        method=MethodName.DEEPFAKE_SCREEN_COC,
        verdict=Verdict.SUSPICIOUS_INDICATION,
        score=0.9986,
        score_definition="fake_prob = softmax(logits)[fake]",
        evidence={"face_detected": True, "fake_prob": 0.9986},
        limitations=["classifier screen-recapture ghi nhận dấu hiệu mạnh"],
    )
    card = MethodCard(res)
    qtbot.addWidget(card)
    from PySide6.QtWidgets import QLabel

    shown = " ".join(lbl.text() for lbl in card.findChildren(QLabel))
    assert "0.9986" in shown
    assert strings.VERDICT_LABEL["suspicious_indication"] in shown


@needs_screening
def test_analysis_service_flags_local_fake_and_clears_real() -> None:
    from imageguard.services.analysis_service import AnalysisRequest, AnalysisService

    svc = AnalysisService()
    methods = list(DEEPFAKE_SCREENING_METHODS)

    fake_report = svc.analyze(
        AnalysisRequest(path=FAKE_PATH, methods=methods, include_exif=False)
    )
    real_report = svc.analyze(
        AnalysisRequest(path=REAL_PATH, methods=methods, include_exif=False)
    )

    fake_hits = [
        r
        for r in fake_report.methods
        if r.method.value in {
            MethodName.DEEPFAKE_ML.value,
            MethodName.DEEPFAKE_SCREEN_COC.value,
            MethodName.DEEPFAKE_ENSEMBLE.value,
        }
        and r.verdict is Verdict.SUSPICIOUS_INDICATION
        and r.score is not None
        and r.score == r.score
    ]
    assert fake_hits, (
        "screening path must emit SUSPICIOUS_INDICATION with a finite score "
        f"on labeled fake; got {[ (r.method.value, r.verdict.value, r.score) for r in fake_report.methods ]}"
    )
    assert fake_report.overall_verdict is Verdict.SUSPICIOUS_INDICATION

    real_hits = [
        r
        for r in real_report.methods
        if r.verdict is Verdict.SUSPICIOUS_INDICATION
    ]
    assert not real_hits, (
        "screening path must not flag labeled real; got "
        f"{[ (r.method.value, r.verdict.value, r.score) for r in real_report.methods ]}"
    )
    assert real_report.overall_verdict is not Verdict.SUSPICIOUS_INDICATION


@needs_screening
def test_default_copy_move_methods_do_not_flag_labeled_deepfake() -> None:
    from imageguard.services.analysis_service import AnalysisRequest, AnalysisService

    report = AnalysisService().analyze(
        AnalysisRequest(
            path=FAKE_PATH,
            methods=list(DEFAULT_ANALYZE_METHODS),
            include_exif=False,
        )
    )
    assert report.overall_verdict is not Verdict.SUSPICIOUS_INDICATION


@needs_screening
def test_gui_face_open_method_set_flags_fake_and_clears_real(qtbot) -> None:
    """GUI load_path on a face enables screening; that exact set is what Analyze sends."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from imageguard.services.analysis_service import AnalysisRequest, AnalysisService
    from imageguard.ui.screens.analyze import AnalyzeScreen

    svc = AnalysisService()
    screen = AnalyzeScreen(svc)
    qtbot.addWidget(screen)
    screen.btn_preset_default.click()
    screen.load_path(FAKE_PATH)
    methods = screen._methods()
    assert set(DEFAULT_ANALYZE_METHODS) <= set(methods)
    assert set(DEEPFAKE_SCREENING_METHODS) <= set(methods)

    fake = svc.analyze(
        AnalysisRequest(path=FAKE_PATH, methods=methods, include_exif=False)
    )
    ml_hits = [
        r
        for r in fake.methods
        if r.method.value in {
            MethodName.DEEPFAKE_ML.value,
            MethodName.DEEPFAKE_SCREEN_COC.value,
            MethodName.DEEPFAKE_ENSEMBLE.value,
        }
        and r.verdict is Verdict.SUSPICIOUS_INDICATION
    ]
    assert ml_hits, (
        "GUI post-open methods must flag labeled fake via an ML detector; got "
        f"{[(r.method.value, r.verdict.value, r.score) for r in fake.methods]}"
    )
    assert fake.overall_verdict is Verdict.SUSPICIOUS_INDICATION

    screen.btn_preset_default.click()
    screen.load_path(REAL_PATH)
    real_methods = screen._methods()
    assert set(DEEPFAKE_SCREENING_METHODS) <= set(real_methods)
    real = svc.analyze(
        AnalysisRequest(path=REAL_PATH, methods=real_methods, include_exif=False)
    )
    assert not [
        r
        for r in real.methods
        if r.method.value in {
            MethodName.DEEPFAKE_ML.value,
            MethodName.DEEPFAKE_SCREEN_COC.value,
            MethodName.DEEPFAKE_ENSEMBLE.value,
        }
        and r.verdict is Verdict.SUSPICIOUS_INDICATION
    ]
    assert real.overall_verdict is not Verdict.SUSPICIOUS_INDICATION
