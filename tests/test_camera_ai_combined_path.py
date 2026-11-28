"""Camera/AI screening path on labeled copy-move AND labeled deepfake pairs.

Drives shipped ``AnalysisService`` with ``CAMERA_AI_METHODS``. Does not
reimplement detectors or hard-code expected scores. Skip (not pass) when
labeled images, ONNX weights, or onnxruntime are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from imageguard.core.enums import MethodName, Verdict
from imageguard.services.analysis_service import AnalysisRequest, AnalysisService
from imageguard.ui.method_presets import CAMERA_AI_METHODS, DEEPFAKE_SCREENING_METHODS
from imageguard.ui.widgets.sample_dialog import _DEMO_SPECS

CM_FAKE = Path("data/comofod_examples/028_F.png")
CM_ORIG = Path("data/comofod_examples/028_O.png")
DF_FAKE = Path("data/deepfake-coc-pilot/case-03.png")
DF_REAL = Path("data/deepfake-coc-pilot/case-01.png")

_COPYMOVE = {MethodName.COPY_MOVE_KEYPOINT.value, MethodName.COPY_MOVE_BLOCK.value}
_DEEPFAKE_ML = {
    MethodName.DEEPFAKE_ML.value,
    MethodName.DEEPFAKE_SCREEN_COC.value,
    MethodName.DEEPFAKE_ENSEMBLE.value,
}


def _combined_blocker() -> str | None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return "onnxruntime not installed"
    missing = [str(p) for p in (CM_FAKE, CM_ORIG, DF_FAKE, DF_REAL) if not p.is_file()]
    if missing:
        return "labeled images not installed"
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


needs_combined = pytest.mark.skipif(
    _combined_blocker() is not None,
    reason=_combined_blocker() or "combined screening assets unavailable",
)


def test_camera_ai_method_set_includes_copy_move_and_deepfake_detectors() -> None:
    specs = set(CAMERA_AI_METHODS)
    assert {"keypoint", "block"} & specs
    assert set(DEEPFAKE_SCREENING_METHODS) <= specs


def test_demo_recommended_methods_keep_deepfake_off_block_and_splicing_on_copy_move() -> None:
    deepfake = set(_DEMO_SPECS["deepfake"][1])
    splicing = set(_DEMO_SPECS["splicing"][1])
    assert set(DEEPFAKE_SCREENING_METHODS) <= deepfake
    assert "block" not in deepfake
    assert {"keypoint", "block"} & splicing


def _finite_suspicious(results, names: set[str]):
    return [
        r
        for r in results
        if r.method.value in names
        and r.verdict is Verdict.SUSPICIOUS_INDICATION
        and r.score is not None
        and r.score == r.score
    ]


@needs_combined
def test_camera_ai_methods_flag_copy_move_fake_and_clear_original() -> None:
    svc = AnalysisService()
    methods = list(CAMERA_AI_METHODS)
    fake = svc.analyze(AnalysisRequest(path=CM_FAKE, methods=methods, include_exif=False))
    orig = svc.analyze(AnalysisRequest(path=CM_ORIG, methods=methods, include_exif=False))
    assert _finite_suspicious(fake.methods, _COPYMOVE), (
        "Camera/AI set must flag labeled copy-move; got "
        f"{[(r.method.value, r.verdict.value, r.score) for r in fake.methods]}"
    )
    assert fake.overall_verdict is Verdict.SUSPICIOUS_INDICATION
    assert orig.overall_verdict is not Verdict.SUSPICIOUS_INDICATION


@needs_combined
def test_camera_ai_methods_flag_deepfake_face_and_clear_real() -> None:
    svc = AnalysisService()
    methods = list(CAMERA_AI_METHODS)
    fake = svc.analyze(AnalysisRequest(path=DF_FAKE, methods=methods, include_exif=False))
    real = svc.analyze(AnalysisRequest(path=DF_REAL, methods=methods, include_exif=False))
    assert _finite_suspicious(fake.methods, _DEEPFAKE_ML), (
        "Camera/AI set must flag labeled deepfake; got "
        f"{[(r.method.value, r.verdict.value, r.score) for r in fake.methods]}"
    )
    assert fake.overall_verdict is Verdict.SUSPICIOUS_INDICATION
    assert real.overall_verdict is not Verdict.SUSPICIOUS_INDICATION
    assert not [
        r for r in real.methods if r.method.value in _DEEPFAKE_ML and r.verdict is Verdict.SUSPICIOUS_INDICATION
    ]
