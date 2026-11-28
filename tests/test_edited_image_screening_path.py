"""User-facing edited-image path: default methods + labeled copy-move/splice.

Drives shipped ``AnalysisService`` / ``generate_demo_sample``. Does not
reimplement detectors or hard-code expected scores.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from imageguard.core.enums import Verdict
from imageguard.services.analysis_service import AnalysisRequest, AnalysisService
from imageguard.ui.method_presets import (
    ALL_ANALYZE_METHODS,
    CAMERA_AI_METHODS,
    DEEPFAKE_SCREENING_METHODS,
    DEFAULT_ANALYZE_METHODS,
    DEMO_SPLICING_METHODS,
)
from imageguard.ui.widgets.sample_dialog import generate_demo_sample

COMOFOD_FAKE = Path("data/comofod_examples/028_F.png")
COMOFOD_ORIG = Path("data/comofod_examples/028_O.png")


def test_default_and_camera_presets_include_conclusive_edit_detectors() -> None:
    assert DEFAULT_ANALYZE_METHODS == ("keypoint", "block", "ela")
    assert {"keypoint", "block"} <= set(CAMERA_AI_METHODS)
    assert set(DEEPFAKE_SCREENING_METHODS) <= set(CAMERA_AI_METHODS)
    assert {"keypoint", "block"} <= set(DEMO_SPLICING_METHODS)
    assert set(CAMERA_AI_METHODS) | set(DEFAULT_ANALYZE_METHODS) <= set(ALL_ANALYZE_METHODS)
    assert len(ALL_ANALYZE_METHODS) == len(set(ALL_ANALYZE_METHODS))


@pytest.mark.skipif(
    not COMOFOD_FAKE.is_file() or not COMOFOD_ORIG.is_file(),
    reason="CoMoFoD example pair not installed",
)
def test_default_methods_flag_labeled_copy_move_and_clear_original() -> None:
    svc = AnalysisService()
    methods = list(DEFAULT_ANALYZE_METHODS)
    fake = svc.analyze(AnalysisRequest(path=COMOFOD_FAKE, methods=methods, include_exif=False))
    orig = svc.analyze(AnalysisRequest(path=COMOFOD_ORIG, methods=methods, include_exif=False))

    fake_hits = [
        r
        for r in fake.methods
        if r.method.value in {"copy_move_keypoint", "copy_move_block"}
        and r.verdict is Verdict.SUSPICIOUS_INDICATION
        and r.score is not None
        and r.score == r.score
    ]
    assert fake_hits, (
        "default methods must flag labeled copy-move; got "
        f"{[(r.method.value, r.verdict.value, r.score) for r in fake.methods]}"
    )
    assert fake.overall_verdict is Verdict.SUSPICIOUS_INDICATION
    assert orig.overall_verdict is not Verdict.SUSPICIOUS_INDICATION


def test_splicing_demo_recommended_methods_flag_via_shipped_service(tmp_path: Path) -> None:
    path, methods = generate_demo_sample("splicing", out_dir=tmp_path)
    assert {"keypoint", "block"} <= set(methods)
    svc = AnalysisService()
    report = svc.analyze(AnalysisRequest(path=path, methods=list(methods), include_exif=False))
    hits = [
        r
        for r in report.methods
        if r.verdict is Verdict.SUSPICIOUS_INDICATION
        and r.score is not None
        and r.score == r.score
    ]
    assert hits, (
        "splicing demo recommended methods must include a conclusive detector; got "
        f"{[(r.method.value, r.verdict.value, r.score) for r in report.methods]}"
    )
    assert report.overall_verdict is Verdict.SUSPICIOUS_INDICATION
