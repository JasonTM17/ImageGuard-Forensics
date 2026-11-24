"""AnalysisService phải gắn cue mốc cùng phiên, không kết luận deepfake."""

from __future__ import annotations

import cv2

from imageguard.core.enums import Verdict
from imageguard.services.analysis_service import AnalysisRequest, AnalysisService
from tests.test_face_roi_cross_compare import _face_frame


def _write_png(path, frame) -> None:
    ok, encoded = cv2.imencode(".png", frame)
    assert ok
    path.write_bytes(encoded.tobytes())


def _deepfake(report):
    return next(item for item in report.methods if item.method.value == "deepfake_cues")


def test_service_without_reference_omits_reference_evidence(tmp_path) -> None:
    suspect = tmp_path / "suspect.png"
    _write_png(suspect, _face_frame(smooth_face=True))
    report = AnalysisService().analyze(
        AnalysisRequest(path=suspect, methods=["deepfake"], include_exif=False)
    )
    evidence = _deepfake(report).evidence
    assert "reference_sha256" not in evidence
    assert _deepfake(report).verdict is not Verdict.SUSPICIOUS_INDICATION


def test_service_reference_marks_smoothed_face_without_calling_it_fake(tmp_path) -> None:
    reference = tmp_path / "reference.png"
    suspect = tmp_path / "suspect.png"
    _write_png(reference, _face_frame(smooth_face=False))
    _write_png(suspect, _face_frame(smooth_face=True))
    report = AnalysisService().analyze(
        AnalysisRequest(
            path=suspect,
            methods=["deepfake"],
            include_exif=False,
            reference_path=reference,
        )
    )
    result = _deepfake(report)
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.verdict is not Verdict.SUSPICIOUS_INDICATION
    assert "reference_sha256" in result.evidence
    assert result.evidence["ref_combined_cue"] >= 0.50
    blob = " ".join(str(value) for value in result.evidence.values())
    assert "\\" not in blob
    assert "reference.png" not in blob


def test_service_identical_reference_does_not_invent_a_cue(tmp_path) -> None:
    frame = _face_frame(smooth_face=False)
    reference = tmp_path / "reference.png"
    suspect = tmp_path / "suspect.png"
    _write_png(reference, frame)
    _write_png(suspect, frame.copy())
    alone = AnalysisService().analyze(
        AnalysisRequest(path=suspect, methods=["deepfake"], include_exif=False)
    )
    compared = AnalysisService().analyze(
        AnalysisRequest(
            path=suspect,
            methods=["deepfake"],
            include_exif=False,
            reference_path=reference,
        )
    )
    base = _deepfake(alone)
    result = _deepfake(compared)
    assert result.verdict is base.verdict
    assert result.verdict is not Verdict.SUSPICIOUS_INDICATION
    assert result.evidence["ref_combined_cue"] == 0.0
