"""AnalysisService phải gắn cue mốc cùng phiên, không kết luận deepfake."""

from __future__ import annotations

import cv2
import numpy as np

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


def test_service_passes_quality_thresholds_to_reference_comparison(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import imageguard.forensics.auxiliary.face_roi_cross_compare as comparison
    import imageguard.services.analysis_service as service
    from imageguard.core.config import load_thresholds
    from tests.test_face_roi_cross_compare import _base_result

    monkeypatch.setattr(
        comparison, "_detect_face_roi",
        lambda arr, gray, block, cfg=None: ((0, 0, gray.shape[1], gray.shape[0]), True, "test_face"),
    )
    monkeypatch.setattr(
        service, "load_method",
        lambda spec: SimpleNamespace(analyze=lambda *args: _base_result(score=0.2)),
    )
    rng = np.random.default_rng(19)
    reference = tmp_path / "reference.png"
    suspect = tmp_path / "suspect.png"
    _write_png(reference, rng.integers(50, 190, (64, 64, 3), dtype=np.uint8))
    _write_png(suspect, rng.integers(50, 190, (128, 128, 3), dtype=np.uint8))
    cfg = load_thresholds()
    cfg.raw["deepfake"]["min_face_side"]["value"] = 96

    result = _deepfake(AnalysisService(cfg=cfg).analyze(AnalysisRequest(
        path=suspect, reference_path=reference, methods=["deepfake"], include_exif=False,
    )))

    assert result.evidence["ref_status"] == "quality_unavailable"
    assert result.evidence["ref_quality_reason"] == "roi_too_small"
    assert result.evidence["sus_quality_reason"] == ""
    assert result.evidence["reference_cue_flag"] is False
    assert result.score == 0.2
    assert result.verdict is Verdict.NO_INDICATION
