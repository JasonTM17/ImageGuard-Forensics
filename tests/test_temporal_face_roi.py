"""ROI đứng yên cho điểm cao hơn ROI nhiễu. Không mở webcam."""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.forensics.auxiliary.temporal_face_roi import (
    TEMPORAL_SCORE_MAX_DELTA,
    TEMPORAL_STATIC_MIN,
    apply_temporal_cue,
    temporal_static_cue,
)
from tests.test_deepfake_cues import _synthetic_face_frame


def _result(score: float = 0.1) -> AlgorithmResult:
    return AlgorithmResult(
        method=MethodName.DEEPFAKE_CUES,
        verdict=Verdict.NO_INDICATION,
        score=score,
        score_definition="test score",
    )


def test_identical_frames_are_more_static_than_noisy_roi() -> None:
    base = _synthetic_face_frame(smooth_face=False)
    still = temporal_static_cue([base.copy() for _ in range(5)])
    rng = np.random.default_rng(3)
    noisy = []
    for _ in range(5):
        frame = base.astype(np.int16)
        frame += rng.integers(-40, 41, frame.shape, dtype=np.int16)
        noisy.append(np.clip(frame, 0, 255).astype(np.uint8))
    moving = temporal_static_cue(noisy)
    assert float(still["temporal_static_score"]) > float(moving["temporal_static_score"])
    assert float(still["temporal_static_score"]) >= TEMPORAL_STATIC_MIN
    assert float(moving["temporal_static_score"]) < TEMPORAL_STATIC_MIN


def test_apply_temporal_cue_never_returns_suspicious() -> None:
    base = _synthetic_face_frame(smooth_face=True)
    out = apply_temporal_cue(_result(0.2), [base.copy() for _ in range(4)])
    assert out.verdict != Verdict.SUSPICIOUS_INDICATION
    assert out.score is not None
    assert out.score <= 0.2 + TEMPORAL_SCORE_MAX_DELTA + 1e-6
    assert "temporal_static_score" in out.evidence
    assert "\\" not in str(out.evidence["temporal_static_score"])
    assert out.score == 0.2
    assert out.verdict is Verdict.NO_INDICATION
    assert out.evidence["temporal_status"] == "static"


def test_static_face_remains_acquisition_evidence_for_inconclusive_result() -> None:
    base = _synthetic_face_frame(smooth_face=False)
    result = _result(0.42)
    result.verdict = Verdict.INCONCLUSIVE
    out = apply_temporal_cue(result, [base.copy() for _ in range(4)])

    assert out.score == result.score
    assert out.verdict is result.verdict
    assert out.evidence["temporal_status"] == "static"
    assert "gần như đứng yên" in out.evidence["temporal_reason"]
    assert not any("biến thiên theo thời gian" in lim for lim in out.limitations)
    assert "temporal_status" not in result.evidence  # no mutation of the input


def test_temporal_measurement_failure_preserves_spatial_result() -> None:
    out = apply_temporal_cue(_result(0.2), [np.array([]), np.array([])])

    assert out.score == 0.2
    assert out.verdict is Verdict.NO_INDICATION
    assert out.evidence["temporal_status"] == "error"


def test_laplacian_uses_documented_default_ksize1(monkeypatch) -> None:
    import cv2

    base = _synthetic_face_frame(smooth_face=False)
    original = cv2.Laplacian
    seen: list[int | None] = []

    def spy(src, ddepth, **kwargs):
        seen.append(kwargs.get("ksize"))
        return original(src, ddepth, **kwargs)

    monkeypatch.setattr(cv2, "Laplacian", spy)
    temporal_static_cue([base.copy(), base.copy()])
    assert seen and all(k in (None, 1) for k in seen)


def test_static_non_face_burst_does_not_change_score_or_verdict() -> None:
    frame = np.full((96, 128, 3), 120, dtype=np.uint8)
    out = apply_temporal_cue(_result(0.2), [frame.copy() for _ in range(4)])
    assert out.score == 0.2
    assert out.verdict == Verdict.NO_INDICATION
    assert out.evidence["temporal_face_detected"] == 0
