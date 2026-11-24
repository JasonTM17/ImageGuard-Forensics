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
