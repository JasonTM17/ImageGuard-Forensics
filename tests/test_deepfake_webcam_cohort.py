"""Cohort tổng hợp kiểu webcam. Không công bố độ chính xác trên camera thật."""

from __future__ import annotations

import cv2
import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import Verdict
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues
from tests.test_deepfake_cues import _doc, _synthetic_face_frame

_ALLOWED = {
    Verdict.INCONCLUSIVE,
    Verdict.NO_INDICATION,
    Verdict.NOT_APPLICABLE,
}


def _checkerboard_face() -> np.ndarray:
    frame = _synthetic_face_frame(smooth_face=True)
    yy, xx = np.mgrid[0:320, 0:320]
    checks = ((xx // 8 + yy // 8) % 2).astype(np.uint8) * 255
    mask = np.zeros((320, 320), dtype=np.uint8)
    cv2.ellipse(mask, (210, 125), (56, 72), 0, 0, 360, 255, -1)
    frame[mask > 0] = checks[mask > 0, None]
    return frame


def test_webcam_shaped_frames_stay_exploratory() -> None:
    frames = {
        "textured": _synthetic_face_frame(smooth_face=False),
        "smooth": _synthetic_face_frame(smooth_face=True),
        "checkerboard": _checkerboard_face(),
    }
    cfg = load_thresholds()
    for name, frame in frames.items():
        result = DeepfakeCues().analyze(_doc(frame, f"{name}.png"), cfg, None)
        assert result.verdict in _ALLOWED, name
