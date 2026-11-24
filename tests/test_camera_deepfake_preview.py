"""Kiểm tra format_deepfake_preview trên khung mặt mô phỏng."""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import Verdict
from imageguard.ui import strings
from imageguard.ui.widgets.camera_deepfake_preview import (
    format_burst_deepfake_preview,
    format_deepfake_preview,
)
from tests.test_deepfake_cues import _synthetic_face_frame


def test_format_deepfake_preview_smooth_face_disclaimer_and_verdict() -> None:
    frame = _synthetic_face_frame(smooth_face=True)
    text = format_deepfake_preview(frame)

    assert text.startswith(strings.DEEPFAKE_PREVIEW_TITLE)
    assert strings.DEEPFAKE_PREVIEW_DISCLAIMER in text
    assert Verdict.SUSPICIOUS_INDICATION.value not in text
    assert "đã xác định giả" not in text.lower()
    assert "face_detected:" in text


def test_still_face_burst_is_exploratory_not_a_deepfake_label() -> None:
    frame = _synthetic_face_frame(smooth_face=False)
    text = format_burst_deepfake_preview([frame.copy() for _ in range(5)])
    assert "temporal_static_score:" in text
    assert strings.VERDICT_LABEL["inconclusive"] in text
    assert Verdict.SUSPICIOUS_INDICATION.value not in text
    assert "đã xác định giả" not in text.lower()


def test_moving_face_burst_stays_below_static_deepfake_wording() -> None:
    base = _synthetic_face_frame(smooth_face=False).astype(np.int16)
    rng = np.random.default_rng(5)
    frames = []
    for _ in range(5):
        noisy = base + rng.integers(-40, 41, base.shape, dtype=np.int16)
        frames.append(np.clip(noisy, 0, 255).astype(np.uint8))
    text = format_burst_deepfake_preview(frames)
    assert strings.DEEPFAKE_PREVIEW_DISCLAIMER in text
    assert strings.CAMERA_BURST_HINT not in text
    assert "đã xác định giả" not in text.lower()


def test_camera_next_step_does_not_call_the_live_frame_a_deepfake() -> None:
    text = strings.ANALYZE_CAMERA_NEXT_STEP
    assert "không phải kết luận deepfake" in text
    assert "mốc cùng phiên" in text
    assert "đã xác định giả" not in text.lower()
    assert strings.DEEPFAKE_VERDICT_SCREEN not in strings.VERDICT_LABEL.values()