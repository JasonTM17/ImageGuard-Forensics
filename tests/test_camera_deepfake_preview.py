"""Kiểm tra format_deepfake_preview trên khung mặt mô phỏng."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import Verdict
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues
from imageguard.ui import strings
from imageguard.ui.widgets.camera_deepfake_preview import (
    format_burst_deepfake_preview,
    format_deepfake_preview,
)
from tests.test_deepfake_cues import _doc, _synthetic_face_frame


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
    spatial = DeepfakeCues().analyze(_doc(frame, "base.png"), load_thresholds())
    assert "temporal_static_score:" in text
    assert strings.VERDICT_LABEL[spatial.verdict.value] in text
    assert "không cộng vào điểm" in text
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

def test_static_non_face_burst_does_not_show_face_or_burst_claims() -> None:
    frame = np.full((96, 128, 3), 120, dtype=np.uint8)
    text = format_burst_deepfake_preview([frame.copy() for _ in range(5)])
    assert strings.DEEPFAKE_PREVIEW_LIVE_TEXTURE not in text
    assert strings.CAMERA_BURST_HINT not in text
    assert "temporal_frame_count: 5" in text
    assert "temporal_static_score:" not in text
    assert "chưa định vị được vùng mặt" in text.lower()


def test_camera_preview_explains_small_frame() -> None:
    text = format_deepfake_preview(np.zeros((32, 32, 3), dtype=np.uint8))

    assert strings.VERDICT_LABEL["not_applicable"] in text
    assert "ảnh quá nhỏ (32x32)" in text


def test_empty_burst_has_actionable_message_instead_of_index_error() -> None:
    text = format_burst_deepfake_preview([])

    assert strings.DEEPFAKE_PREVIEW_NO_FRAMES in text
    assert "temporal_static_score:" not in text


def test_preview_shows_score_scale_and_quality_reason() -> None:
    from imageguard.core.enums import MethodName
    from imageguard.core.models import AlgorithmResult
    from imageguard.ui.widgets.camera_deepfake_preview import _format_result

    result = AlgorithmResult(
        method=MethodName.DEEPFAKE_CUES,
        verdict=Verdict.INCONCLUSIVE,
        score=0.2,
        score_definition="uncalibrated exploratory spatial cue",
        evidence={"face_detected": True, "calibration_status": "uncalibrated"},
        limitations=["Ánh sáng không đồng đều; cần thêm mẫu đối chiếu."],
    )
    text = _format_result(result)

    assert "0.200/1" in text
    assert "không phải xác suất thật/giả" in text
    assert result.limitations[0] in text
