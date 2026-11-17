"""Contract & behavioral tests cho DeepfakeCues (kiến trúc 4 trụ cột DIP, AUX_METHODS)."""

from __future__ import annotations

import threading

import cv2
import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues, _detect_face_roi
from imageguard.forensics.fusion import AUX_METHODS


def _doc(arr: np.ndarray, name: str = "sample.png") -> ImageDocument:
    h, w = arr.shape[:2]
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    return ImageDocument(
        array=arr,
        width=w,
        height=h,
        channels=ch,
        image_format="png",
        sha256="d" * 64,
        file_bytes=int(arr.size),
        file_name=name,
    )


def _synthetic_face_frame(smooth_face: bool = False) -> np.ndarray:
    """Tạo khung hình BGR 320x320 mô phỏng nền kết cấu + khuôn mặt màu da có dải mắt đối xứng."""
    rng = np.random.default_rng(2026)
    frame = np.clip(95 + rng.normal(0.0, 14.0, (320, 320, 3)), 0, 255).astype(np.uint8)
    # Vẽ nền khác màu da ở 4 góc để kiểm chứng phân đoạn da YCrCb/HSV.
    frame[:, :, 0] = np.clip(frame[:, :, 0].astype(np.int16) + 35, 0, 255).astype(np.uint8)

    # Oval khuôn mặt màu da (BGR ~ (135, 168, 215)) lệch nhẹ khỏi tâm [x=150..270, y=50..200].
    face_mask = np.zeros((320, 320), dtype=np.uint8)
    cv2.ellipse(face_mask, (210, 125), (56, 72), 0, 0, 360, 255, -1)
    face_patch = np.zeros((320, 320, 3), dtype=np.float32)
    face_patch[:, :] = np.array([135.0, 168.0, 215.0], dtype=np.float32)
    # Thêm hai vùng mắt tối đối xứng để Haar two_v & SAT đối xứng gương nhận diện.
    cv2.circle(face_patch, (188, 108), 11, (55.0, 75.0, 105.0), -1)
    cv2.circle(face_patch, (232, 108), 11, (55.0, 75.0, 105.0), -1)
    # Thêm nhiễu cảm biến trên mặt hoặc làm mịn kiểu AI/Deepfake.
    if not smooth_face:
        face_patch += rng.normal(0.0, 13.0, (320, 320, 3)).astype(np.float32)
    else:
        face_patch = cv2.GaussianBlur(face_patch, (15, 15), 4.5)

    m = face_mask > 0
    frame[m] = np.clip(face_patch[m], 0, 255).astype(np.uint8)
    return frame


def test_deepfake_cues_skin_haar_dip_detects_face_roi() -> None:
    frame = _synthetic_face_frame(smooth_face=True)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (rx, ry, rw, rh), face_detected, roi_source = _detect_face_roi(frame, gray, 8)
    assert face_detected is True
    assert roi_source in {"skin_haar_dip", "haar_cascade"}
    # ROI phải bám vào nửa phải phía trên (quanh tâm x=210, y=125), không phải fallback (80,80).
    assert rx >= 135 and rx + rw <= 290
    assert ry >= 35 and ry + rh <= 220

    res = DeepfakeCues().analyze(_doc(frame, "face_smooth.png"), load_thresholds())
    assert res.verdict is Verdict.INCONCLUSIVE
    assert res.evidence["face_detected"] is True
    assert res.evidence["roi_source"] in {"skin_haar_dip", "haar_cascade"}
    for key in ("smooth_cue", "seam_cue", "lbp_cue", "lbp_chi2", "freq_cue", "spectral_rolloff"):
        assert key in res.evidence


def test_deepfake_cues_detects_smooth_synthetic_roi_as_inconclusive() -> None:
    rng = np.random.default_rng(123)
    img = np.clip(128 + rng.normal(0.0, 18.0, (256, 256, 3)), 0, 255).astype(np.uint8)
    # Mô phỏng vùng bị làm mịn/sinh AI ở vùng trung tâm [64:192, 64:192].
    img[64:192, 64:192] = cv2.GaussianBlur(img[64:192, 64:192], (15, 15), 5.0)

    cfg = load_thresholds()
    res = DeepfakeCues().analyze(_doc(img), cfg)

    assert res.method is MethodName.DEEPFAKE_CUES
    assert MethodName.DEEPFAKE_CUES in AUX_METHODS
    assert res.verdict is Verdict.INCONCLUSIVE
    assert res.verdict is not Verdict.SUSPICIOUS_INDICATION
    assert res.score is not None and 0.35 <= res.score <= 1.0
    assert res.score_definition
    assert res.limitations
    assert res.heatmap is not None and res.heatmap.shape == (256, 256)
    assert res.mask is not None and res.mask.shape == (256, 256)
    assert int(np.count_nonzero(res.mask)) > 0
    assert res.evidence["lbp_chi2"] > 0.0
    assert res.params_hash == cfg.snapshot_hash()


def test_deepfake_cues_detects_fft_checkerboard_artifact() -> None:
    rng = np.random.default_rng(77)
    img = np.clip(128 + rng.normal(0.0, 10.0, (256, 256, 3)), 0, 255).astype(np.float32)
    yy, xx = np.mgrid[64:192, 64:192]
    # Lưới tuần hoàn tần số cao điển hình của GAN/Diffusion upsampling (chu kỳ 3px).
    grid = 22.0 * np.cos(2.0 * np.pi * xx / 3.0) * np.cos(2.0 * np.pi * yy / 3.0)
    img[64:192, 64:192] += grid[:, :, None]
    u8 = np.clip(img, 0, 255).astype(np.uint8)

    res = DeepfakeCues().analyze(_doc(u8, "checkerboard.png"), load_thresholds())
    assert res.evidence["freq_cue"] >= 0.45
    assert res.evidence["fft_hf_peak_ratio"] > 4.0
    assert res.verdict in {Verdict.INCONCLUSIVE, Verdict.NO_INDICATION}
    assert res.verdict is not Verdict.SUSPICIOUS_INDICATION


def test_deepfake_cues_non_face_smooth_camera_scene_avoids_false_positive() -> None:
    """Mô phỏng cảnh nền camera (tường/vật thể hơi mịn ở giữa, không có mặt) -> không báo giả."""
    rng = np.random.default_rng(42)
    bg = np.clip(110 + rng.normal(0.0, 4.5, (256, 256, 3)), 0, 255).astype(np.float32)
    # Vùng trung tâm chuyển sáng nhẹ có kết cấu tự nhiên chứ không có đường ghép/mất vi cấu trúc.
    yy, xx = np.mgrid[0:256, 0:256]
    bg += (12.0 * np.sin(xx / 28.0) * np.cos(yy / 28.0))[:, :, None]
    # Giảm nhẹ nhiễu ở trung tâm (giống DOF/bề mặt tường hơi mờ tự nhiên ~35% chứ không phải deepfake).
    center_noise = rng.normal(0.0, 3.0, (128, 128, 3)).astype(np.float32)
    bg[64:192, 64:192] = np.clip(bg[64:192, 64:192] * 0.9 + center_noise, 0, 255)
    frame = np.clip(bg, 0, 255).astype(np.uint8)

    res = DeepfakeCues().analyze(_doc(frame, "scene.png"), load_thresholds())
    assert res.evidence["face_detected"] is False
    assert res.verdict is Verdict.NO_INDICATION
    assert res.verdict is not Verdict.SUSPICIOUS_INDICATION


def test_deepfake_cues_uniform_noise_no_indication_and_never_suspicious() -> None:
    rng = np.random.default_rng(9)
    img = np.clip(128 + rng.normal(0.0, 16.0, (256, 256, 3)), 0, 255).astype(np.uint8)

    res = DeepfakeCues().analyze(_doc(img), load_thresholds())

    assert res.verdict is Verdict.NO_INDICATION
    assert res.verdict is not Verdict.SUSPICIOUS_INDICATION
    assert res.score is not None and res.score < 0.35
    assert res.score_definition
    assert res.mask is None


def test_deepfake_cues_flat_image_no_indication() -> None:
    flat = np.full((128, 128, 3), 120, dtype=np.uint8)
    res = DeepfakeCues().analyze(_doc(flat, "flat.png"), load_thresholds())
    assert res.verdict is Verdict.NO_INDICATION
    assert res.score is not None and res.score < 0.35

    black = np.zeros((128, 128, 3), dtype=np.uint8)
    res_black = DeepfakeCues().analyze(_doc(black, "black.png"), load_thresholds())
    assert res_black.verdict is Verdict.NO_INDICATION
    assert res_black.score is not None and res_black.score < 0.35


def test_deepfake_cues_tiny_image_not_applicable() -> None:
    tiny = np.zeros((32, 32, 3), dtype=np.uint8)
    res = DeepfakeCues().analyze(_doc(tiny, "tiny.png"), load_thresholds())
    assert res.verdict is Verdict.NOT_APPLICABLE
    assert res.score is None
    assert res.limitations


def test_deepfake_cues_cooperative_cancel() -> None:
    rng = np.random.default_rng(5)
    img = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    ev = threading.Event()
    ev.set()

    res = DeepfakeCues().analyze(_doc(img), load_thresholds(), cancel=ev)
    assert res.verdict is Verdict.INCONCLUSIVE
    assert any("cancelled" in item for item in res.limitations)


def test_deepfake_cues_dual_camera_cross_sensor_detects_anomalous_half() -> None:
    rng = np.random.default_rng(42)
    h, w_half = 240, 320
    # Nửa trái (camera 1): nền nhiễu tự nhiên + vùng mặt skin-tone làm mịn nhân tạo
    left = np.full((h, w_half, 3), 120, dtype=np.uint8)
    left = np.clip(left.astype(np.int16) + rng.normal(0, 16, left.shape), 0, 255).astype(np.uint8)
    cv2.ellipse(left, (160, 120), (54, 70), 0, 0, 360, (145, 165, 205), -1)
    # Nửa phải (camera 2): cảnh tự nhiên có nhiễu cảm biến đồng đều
    right = np.full((h, w_half, 3), 120, dtype=np.uint8)
    right = np.clip(right.astype(np.int16) + rng.normal(0, 16, right.shape), 0, 255).astype(np.uint8)

    dual = np.hstack([left, right])
    res = DeepfakeCues().analyze(_doc(dual, "dual.png"), load_thresholds())
    assert res.evidence["dual_camera_mode"] is True
    assert res.evidence["dual_roi_side"] == "left"
    assert res.evidence["dual_sigma_left"] < res.evidence["dual_sigma_right"]
    assert "seam_hough_cue" in res.evidence
    assert "seam_hough_lines" in res.evidence
    assert "hog_asym" in res.evidence
    assert res.verdict is Verdict.INCONCLUSIVE
