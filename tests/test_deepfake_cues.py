"""Contract & behavioral tests cho DeepfakeCues (6 trụ cột DIP, AUX_METHODS)."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues, _detect_face_roi
from imageguard.forensics.fusion import AUX_METHODS


def _doc(arr: np.ndarray, name: str = "sample.png", *, dual: bool = False) -> ImageDocument:
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
        capture_layout="dual_side_by_side" if dual else "single",
        capture_split_x=w // 2 if dual else None,
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
    assert res.evidence["cue_flag"] is True
    assert res.score is not None and 0.35 <= res.score <= 1.0
    assert res.mask is not None and np.count_nonzero(res.mask) > 0
    assert res.evidence["roi_source"] in {"skin_haar_dip", "haar_cascade"}
    for key in ("smooth_cue", "seam_cue", "lbp_cue", "lbp_chi2", "freq_cue", "spectral_rolloff"):
        assert key in res.evidence


def test_smooth_patch_without_face_is_diagnostic_only() -> None:
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
    assert res.score is None
    assert res.evidence["cue_flag"] is False
    assert res.evidence["unavailable_reason"] == "no_face"
    assert res.score_definition
    assert res.limitations
    assert res.heatmap is None
    assert res.mask is None
    assert res.evidence["lbp_chi2"] > 0.0
    assert res.params_hash == cfg.snapshot_hash()


def test_freq_scale_keeps_webcam_peak_below_saturation() -> None:
    """peak_ratio ~8 (mặt webcam) không được trở lại freq=1; lưới ~29 vẫn bão hòa."""
    source = Path("src/imageguard/forensics/auxiliary/deepfake_cues.py").read_text(
        encoding="utf-8"
    )
    assert "(peak_ratio - 4.0) / 14.0" in source
    assert "(peak_ratio - 3.3) / 2.2" not in source
    mild = float(np.clip((8.0 - 4.0) / 14.0, 0.0, 1.0))
    assert mild < 0.5


def test_deepfake_cues_detects_fft_checkerboard_artifact() -> None:
    rng = np.random.default_rng(77)
    img = np.clip(128 + rng.normal(0.0, 10.0, (256, 256, 3)), 0, 255).astype(np.float32)
    yy, xx = np.mgrid[64:192, 64:192]
    # Lưới tuần hoàn tần số cao điển hình của GAN/Diffusion upsampling (chu kỳ 3px).
    grid = 22.0 * np.cos(2.0 * np.pi * xx / 3.0) * np.cos(2.0 * np.pi * yy / 3.0)
    img[64:192, 64:192] += grid[:, :, None]
    u8 = np.clip(img, 0, 255).astype(np.uint8)

    res = DeepfakeCues().analyze(_doc(u8, "checkerboard.png"), load_thresholds())
    assert res.evidence["fft_hf_peak_ratio"] > 4.0
    # A periodic grid concentrates spectral power even when the percentile
    # amplitude peak misses some narrow peaks. The legacy raw top-eight
    # share is diagnostic only; normalized power drives the periodic cue.
    assert res.evidence["fft_hf_conc8"] > 0.02
    assert 0.45 <= res.evidence["freq_cue"] <= 1.0
    assert res.verdict in {Verdict.INCONCLUSIVE, Verdict.NO_INDICATION}
    assert res.verdict is not Verdict.SUSPICIOUS_INDICATION


def test_fft_grid_concentration_catches_upsampling_but_not_face() -> None:
    """Hồi quy Wukong claim 5: upsample NN 4x của mặt thật vẫn bị cờ (freq_cue cao),
    trong khi mặt textured tự nhiên thì freq_cue ~0 (không báo giả pillar 4)."""
    cfg = load_thresholds()
    face = _synthetic_face_frame(smooth_face=False)
    small = cv2.resize(face, (face.shape[1] // 4, face.shape[0] // 4), interpolation=cv2.INTER_AREA)
    up4 = cv2.resize(small, (face.shape[1], face.shape[0]), interpolation=cv2.INTER_NEAREST)

    res_face = DeepfakeCues().analyze(_doc(face, "face.png"), cfg)
    res_up = DeepfakeCues().analyze(_doc(up4, "face_nn4x.png"), cfg)

    assert res_face.evidence["freq_cue"] < 0.15
    assert res_face.evidence["fft_hf_conc8"] < 0.006
    assert res_up.evidence["freq_cue"] >= 0.55
    assert res_up.evidence["fft_hf_conc8"] > res_face.evidence["fft_hf_conc8"]
    for res in (res_face, res_up):
        assert res.verdict in {Verdict.INCONCLUSIVE, Verdict.NO_INDICATION, Verdict.NOT_APPLICABLE}
        assert res.verdict is not Verdict.SUSPICIOUS_INDICATION


def _face_like_frame(seed: int) -> np.ndarray:
    """Mặt tổng hợp có seed (ellipse da + 2 mắt + nhiễu cảm biến) để cohort nhiều seed."""
    rng = np.random.default_rng(seed)
    img = np.clip(128 + rng.normal(0, 14, (320, 320, 3)), 0, 255).astype(np.float32)
    cv2.ellipse(img, (160, 150), (70, 90), 0, 0, 360, (170, 140, 120), -1)
    cv2.circle(img, (135, 135), 9, (30, 30, 40), -1)
    cv2.circle(img, (185, 135), 9, (30, 30, 40), -1)
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _grid3px_frame(seed: int) -> np.ndarray:
    """Lưới tuần hoàn 3px kiểu GAN/Diffusion upsampling trên nền nhiễu (chu kỳ 3px)."""
    rng = np.random.default_rng(seed)
    img = np.clip(128 + rng.normal(0, 10, (256, 256, 3)), 0, 255).astype(np.float32)
    yy, xx = np.mgrid[64:192, 64:192]
    img[64:192, 64:192] += (22.0 * np.cos(2 * np.pi * xx / 3.0) * np.cos(2 * np.pi * yy / 3.0))[:, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def test_fft_conc8_threshold_separates_faces_from_grids_across_seeds() -> None:
    """Ngưỡng grid_score (conc8 > 0.006) không overfit 1 seed: cohort 20 seed,
    mọi mặt đều conc8 < 0.006, mọi lưới 3px đều conc8 > 0.02 (biên ~16x).
    Số đo từ chạy thật 2026-11-24 (plan 261124-2152 ledger) — cohort synthetic,
    không phải số đo camera thật."""
    from imageguard.forensics.auxiliary.deepfake_cues import _fft_highfreq_cue

    def _conc8(img: np.ndarray) -> float:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape
        roi = g[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        return _fft_highfreq_cue(roi)[3]

    face_concs = [_conc8(_face_like_frame(1000 + s)) for s in range(20)]
    grid_concs = [_conc8(_grid3px_frame(2000 + s)) for s in range(20)]

    assert max(face_concs) < 0.006
    assert min(grid_concs) > 0.02
    assert min(grid_concs) > 8 * max(face_concs)


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
    assert res.verdict is Verdict.INCONCLUSIVE
    assert res.score is None and res.evidence["cue_flag"] is False
    assert res.verdict is not Verdict.SUSPICIOUS_INDICATION


def test_deepfake_cues_uniform_noise_abstains_without_face() -> None:
    rng = np.random.default_rng(9)
    img = np.clip(128 + rng.normal(0.0, 16.0, (256, 256, 3)), 0, 255).astype(np.uint8)

    res = DeepfakeCues().analyze(_doc(img), load_thresholds())

    assert res.verdict is Verdict.INCONCLUSIVE
    assert res.verdict is not Verdict.SUSPICIOUS_INDICATION
    assert res.score is None
    assert res.evidence["unavailable_reason"] == "no_face"
    assert res.score_definition
    assert res.mask is None


def test_deepfake_cues_flat_image_abstains() -> None:
    flat = np.full((128, 128, 3), 120, dtype=np.uint8)
    res = DeepfakeCues().analyze(_doc(flat, "flat.png"), load_thresholds())
    assert res.verdict is Verdict.INCONCLUSIVE
    assert res.score is None and res.evidence["cue_unavailable"] is True
    assert res.evidence["freq_cue"] == 0.0

    black = np.zeros((128, 128, 3), dtype=np.uint8)
    res_black = DeepfakeCues().analyze(_doc(black, "black.png"), load_thresholds())
    assert res_black.verdict is Verdict.INCONCLUSIVE
    assert res_black.score is None and res_black.evidence["cue_flag"] is False
    assert res_black.evidence["freq_cue"] == 0.0


def test_repeated_background_cannot_replace_a_face_for_deepfake_flag() -> None:
    rng = np.random.default_rng(307)
    pattern = 18 * np.sin(2.0 * np.pi * np.arange(192) / 4)
    image = np.clip(115 + pattern + rng.normal(0, 5, (192, 192)), 0, 255).astype(np.uint8)
    cfg = load_thresholds()
    cfg.raw["deepfake"] = {"score_thresh": 0.05}

    res = DeepfakeCues().analyze(_doc(image), cfg)

    assert res.evidence["face_detected"] is False
    assert res.evidence["freq_cue"] >= 0.55
    assert res.evidence["seam_hough_cue"] >= 0.70
    assert res.score is None
    assert res.evidence["cue_flag"] is False
    assert res.verdict is Verdict.INCONCLUSIVE
    assert res.mask is None


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
    res = DeepfakeCues().analyze(_doc(dual, "dual.png", dual=True), load_thresholds())
    assert res.evidence["dual_camera_mode"] is True
    assert res.evidence["dual_roi_side"] == "left"
    assert res.evidence["dual_sigma_left"] < res.evidence["dual_sigma_right"]
    assert "seam_hough_cue" in res.evidence
    assert "seam_hough_lines" in res.evidence
    assert "hog_asym" in res.evidence
    assert res.verdict is Verdict.INCONCLUSIVE


@pytest.mark.parametrize("face_side", ["left", "right"])
def test_dual_camera_uses_single_detected_face_over_smooth_background(face_side: str) -> None:
    face = _synthetic_face_frame()[:288]
    rng = np.random.default_rng(307)
    gray = np.clip(105 + rng.normal(0, 13, face.shape[:2]), 0, 255).astype(np.uint8)
    gray[72:216, 80:240] = 110
    room = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    frame = np.hstack((face, room) if face_side == "left" else (room, face))

    res = DeepfakeCues().analyze(_doc(frame, dual=True), load_thresholds())

    assert res.evidence["dual_camera_mode"] is True
    assert res.evidence["dual_roi_side"] == face_side
    assert res.evidence["face_detected"] is True
    assert res.verdict is Verdict.NO_INDICATION
    assert res.mask is None
