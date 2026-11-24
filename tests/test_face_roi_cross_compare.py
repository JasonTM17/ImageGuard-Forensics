"""Contract & behavioral tests cho face-ROI cross-compare với ảnh tham chiếu phiên.

Cặp synthetic dùng để chứng minh ``REFERENCE_CUE_MIN``: khung mặt màu da có
nhiễu cảm biến (textured) vs cùng khung mặt bị GaussianBlur mất nhiễu (smooth —
mô phỏng da bị làm mịn kiểu AI/deepfake). Kiểm chứng bằng số đo của chính cặp
này, không suy ra tỉ lệ chính xác trên dữ liệu thật.

Không mở webcam — mọi ImageDocument dựng từ mảng NumPy tổng hợp.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.forensics.auxiliary.face_roi_cross_compare import (
    REFERENCE_CUE_MIN,
    REFERENCE_SCORE_MAX_DELTA,
    apply_reference_cue,
    compare_face_rois,
)

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _doc(arr: np.ndarray, name: str = "frame.png", sha: str = "d" * 64) -> ImageDocument:
    h, w = arr.shape[:2]
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    return ImageDocument(
        array=arr,
        width=w,
        height=h,
        channels=ch,
        image_format="png",
        sha256=sha,
        file_bytes=int(arr.size),
        file_name=name,
    )


def _face_frame(seed: int = 2026, smooth_face: bool = False) -> np.ndarray:
    """Khung BGR 320x320: nền có kết cấu + oval mặt màu da với hai mắt đối xứng.

    ``smooth_face=True`` làm mịn vùng mặt (GaussianBlur) — mất vi kết cấu
    nhiễu cảm biến, mô phỏng mặt sinh AI; ``False`` giữ nhiễu σ=13 tự nhiên.
    """
    rng = np.random.default_rng(seed)
    frame = np.clip(95 + rng.normal(0.0, 14.0, (320, 320, 3)), 0, 255).astype(np.uint8)
    frame[:, :, 0] = np.clip(frame[:, :, 0].astype(np.int16) + 35, 0, 255).astype(np.uint8)

    face_mask = np.zeros((320, 320), dtype=np.uint8)
    cv2.ellipse(face_mask, (210, 125), (56, 72), 0, 0, 360, 255, -1)
    face_patch = np.zeros((320, 320, 3), dtype=np.float32)
    face_patch[:, :] = np.array([135.0, 168.0, 215.0], dtype=np.float32)
    cv2.circle(face_patch, (188, 108), 11, (55.0, 75.0, 105.0), -1)
    cv2.circle(face_patch, (232, 108), 11, (55.0, 75.0, 105.0), -1)
    if not smooth_face:
        face_patch += rng.normal(0.0, 13.0, (320, 320, 3)).astype(np.float32)
    else:
        face_patch = cv2.GaussianBlur(face_patch, (15, 15), 4.5)

    m = face_mask > 0
    frame[m] = np.clip(face_patch[m], 0, 255).astype(np.uint8)
    return frame


def _base_result(score: float | None = 0.20, verdict: Verdict = Verdict.NO_INDICATION) -> AlgorithmResult:
    mask = np.zeros((320, 320), dtype=np.uint8)
    mask[60:190, 160:270] = 255
    heat = np.full((320, 320), 0.4, dtype=np.float32)
    return AlgorithmResult(
        method=MethodName.DEEPFAKE_CUES,
        verdict=verdict,
        score=score,
        score_definition="test combined cue",
        mask=mask,
        heatmap=heat,
        evidence={"variant": "dip_4pillar_deepfake_cues", "face_detected": True},
        limitations=["tín hiệu khám phá"],
        params_hash="cfg123",
        duration_ms=42,
    )


def test_identical_array_gives_zero_cue_and_same_frame() -> None:
    frame = _face_frame()
    ref = _doc(frame, "ref.png", sha="a" * 64)
    sus = _doc(frame.copy(), "sus.png", sha="b" * 64)

    cmp = compare_face_rois(ref, sus)
    assert cmp["same_frame"] is True
    assert cmp["ref_combined_cue"] == 0.0


def test_smooth_face_against_textured_reference_scores_higher() -> None:
    ref = _doc(_face_frame(smooth_face=False), "ref.png", sha="a" * 64)
    sus_smooth = _doc(_face_frame(smooth_face=True), "sus.png", sha="b" * 64)
    sus_new_draw = _doc(_face_frame(seed=777, smooth_face=False), "sus.png", sha="c" * 64)

    cmp_smooth = compare_face_rois(ref, sus_smooth)
    cmp_same_face = compare_face_rois(ref, sus_new_draw)

    assert cmp_smooth["same_frame"] is False
    assert cmp_same_face["same_frame"] is False
    # Mặt làm mịn phải cho cue cao hơn hẳn cùng-mặt-khác-lần-chụp.
    assert cmp_smooth["ref_combined_cue"] > cmp_same_face["ref_combined_cue"]
    # Giá trị đo trên chính cặp synthetic này chứng minh ngưỡng REFERENCE_CUE_MIN:
    # cặp smooth vượt ngưỡng, cặp cùng-mặt nằm dưới ngưỡng.
    assert cmp_smooth["ref_combined_cue"] >= REFERENCE_CUE_MIN
    assert cmp_same_face["ref_combined_cue"] < REFERENCE_CUE_MIN
    for key in ("ref_residual_gap", "ref_lbp_chi2", "ref_fft_peak_delta"):
        assert key in cmp_smooth
        assert cmp_smooth[key] >= 0.0


def test_apply_reference_cue_bumps_no_indication_to_inconclusive() -> None:
    ref = _doc(_face_frame(smooth_face=False), "ref.png", sha="a" * 64)
    sus = _doc(_face_frame(smooth_face=True), "sus.png", sha="b" * 64)
    base = _base_result(score=0.20, verdict=Verdict.NO_INDICATION)

    out = apply_reference_cue(base, ref, sus)

    assert out.verdict is Verdict.INCONCLUSIVE
    assert out.verdict is not Verdict.SUSPICIOUS_INDICATION
    assert out.method is MethodName.DEEPFAKE_CUES
    assert 0.20 < out.score <= 0.20 + REFERENCE_SCORE_MAX_DELTA + 1e-6
    assert out.score - 0.20 <= REFERENCE_SCORE_MAX_DELTA + 1e-6


def test_apply_reference_cue_same_frame_leaves_score_and_verdict() -> None:
    frame = _face_frame()
    ref = _doc(frame, "ref.png", sha="a" * 64)
    sus = _doc(frame.copy(), "sus.png", sha="b" * 64)
    base = _base_result(score=0.20, verdict=Verdict.NO_INDICATION)

    out = apply_reference_cue(base, ref, sus)

    assert out.verdict is Verdict.NO_INDICATION
    assert out.score == 0.20


def test_apply_reference_cue_never_sets_suspicious() -> None:
    # Kịch bản khác biệt cực đoan: nhiễu thuần vs mặt phẳng — cue bão hòa.
    rng = np.random.default_rng(99)
    ref = _doc(
        rng.integers(0, 255, (320, 320, 3), dtype=np.uint8), "ref.png", sha="a" * 64
    )
    sus = _doc(np.full((320, 320, 3), 180, dtype=np.uint8), "sus.png", sha="b" * 64)
    base = _base_result(score=0.05, verdict=Verdict.NO_INDICATION)

    out = apply_reference_cue(base, ref, sus)
    assert out.verdict in {Verdict.NO_INDICATION, Verdict.INCONCLUSIVE}
    assert out.verdict is not Verdict.SUSPICIOUS_INDICATION
    assert out.score <= 0.05 + REFERENCE_SCORE_MAX_DELTA + 1e-6


def test_apply_reference_cue_caps_score_at_one() -> None:
    ref = _doc(_face_frame(smooth_face=False), "ref.png", sha="a" * 64)
    sus = _doc(_face_frame(smooth_face=True), "sus.png", sha="b" * 64)
    base = _base_result(score=0.95, verdict=Verdict.NO_INDICATION)

    out = apply_reference_cue(base, ref, sus)
    assert out.score <= 1.0
    assert out.score <= 0.95 + REFERENCE_SCORE_MAX_DELTA + 1e-6


def test_apply_reference_cue_preserves_fields_and_clean_evidence() -> None:
    ref = _doc(_face_frame(smooth_face=False), "ref.png", sha="a" * 64)
    sus = _doc(_face_frame(smooth_face=True), "sus.png", sha="b" * 64)
    base = _base_result(score=0.20, verdict=Verdict.NO_INDICATION)

    out = apply_reference_cue(base, ref, sus)

    # mask/heatmap/score_definition/limitations gốc được sao chép nguyên.
    assert np.array_equal(out.mask, base.mask)
    assert np.array_equal(out.heatmap, base.heatmap)
    assert out.score_definition == base.score_definition
    assert out.params_hash == base.params_hash
    assert "tín hiệu khám phá" in out.limitations
    assert len(out.limitations) > len(base.limitations)

    # Evidence mới: chỉ sha256 + số đo cue — không đường dẫn, không flag.
    new_keys = set(out.evidence) - set(base.evidence)
    assert new_keys == {
        "reference_sha256",
        "suspect_sha256",
        "ref_combined_cue",
        "ref_residual_gap",
        "ref_lbp_chi2",
        "ref_fft_peak_delta",
    }
    for key in new_keys:
        v = out.evidence[key]
        if key.endswith("sha256"):
            assert isinstance(v, str) and _SHA_RE.match(v)
        else:
            assert isinstance(v, (int, float)) and not isinstance(v, bool)


def test_apply_reference_cue_keeps_other_verdicts() -> None:
    ref = _doc(_face_frame(smooth_face=False), "ref.png", sha="a" * 64)
    sus = _doc(_face_frame(smooth_face=True), "sus.png", sha="b" * 64)

    for verdict in (Verdict.INCONCLUSIVE, Verdict.NOT_APPLICABLE, Verdict.ERROR):
        base = _base_result(score=0.5, verdict=verdict)
        out = apply_reference_cue(base, ref, sus)
        assert out.verdict is verdict
