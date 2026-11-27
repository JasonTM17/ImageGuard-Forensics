"""Regression cases for applicability, configuration and forensic coordinates."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.forensics.auxiliary import deepfake_cues as dc
from imageguard.forensics.fusion import combine
from tests.test_deepfake_cues import _doc, _synthetic_face_frame


def _noise(shape=(256, 256, 3), seed=91):
    return np.clip(np.random.default_rng(seed).normal(120, 15, shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "0.35"])
@pytest.mark.parametrize("schema", [False, True])
def test_invalid_threshold_cannot_silently_disable_or_force_flags(value, schema):
    cfg = load_thresholds()
    cfg.raw["deepfake"]["score_thresh"] = {"value": value, "source": "test"} if schema else value
    with pytest.raises(ValueError, match="deepfake.score_thresh"):
        dc.DeepfakeCues().analyze(_doc(_synthetic_face_frame()), cfg)


def test_yaml_threshold_schema_and_legacy_scalar_have_same_behavior():
    cfg = load_thresholds()
    scalar_cfg = copy.deepcopy(cfg)
    scalar_cfg.raw["deepfake"]["score_thresh"] = cfg.value("deepfake", "score_thresh")
    face = _doc(_synthetic_face_frame(smooth_face=True))
    normal = dc.DeepfakeCues().analyze(face, cfg)
    legacy = dc.DeepfakeCues().analyze(face, scalar_cfg)
    assert normal.score == legacy.score
    assert normal.evidence["cue_flag"] == legacy.evidence["cue_flag"]
    assert normal.evidence["score_is_probability"] is False
    assert normal.evidence["calibration_status"] == "uncalibrated"


def test_panorama_is_one_image_without_explicit_layout(monkeypatch):
    seen = []

    def detect(arr, gray, block, cfg=None):
        seen.append(gray.shape)
        return (420, 48, 96, 128), True, "controlled_face"

    monkeypatch.setattr(dc, "_detect_face_roi", detect)
    result = dc.DeepfakeCues().analyze(_doc(_noise((240, 720, 3))), load_thresholds())
    assert seen == [(240, 720)]
    assert result.evidence["dual_camera_mode"] is False
    assert result.evidence["roi_rect"] == [420, 48, 96, 128]
    assert result.score is not None


def test_full_frame_face_cannot_use_itself_as_background(monkeypatch):
    monkeypatch.setattr(dc, "_detect_face_roi", lambda *_: ((0, 0, 256, 256), True, "forced"))
    cfg = load_thresholds()
    result = dc.DeepfakeCues().analyze(_doc(_noise()), cfg)
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.score is None and result.mask is None and result.heatmap is None
    assert result.evidence["unavailable_reason"] == "insufficient_background"
    assert result.evidence["cue_flag"] is False
    assert result.params_hash == cfg.snapshot_hash()


@pytest.mark.parametrize("rect,flat,reason", [
    ((32, 32, 40, 40), False, "roi_too_small"),
    ((64, 64, 96, 96), True, "low_contrast"),
])
def test_unusable_face_does_not_get_a_clean_or_fake_score(monkeypatch, rect, flat, reason):
    frame = _noise()
    if flat:
        x, y, w, h = rect
        frame[y:y + h, x:x + w] = 120
    monkeypatch.setattr(dc, "_detect_face_roi", lambda *_: (rect, True, "controlled_face"))
    result = dc.DeepfakeCues().analyze(_doc(frame), load_thresholds())
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.score is None and result.mask is None and result.heatmap is None
    assert result.evidence["cue_unavailable"] is True
    assert result.evidence["unavailable_reason"] == reason
    # Missing evidence must not be counted as a second auxiliary detection.
    other = AlgorithmResult(MethodName.NOISE_LOCAL, Verdict.INCONCLUSIVE)
    assert combine([result, other])[1] == []


def test_texture_family_alone_does_not_flag_a_face(monkeypatch):
    monkeypatch.setattr(dc, "_detect_face_roi", lambda *_: ((64, 64, 128, 128), True, "forced"))

    def strong_texture(gray, inside, outside, block):
        return 0.9, 0.6, 0.8, 0.2, np.full(inside.shape, 0.2, dtype=np.float32)

    monkeypatch.setattr(dc, "_lbp_texture_cues", strong_texture)
    monkeypatch.setattr(dc, "_fft_highfreq_cue", lambda *_: (0.0, 1.0, 1.0, 0.0))
    monkeypatch.setattr(dc, "_seam_canny_hough_cue", lambda *_: (0.0, 0))
    cfg = load_thresholds()
    cfg.raw["deepfake"]["score_thresh"]["value"] = 0.05
    result = dc.DeepfakeCues().analyze(_doc(_noise()), cfg)
    assert result.score > 0.05
    assert result.evidence["support_families"] == ["texture"]
    assert result.evidence["cue_flag"] is False
    assert result.verdict is Verdict.NO_INDICATION


def test_dual_prefers_usable_face_to_small_high_drop_face(monkeypatch):
    detections = iter([((64, 64, 96, 96), True, "valid"), ((100, 100, 40, 40), True, "small")])
    stats = iter([(5.0, 10.0, 0.5), (0.5, 10.0, 0.95)])
    monkeypatch.setattr(dc, "_detect_face_roi", lambda *_: next(detections))
    monkeypatch.setattr(dc, "_roi_sigma_stats", lambda *_: next(stats))
    result = dc.DeepfakeCues().analyze(_doc(_noise((256, 512, 3)), dual=True), load_thresholds())
    assert result.evidence["dual_roi_side"] == "left"
    assert result.evidence["cue_unavailable"] is False
    assert result.score is not None


def test_downscaled_roi_coordinates_refer_to_original_image(monkeypatch):
    monkeypatch.setattr(dc, "_detect_face_roi", lambda *_: ((48, 32, 96, 96), True, "forced"))
    cfg = load_thresholds()
    cfg.raw["limits"]["aux_max_side"]["value"] = 256
    result = dc.DeepfakeCues().analyze(_doc(_noise((512, 512, 3))), cfg)
    assert result.evidence["roi_rect_working"] == [48, 32, 96, 96]
    assert result.evidence["roi_rect"] == [96, 64, 192, 192]
    assert result.evidence["roi_coordinate_space"] == "original_pixels"


def test_block_map_does_not_stretch_into_unmeasured_remainder():
    doc = _doc(_noise((131, 133, 3)))
    mapped = dc._cue_map_to_doc(np.ones((8, 8), dtype=np.uint8), doc, None,
                                (131, 133), 0, 16)
    assert np.all(mapped[:128, :128] == 1)
    assert not np.any(mapped[128:, :])
    assert not np.any(mapped[:, 128:])
