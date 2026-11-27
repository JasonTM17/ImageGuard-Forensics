"""Face-localization controls; these synthetic images are not a real-face benchmark."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.core.config import load_thresholds
from imageguard.core.enums import Verdict
from imageguard.forensics.auxiliary import deepfake_cues as dc
from tests.test_deepfake_cues import _doc, _synthetic_face_frame


def _disable_haar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dc.cv2.data, "haarcascades", "")


def _skin_colored_object(kind: str, seed: int) -> np.ndarray:
    """Shading and texture on objects with skin-range color, with no eye pair."""
    rng = np.random.default_rng(seed)
    frame = np.full((320, 320, 3), (125, 90, 65), np.uint8)
    yy, xx = np.mgrid[:180, :140]
    shade = np.array([135.0, 168.0, 215.0]) + rng.uniform(-3, 3)
    if kind == "gradient_panel":
        level = rng.uniform(25, 40) * yy / 179
    elif kind == "wood_stripes":
        level = 12 * np.sin(xx / rng.uniform(4, 7)) + 8 * np.sin(yy / 20)
    elif kind == "horizontal_band":
        level = np.where((yy >= 45) & (yy <= 65), -35, 0)
    elif kind == "textured_panel":
        level = rng.normal(0, 14, (180, 140))
    else:
        assert kind == "shaded_ball"
        mask = np.zeros((320, 320), np.uint8)
        cv2.circle(mask, (160, 160), 78, 255, -1)
        _, columns = np.mgrid[:320, :320]
        color = np.clip(shade + (25 * (columns - 160) / 78)[:, :, None], 0, 255)
        frame[mask > 0] = color.astype(np.uint8)[mask > 0]
        return frame
    frame[60:240, 90:230] = np.clip(shade + level[:, :, None], 0, 255).astype(np.uint8)
    return frame


def _frontal_face_control(shade, angle: float, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((320, 320, 3), (125, 90, 65), np.float32)
    mask = np.zeros((320, 320), np.uint8)
    cv2.ellipse(mask, (160, 150), (62, 82), 0, 0, 360, 255, -1)
    patch = np.empty_like(image)
    patch[:] = shade
    _, xx = np.mgrid[:320, :320]
    patch += ((xx - 160) * 8 / 62)[:, :, None]
    cv2.ellipse(patch, (137, 130), (9, 7), 0, 0, 360, (45, 65, 90), -1)
    cv2.ellipse(patch, (183, 130), (9, 7), 0, 0, 360, (45, 65, 90), -1)
    cv2.ellipse(patch, (160, 182), (16, 4), 0, 0, 360, (65, 85, 115), -1)
    patch += rng.normal(0, noise, patch.shape)
    image[mask > 0] = patch[mask > 0]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.warpAffine(
        image, cv2.getRotationMatrix2D((160, 150), angle, 1.0), (320, 320),
        borderValue=(125, 90, 65),
    )


@pytest.mark.parametrize("kind", [
    "gradient_panel", "wood_stripes", "horizontal_band", "textured_panel", "shaded_ball",
])
def test_skin_range_objects_are_not_enough_to_localize_a_face(monkeypatch, kind):
    _disable_haar(monkeypatch)
    for seed in (12, 87, 406, 1091):
        frame = _skin_colored_object(kind, seed)
        _, detected, source = dc._detect_face_roi(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 16)
        assert detected is False, (kind, seed)
        assert source == "central_fallback"
        result = dc.DeepfakeCues().analyze(_doc(frame), load_thresholds())
        assert result.score is None
        assert result.evidence["cue_flag"] is False
        assert result.evidence["unavailable_reason"] == "no_face"


@pytest.mark.parametrize("shade", [(135, 168, 215), (80, 115, 170), (115, 150, 200)])
@pytest.mark.parametrize("angle", [-8.0, 0.0, 8.0])
@pytest.mark.parametrize("noise", [2.0, 12.0])
def test_bilateral_eye_structure_survives_small_control_variations(monkeypatch, shade, angle, noise):
    _disable_haar(monkeypatch)
    frame = _frontal_face_control(shade, angle, noise, seed=601)
    rect, detected, source = dc._detect_face_roi(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 16)
    assert detected is True
    assert source == "skin_haar_dip"
    x, y, width, height = rect
    assert x < 137 and 183 < x + width
    assert y < 130 and 182 < y + height


def _haar_candidates(monkeypatch: pytest.MonkeyPatch, rectangles: list[tuple]) -> None:
    class ControlledCascade:
        def empty(self):
            return False

        def detectMultiScale(self, *_args, **_kwargs):
            return np.asarray(rectangles, dtype=np.int32)

    # Some supported OpenCV builds omit objdetect/CascadeClassifier entirely.
    # These tests exercise candidate selection independent of that packaging.
    monkeypatch.setattr(dc.cv2, "CascadeClassifier", lambda _path: ControlledCascade(), raising=False)


def test_haar_chooses_measurable_face_over_larger_flat_detection(monkeypatch):
    large, useful = (16, 24, 192, 224), (280, 60, 112, 144)
    gray = np.full((320, 480), 120, np.uint8)
    rng = np.random.default_rng(243)
    gray[60:204, 280:392] = np.clip(rng.normal(128, 14, (144, 112)), 0, 255).astype(np.uint8)
    _haar_candidates(monkeypatch, [large, useful])
    rect, detected, source = dc._detect_face_roi(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray, 16)
    assert (rect, detected, source) == (useful, True, "haar_cascade")


def test_haar_candidate_ranking_respects_contrast_override(monkeypatch):
    large, high_contrast = (16, 24, 192, 224), (280, 60, 112, 144)
    gray = np.full((320, 480), 120, np.uint8)
    gray[24:248, 16:208] = np.linspace(110, 120, 192).astype(np.uint8)[None, :]
    gray[60:204, 280:392] = np.linspace(70, 150, 112).astype(np.uint8)[None, :]
    _haar_candidates(monkeypatch, [large, high_contrast])
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cfg = load_thresholds()
    assert dc._detect_face_roi(image, gray, 16, cfg)[0] == large
    cfg.raw["deepfake"]["min_roi_contrast"]["value"] = 20
    assert dc._detect_face_roi(image, gray, 16, cfg)[0] == high_contrast


def test_haar_candidate_ranking_respects_size_override_after_clipping(monkeypatch):
    # First detection extends below the image and retains only 80px height.
    clipped = (16, 240, 200, 80)
    intact = (280, 60, 96, 128)
    gray = np.tile(np.arange(480, dtype=np.uint16) % 160 + 40, (320, 1)).astype(np.uint8)
    _haar_candidates(monkeypatch, [(16, 240, 200, 200), intact])
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cfg = load_thresholds()
    assert dc._detect_face_roi(image, gray, 16, cfg)[0] == clipped
    cfg.raw["deepfake"]["min_face_side"]["value"] = 90
    assert dc._detect_face_roi(image, gray, 16, cfg)[0] == intact


def test_haar_candidate_ranking_respects_reference_block_override(monkeypatch):
    large, useful = (0, 0, 288, 288), (112, 112, 96, 96)
    gray = np.tile(np.arange(320, dtype=np.uint16) % 160 + 40, (320, 1)).astype(np.uint8)
    _haar_candidates(monkeypatch, [large, useful])
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cfg = load_thresholds()
    assert dc._detect_face_roi(image, gray, 16, cfg)[0] == large
    cfg.raw["deepfake"]["min_reference_blocks"]["value"] = 80
    assert dc._detect_face_roi(image, gray, 16, cfg)[0] == useful


def test_unusable_cascade_candidates_preserve_quality_reason(monkeypatch):
    _haar_candidates(monkeypatch, [(16, 24, 192, 224), (280, 60, 112, 144)])
    frame = np.full((320, 480, 3), 120, np.uint8)
    result = dc.DeepfakeCues().analyze(_doc(frame), load_thresholds())
    assert result.evidence["roi_source"] == "haar_cascade"
    assert result.evidence["unavailable_reason"] == "low_contrast"
    assert result.score is None
    assert result.verdict is Verdict.INCONCLUSIVE


def test_supported_opencv_runtime_has_a_loadable_haar_cascade():
    """A dependency upgrade must not silently remove the primary face detector."""
    cascade, status = dc._load_haar_cascade()
    assert status == "available", f"OpenCV {cv2.__version__}: {status}"
    assert cascade is not None and not cascade.empty()


@pytest.mark.parametrize("missing", ["classifier", "data", "load"])
def test_custom_build_without_haar_reports_fallback_explicitly(monkeypatch, missing):
    if missing == "classifier":
        monkeypatch.delattr(dc.cv2, "CascadeClassifier", raising=False)
        expected = "missing_classifier"
    elif missing == "data":
        monkeypatch.setattr(dc.cv2.data, "haarcascades", "")
        expected = "missing_cascade_data"
    else:
        def broken_classifier(_path):
            raise cv2.error("controlled cascade load failure")

        monkeypatch.setattr(dc.cv2, "CascadeClassifier", broken_classifier, raising=False)
        expected = "cascade_load_failed"
    frame = _synthetic_face_frame()
    result = dc.DeepfakeCues().analyze(_doc(frame), load_thresholds())
    assert result.evidence["haar_cascade_status"] == expected
    assert result.evidence["face_detector_backend"] == "skin_fallback_only"
    assert result.evidence["roi_source"] == "skin_haar_dip"
    assert any("OpenCV" in text and "Haar" in text for text in result.limitations)
