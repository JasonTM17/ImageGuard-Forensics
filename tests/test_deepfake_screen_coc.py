"""Contract tests cho DeepfakeScreenCoC (``deepfake_screen``)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.forensics.fusion import AUX_METHODS
from imageguard.forensics.ml import deepfake_screen_coc as dsc
from imageguard.forensics.ml.deepfake_screen_coc import (
    DeepfakeScreenCoC,
    align_face_coc,
    infer_fake_prob,
    preprocess_aligned,
    verdict_for_score,
)
from imageguard.forensics.registry import load_method
from tests.test_deepfake_cues import _doc

MODEL_PRESENT = (
    dsc.MODEL_PATH.is_file()
    and dsc.MODEL_PATH.stat().st_size == dsc.MODEL_BYTES
    and dsc.model_integrity_error() is None
)
needs_model = pytest.mark.skipif(not MODEL_PRESENT, reason="CoC ONNX weights not installed")


@needs_model
def test_coc_onnx_loads_and_returns_finite_probability() -> None:
    prob = infer_fake_prob(np.zeros((1, 3, 299, 299), dtype=np.float32))
    assert 0.0 <= prob <= 1.0


def test_preprocess_aligned_bgr_mean_half() -> None:
    crop = np.full((40, 60, 3), (255, 128, 0), dtype=np.uint8)
    tensor = preprocess_aligned(crop)
    assert tensor.shape == (1, 3, 299, 299)
    assert tensor.dtype == np.float32
    # Không đổi BGR sang RGB: kênh 0 là B=255 → +1; kênh 2 là R=0 → -1.
    assert float(tensor[0, 0, 0, 0]) == pytest.approx(1.0, abs=1e-5)
    assert float(tensor[0, 2, 0, 0]) == pytest.approx(-1.0, abs=1e-5)


def test_align_face_coc_shape_and_invalid_landmarks() -> None:
    img = np.zeros((180, 180, 3), dtype=np.uint8)
    pts = np.array(
        [[60, 70], [120, 70], [90, 95], [70, 130], [110, 130]],
        dtype=np.float64,
    )
    aligned = align_face_coc(img, pts)
    assert aligned is not None
    assert aligned.shape == (299, 299, 3)
    assert align_face_coc(img, pts[:4]) is None
    bad = pts.copy()
    bad[0, 0] = np.nan
    assert align_face_coc(img, bad) is None


def test_verdict_boundary_mapping() -> None:
    assert verdict_for_score(0.7, 0.7, 0.3) is Verdict.SUSPICIOUS_INDICATION
    assert verdict_for_score(0.3, 0.7, 0.3) is Verdict.NO_INDICATION
    assert verdict_for_score(0.5, 0.7, 0.3) is Verdict.INCONCLUSIVE


def test_infer_fake_prob_rejects_invalid_output(monkeypatch) -> None:
    class BadSession:
        def run(self, _outputs, _inputs):
            return (np.array([[0.0, np.nan]], dtype=np.float32),)

    monkeypatch.setattr(dsc, "_session", lambda: BadSession())
    with pytest.raises(ValueError, match="invalid CoC logits"):
        infer_fake_prob(np.zeros((1, 3, 299, 299), dtype=np.float32))


def test_deepfake_screen_not_in_aux_methods() -> None:
    assert MethodName.DEEPFAKE_SCREEN_COC not in AUX_METHODS


def test_registry_loads_deepfake_screen() -> None:
    assert type(load_method("deepfake_screen")).__name__ == "DeepfakeScreenCoC"


def test_missing_model_is_not_applicable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dsc, "MODEL_PATH", tmp_path / "missing.onnx")
    result = DeepfakeScreenCoC().analyze(
        _doc(np.zeros((96, 96, 3), dtype=np.uint8)),
        load_thresholds(),
        threading.Event(),
    )
    assert result.method is MethodName.DEEPFAKE_SCREEN_COC
    assert result.verdict is Verdict.NOT_APPLICABLE
    assert result.score is None
    assert result.evidence["model_available"] is False
