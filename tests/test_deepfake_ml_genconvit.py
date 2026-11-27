"""Contract & behavioral tests cho DeepfakeGenConViT (``deepfake_ml``).

Hai lớp: unit thuần (không cần model — monkeypatch) và model-dependent
(skip khi file model/YuNet vắng mặt, vd CI không bundle weights).
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.forensics.fusion import AUX_METHODS, combine
from imageguard.forensics.ml import deepfake_genconvit as dgc
from imageguard.forensics.ml.deepfake_genconvit import (
    DeepfakeGenConViT,
    aggregate_fake_frac,
    preprocess_crop,
    verdict_for_score,
)
from imageguard.forensics.registry import load_method
from tests.test_deepfake_cues import _doc, _synthetic_face_frame

MODEL_PRESENT = (
    dgc.MODEL_PATH.is_file()
    and Path(dgc.MODEL_PATH).stat().st_size == dgc.MODEL_BYTES
    and dgc.yunet_integrity_error() is None
    and dgc.model_integrity_error() is None
)
needs_model = pytest.mark.skipif(not MODEL_PRESENT, reason="model weights not installed")


# ---------- unit thuần ----------


def test_preprocess_crop_matches_imagenet_pipeline() -> None:
    crop = np.full((40, 60, 3), (255, 128, 0), dtype=np.uint8)  # BGR cam
    tensor = preprocess_crop(crop)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    # kênh R=0 → (0/255 − 0.485)/0.229 ≈ −2.118; kênh B=255 → (1 − 0.406)/0.225 ≈ 2.64
    r = tensor[0, 0, 0, 0]
    b = tensor[0, 2, 0, 0]
    assert abs(float(r) - (0.0 - 0.485) / 0.229) < 1e-4
    assert abs(float(b) - (1.0 - 0.406) / 0.225) < 1e-4


def test_preprocess_crop_skips_resize_when_already_224() -> None:
    crop = np.zeros((224, 224, 3), dtype=np.uint8)
    tensor = preprocess_crop(crop)
    assert tensor.shape == (1, 3, 224, 224)


def test_aggregate_fake_frac_author_rule() -> None:
    # sigmoid fake > sigmoid real → frac > 0.5 (quy tắc FAKE của tác giả)
    sig = np.array([[0.9, 0.1], [0.8, 0.2]])
    frac, mf, mr = aggregate_fake_frac(sig)
    assert frac == pytest.approx(0.85 / 1.0, abs=1e-6)
    assert mf == pytest.approx(0.85) and mr == pytest.approx(0.15)
    # cân bằng → 0.5
    frac_eq, *_ = aggregate_fake_frac(np.array([[0.5, 0.5]]))
    assert frac_eq == pytest.approx(0.5)


def test_verdict_boundary_mapping() -> None:
    assert verdict_for_score(0.7, 0.7, 0.3) is Verdict.SUSPICIOUS_INDICATION
    assert verdict_for_score(0.99, 0.7, 0.3) is Verdict.SUSPICIOUS_INDICATION
    assert verdict_for_score(0.3, 0.7, 0.3) is Verdict.NO_INDICATION
    assert verdict_for_score(0.0, 0.7, 0.3) is Verdict.NO_INDICATION
    assert verdict_for_score(0.5, 0.7, 0.3) is Verdict.INCONCLUSIVE
    assert verdict_for_score(0.6999, 0.7, 0.3) is Verdict.INCONCLUSIVE


def test_deepfake_ml_not_in_aux_methods() -> None:
    # Method ML được phép trả SUSPICIOUS — không thuộc contract aux exploratory.
    assert MethodName.DEEPFAKE_ML not in AUX_METHODS


def test_fusion_promotes_deepfake_ml_suspicious() -> None:
    frame = _synthetic_face_frame(smooth_face=False)
    doc = _doc(frame, "face.png")
    cfg = load_thresholds()
    method = DeepfakeGenConViT()
    monkey_result = method.analyze(doc, cfg)  # verdict thật hoặc NOT_APPLICABLE — không quan trọng
    assert monkey_result.verdict is not Verdict.ERROR
    # Contract fusion: bất kỳ SUSPICIOUS nào từ method cũng promote overall.
    from imageguard.core.models import AlgorithmResult

    forged = AlgorithmResult(method=MethodName.DEEPFAKE_ML, verdict=Verdict.SUSPICIOUS_INDICATION)
    overall, _, _ = combine([forged])
    assert overall is Verdict.SUSPICIOUS_INDICATION


def test_missing_model_returns_not_applicable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(dgc, "MODEL_PATH", tmp_path / "nope.onnx")
    res = DeepfakeGenConViT().analyze(_doc(np.zeros((256, 256, 3), np.uint8)), load_thresholds())
    assert res.verdict is Verdict.NOT_APPLICABLE
    assert res.evidence["model_available"] is False
    assert any("docs/third-party-models.md" in item for item in res.limitations)


def test_integrity_mismatch_returns_error(monkeypatch, tmp_path) -> None:
    bad = tmp_path / "genconvit_ed_inference.onnx"
    bad.write_bytes(b"corrupt")
    monkeypatch.setattr(dgc, "MODEL_PATH", bad)
    res = DeepfakeGenConViT().analyze(_doc(np.zeros((256, 256, 3), np.uint8)), load_thresholds())
    assert res.verdict is Verdict.ERROR
    assert any("integrity" in item for item in res.limitations)


def test_cancel_returns_early(monkeypatch) -> None:
    if not MODEL_PRESENT:
        monkeypatch.setattr(dgc, "MODEL_PATH", Path("nonexistent"))
    ev = threading.Event()
    ev.set()
    res = DeepfakeGenConViT().analyze(
        _doc(np.zeros((128, 128, 3), np.uint8)), load_thresholds(), cancel=ev
    )
    assert res.verdict in {Verdict.INCONCLUSIVE, Verdict.NOT_APPLICABLE}


def test_registry_loads_deepfake_ml() -> None:
    assert type(load_method("deepfake_ml")).__name__ == "DeepfakeGenConViT"


# ---------- model-dependent (skip khi thiếu weights) ----------


@needs_model
def test_no_face_noise_returns_not_applicable() -> None:
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    res = DeepfakeGenConViT().analyze(_doc(noise, "noise.png"), load_thresholds())
    # Nhiễu thuần không có mặt — YuNet không được bịa mặt.
    assert res.verdict is Verdict.NOT_APPLICABLE
    assert res.evidence.get("face_detected") is False


@needs_model
def test_real_model_deterministic_on_same_input() -> None:
    frame = _synthetic_face_frame(smooth_face=False)
    doc = _doc(frame, "face.png")
    cfg = load_thresholds()
    r1 = DeepfakeGenConViT().analyze(doc, cfg)
    r2 = DeepfakeGenConViT().analyze(doc, cfg)
    assert r1.verdict == r2.verdict
    assert r1.score == r2.score
    assert r1.verdict in {
        Verdict.SUSPICIOUS_INDICATION,
        Verdict.INCONCLUSIVE,
        Verdict.NO_INDICATION,
        Verdict.NOT_APPLICABLE,
    }
    # NOT_APPLICABLE sớm không mang params_hash (convention deepfake_cues);
    # scored path mới có — quan trọng là deterministic giữa hai lần chạy.
    assert r1.params_hash == r2.params_hash
    if r1.verdict not in {Verdict.NOT_APPLICABLE}:
        assert r1.params_hash == cfg.snapshot_hash()


@needs_model
def test_mask_only_when_scored_and_suspicious_or_inconclusive() -> None:
    frame = _synthetic_face_frame(smooth_face=False)
    res = DeepfakeGenConViT().analyze(_doc(frame, "face.png"), load_thresholds())
    if res.verdict in {Verdict.SUSPICIOUS_INDICATION, Verdict.INCONCLUSIVE}:
        assert res.mask is not None and int(np.count_nonzero(res.mask)) > 0
    else:
        assert res.mask is None


@needs_model
def test_analysis_service_integration() -> None:
    from imageguard.services.analysis_service import AnalysisRequest, AnalysisService

    frame = _synthetic_face_frame(smooth_face=False)
    path = Path("data") / "_tmp_dfml_probe.png"
    path.parent.mkdir(exist_ok=True)
    ok, buf = cv2.imencode(".png", frame)
    assert ok
    path.write_bytes(buf.tobytes())
    try:
        report = AnalysisService().analyze(
            AnalysisRequest(path=path, methods=["deepfake_ml"], include_exif=False)
        )
        method = next(m for m in report.methods if m.method is MethodName.DEEPFAKE_ML)
        assert method.verdict is not Verdict.ERROR
        assert report.overall_verdict in {
            Verdict.SUSPICIOUS_INDICATION,
            Verdict.INCONCLUSIVE,
            Verdict.NO_INDICATION,
        }
    finally:
        path.unlink(missing_ok=True)
