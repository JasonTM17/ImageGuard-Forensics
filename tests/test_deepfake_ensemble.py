"""Contract & behavioral tests cho DeepfakeEnsemble (``deepfake_ensemble``).

Unit thuần cho hàm gộp điểm (không cần model thật) + behavioral test chạy
``analyze`` với các member giả monkeypatch — không cần weights ONNX/YuNet.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.evaluation import deepfake_ensemble_study as ens_study
from imageguard.forensics.fusion import AUX_METHODS, combine
from imageguard.forensics.ml import deepfake_ensemble as ens
from imageguard.forensics.ml.deepfake_ensemble import (
    DeepfakeEnsemble,
    combine_members,
    member_score,
)
from imageguard.forensics.registry import load_method
from imageguard.services.analysis_service import _SPEC_TO_METHOD
from tests.test_deepfake_cues import _doc, _synthetic_face_frame


def _result(score, verdict, *, method=MethodName.DEEPFAKE_CUES, mask=None) -> AlgorithmResult:
    return AlgorithmResult(
        method=method, verdict=verdict,
        score=score, score_definition="fake-direction index [0,1] for tests",
        mask=mask, evidence={},
    )


class _FakeMember:
    def __init__(self, result: AlgorithmResult | None = None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc

    def analyze(self, doc, cfg, cancel=None) -> AlgorithmResult:
        if self._exc is not None:
            raise self._exc
        return self._result


# ---------- member_score ----------


def test_member_score_contract() -> None:
    assert member_score(_result(0.42, Verdict.SUSPICIOUS_INDICATION)) == pytest.approx(0.42)
    assert member_score(_result(0.0, Verdict.NO_INDICATION)) == 0.0
    assert member_score(_result(1.0, Verdict.INCONCLUSIVE)) == 1.0
    assert member_score(_result(None, Verdict.INCONCLUSIVE)) is None
    assert member_score(_result(0.5, Verdict.ERROR)) is None
    assert member_score(_result(0.5, Verdict.NOT_APPLICABLE)) is None
    assert member_score(_result(1.5, Verdict.SUSPICIOUS_INDICATION)) is None
    assert member_score(_result(-0.1, Verdict.NO_INDICATION)) is None
    assert member_score(_result(float("nan"), Verdict.INCONCLUSIVE)) is None
    assert member_score(_result(True, Verdict.INCONCLUSIVE)) is None


# ---------- combine_members ----------


def test_combine_weighted_mean_and_verdicts() -> None:
    weights = {"a": 1.0, "b": 2.0, "c": 1.5}
    out = combine_members(
        {"a": 0.9, "b": 0.8, "c": 0.85}, weights,
        min_members=2, fake_thresh=0.7, real_thresh=0.3,
    )
    assert out["combined"] == pytest.approx((0.9 * 1 + 0.8 * 2 + 0.85 * 1.5) / 4.5)
    assert out["verdict"] is Verdict.SUSPICIOUS_INDICATION
    assert out["agreement_fake_frac"] == pytest.approx(1.0)
    assert out["members_available"] == 3

    out_low = combine_members(
        {"a": 0.1, "b": 0.2, "c": 0.15}, weights,
        min_members=2, fake_thresh=0.7, real_thresh=0.3,
    )
    assert out_low["verdict"] is Verdict.NO_INDICATION
    assert out_low["agreement_fake_frac"] == pytest.approx(0.0)

    out_mid = combine_members(
        {"a": 0.5, "b": 0.5, "c": 0.5}, weights,
        min_members=2, fake_thresh=0.7, real_thresh=0.3,
    )
    assert out_mid["verdict"] is Verdict.INCONCLUSIVE


def test_combine_excludes_zero_weight_and_handles_empty() -> None:
    weights = {"a": 1.0, "b": 0.0}
    out = combine_members({"a": 0.9, "b": 0.9}, weights,
                          min_members=1, fake_thresh=0.7, real_thresh=0.3)
    assert out["combined"] == pytest.approx(0.9)
    assert out["weights_used"] == {"a": 1.0}

    empty = combine_members({}, weights, min_members=1, fake_thresh=0.7, real_thresh=0.3)
    assert empty["combined"] is None
    assert empty["verdict"] is Verdict.NOT_APPLICABLE

    below_min = combine_members({"a": 0.9}, {"a": 1.0},
                                min_members=2, fake_thresh=0.7, real_thresh=0.3)
    assert below_min["combined"] == pytest.approx(0.9)
    assert below_min["verdict"] is Verdict.INCONCLUSIVE


# ---------- wiring ----------


def test_registry_and_service_mapping() -> None:
    assert isinstance(load_method("deepfake_ensemble"), DeepfakeEnsemble)
    assert _SPEC_TO_METHOD["deepfake_ensemble"] is MethodName.DEEPFAKE_ENSEMBLE
    assert MethodName.DEEPFAKE_ENSEMBLE not in AUX_METHODS


def test_fusion_promotes_ensemble_suspicious() -> None:
    r = _result(0.9, Verdict.SUSPICIOUS_INDICATION, method=MethodName.DEEPFAKE_ENSEMBLE)
    verdict, signals, _ = combine([r])
    assert verdict is Verdict.SUSPICIOUS_INDICATION
    assert signals == ["deepfake_ensemble"]


def _patch_members(monkeypatch, cues=None, ml=None, screen=None) -> None:
    monkeypatch.setattr(ens, "DeepfakeCues", lambda: cues)
    monkeypatch.setattr(ens, "DeepfakeGenConViT", lambda: ml)
    monkeypatch.setattr(ens, "DeepfakeScreenCoC", lambda: screen)


def test_analyze_combines_available_members(monkeypatch) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    _patch_members(
        monkeypatch,
        cues=_FakeMember(_result(0.9, Verdict.INCONCLUSIVE)),
        ml=_FakeMember(_result(0.8, Verdict.SUSPICIOUS_INDICATION,
                               method=MethodName.DEEPFAKE_ML, mask=mask)),
        screen=_FakeMember(_result(0.85, Verdict.SUSPICIOUS_INDICATION,
                                   method=MethodName.DEEPFAKE_SCREEN_COC)),
    )
    cfg = load_thresholds()
    res = DeepfakeEnsemble().analyze(_doc(_synthetic_face_frame()), cfg)
    assert res.method is MethodName.DEEPFAKE_ENSEMBLE
    assert res.verdict is Verdict.SUSPICIOUS_INDICATION
    assert res.score == pytest.approx((0.9 * 1.0 + 0.8 * 2.0 + 0.85 * 1.5) / 4.5, abs=1e-3)
    ev = res.evidence
    assert ev["members_available"] == 3
    assert ev["agreement_fake_frac"] == pytest.approx(1.0)
    assert ev["members"]["deepfake_ml"]["available"] is True
    assert ev["members"]["deepfake_ml"]["score"] == pytest.approx(0.8)
    assert ev["mask_from"] == "deepfake_ml"
    assert res.mask is mask
    assert res.score_definition


def test_analyze_excludes_unavailable_members(monkeypatch) -> None:
    _patch_members(
        monkeypatch,
        cues=_FakeMember(_result(None, Verdict.NOT_APPLICABLE)),
        ml=_FakeMember(_result(0.9, Verdict.SUSPICIOUS_INDICATION, method=MethodName.DEEPFAKE_ML)),
        screen=_FakeMember(_result(0.8, Verdict.SUSPICIOUS_INDICATION,
                                   method=MethodName.DEEPFAKE_SCREEN_COC)),
    )
    res = DeepfakeEnsemble().analyze(_doc(_synthetic_face_frame()), load_thresholds())
    assert res.verdict is Verdict.SUSPICIOUS_INDICATION
    assert res.evidence["members"]["deepfake_cues"]["available"] is False
    assert res.evidence["members"]["deepfake_cues"]["reason"] == "not_applicable"
    assert res.evidence["members_available"] == 2
    # chỉ 2 member: (0.9*2 + 0.8*1.5)/3.5
    assert res.score == pytest.approx((0.9 * 2.0 + 0.8 * 1.5) / 3.5, abs=1e-3)


def test_analyze_below_min_members_is_inconclusive(monkeypatch) -> None:
    _patch_members(
        monkeypatch,
        cues=_FakeMember(_result(None, Verdict.NOT_APPLICABLE)),
        ml=_FakeMember(_result(0.99, Verdict.SUSPICIOUS_INDICATION, method=MethodName.DEEPFAKE_ML)),
        screen=_FakeMember(_result(None, Verdict.ERROR, method=MethodName.DEEPFAKE_SCREEN_COC)),
    )
    res = DeepfakeEnsemble().analyze(_doc(_synthetic_face_frame()), load_thresholds())
    assert res.verdict is Verdict.INCONCLUSIVE
    assert res.evidence["members_available"] == 1
    assert any("detector khả dụng" in lim for lim in res.limitations)


def test_analyze_promotes_when_ml_member_suspicious_but_mean_is_gray(monkeypatch) -> None:
    """Cues/GenConViT gray-zone scores must not hide a CoC/ML SUSPICIOUS member."""
    _patch_members(
        monkeypatch,
        cues=_FakeMember(_result(0.39, Verdict.NO_INDICATION)),
        ml=_FakeMember(_result(0.34, Verdict.INCONCLUSIVE, method=MethodName.DEEPFAKE_ML)),
        screen=_FakeMember(_result(0.99, Verdict.SUSPICIOUS_INDICATION,
                                   method=MethodName.DEEPFAKE_SCREEN_COC)),
    )
    res = DeepfakeEnsemble().analyze(_doc(_synthetic_face_frame()), load_thresholds())
    assert res.verdict is Verdict.SUSPICIOUS_INDICATION
    assert res.evidence["promoted_from_member"] == "deepfake_screen"
    assert res.score is not None and res.score < load_thresholds().value("deepfake_ensemble", "fake_thresh")
    overall, signals, _ = combine([res])
    assert overall is Verdict.SUSPICIOUS_INDICATION
    assert "deepfake_ensemble" in signals


def test_analyze_no_members_not_applicable(monkeypatch) -> None:
    _patch_members(
        monkeypatch,
        cues=_FakeMember(_result(None, Verdict.NOT_APPLICABLE)),
        ml=_FakeMember(exc=RuntimeError("boom")),
        screen=_FakeMember(_result(None, Verdict.NOT_APPLICABLE)),
    )
    res = DeepfakeEnsemble().analyze(_doc(_synthetic_face_frame()), load_thresholds())
    assert res.verdict is Verdict.NOT_APPLICABLE
    assert res.score is None
    assert res.evidence["members"]["deepfake_ml"]["reason"] == "error:RuntimeError"


def test_analyze_with_learned_model_uses_dip_weight(monkeypatch) -> None:
    monkeypatch.setattr(
        ens, "LearnedDeepfakeCues",
        lambda model: _FakeMember(_result(0.95, Verdict.INCONCLUSIVE)),
    )
    monkeypatch.setattr(ens, "DeepfakeGenConViT",
                        lambda: _FakeMember(_result(0.2, Verdict.NO_INDICATION, method=MethodName.DEEPFAKE_ML)))
    monkeypatch.setattr(ens, "DeepfakeScreenCoC",
                        lambda: _FakeMember(_result(0.2, Verdict.NO_INDICATION, method=MethodName.DEEPFAKE_SCREEN_COC)))
    model = {"model_sha256": "a" * 64}
    res = DeepfakeEnsemble(model).analyze(_doc(_synthetic_face_frame()), load_thresholds())
    ev = res.evidence
    assert ev["learned_model"] is True
    assert ev["model_sha256"] == "a" * 64
    assert "learned_dip" in ev["members"]
    assert "deepfake_cues" not in ev["members"]
    assert ev["members"]["learned_dip"]["weight"] == pytest.approx(
        load_thresholds().value("deepfake_ensemble", "weight_learned_dip")
    )
    # (0.95*2.0 + 0.2*2.0 + 0.2*1.5)/5.5 ≈ 0.4636 → vùng xám
    assert res.verdict is Verdict.INCONCLUSIVE
    assert res.score == pytest.approx((0.95 * 2.0 + 0.2 * 2.0 + 0.2 * 1.5) / 5.5, abs=1e-3)


# ---------- evaluation study ----------


def test_classification_metrics_counts() -> None:
    m = ens_study.classification_metrics([0.9, 0.8, 0.2, 0.1], [True, True, False, False], threshold=0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 0, 2, 0)
    assert m["accuracy"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0
    assert m["f1"] == 1.0 and m["specificity"] == 1.0
    m2 = ens_study.classification_metrics([0.9, 0.4, 0.6, 0.1], [True, True, False, False], threshold=0.5)
    assert (m2["tp"], m2["fp"], m2["tn"], m2["fn"]) == (1, 1, 1, 1)
    assert m2["accuracy"] == 0.5
    with pytest.raises(ValueError):
        ens_study.classification_metrics([0.1], [True, False], threshold=0.5)


def _write_dataset(root: Path) -> dict:
    for name in ("real.png", "fake.png"):
        cv2.imwrite(str(root / name), _synthetic_face_frame(smooth_face=(name == "fake.png")))
    manifest = {
        "version": 1,
        "samples": [
            {"id": "r1", "path": "real.png", "label": "real", "split": "test", "source_group": "g-real"},
            {"id": "f1", "path": "fake.png", "label": "fake", "split": "test", "source_group": "g-fake"},
        ],
    }
    return manifest


def test_ensemble_study_runs_detectors_and_ensemble(monkeypatch, tmp_path) -> None:
    from imageguard.evaluation import deepfake_ensemble_study as study

    class _ByName:
        def analyze(self, doc, cfg, cancel=None):
            fake = doc.file_name.startswith("fake")
            return AlgorithmResult(
                method=MethodName.DEEPFAKE_CUES,
                verdict=Verdict.SUSPICIOUS_INDICATION if fake else Verdict.NO_INDICATION,
                score=0.9 if fake else 0.1, score_definition="fake-direction index for tests",
            )

    monkeypatch.setattr(study, "_build_detectors", lambda model: {"cues": _ByName(), "ml": _ByName(), "screen": _ByName()})

    class _FakeEnsemble:
        def __init__(self, model=None) -> None:
            pass

        def analyze(self, doc, cfg, cancel=None):
            fake = doc.file_name.startswith("fake")
            return AlgorithmResult(
                method=MethodName.DEEPFAKE_ENSEMBLE,
                verdict=Verdict.SUSPICIOUS_INDICATION if fake else Verdict.NO_INDICATION,
                score=0.85 if fake else 0.15, score_definition="weighted mean for tests",
                evidence={"members_available": 3, "agreement_fake_frac": 1.0 if fake else 0.0},
            )

    monkeypatch.setattr(study, "DeepfakeEnsemble", _FakeEnsemble)
    manifest = _write_dataset(tmp_path)
    report = study.run_deepfake_ensemble_study(manifest, tmp_path, split="test")
    assert report["protocol"] == "deepfake-ensemble-study-v1"
    assert set(report["detectors"]) == {"deepfake_cues", "deepfake_ml", "deepfake_screen", "deepfake_ensemble"}
    for summary in report["detectors"].values():
        assert summary["coverage"] == 1.0
        assert summary["score_roc_auc_eligible"] == pytest.approx(1.0)
        m = summary["classification_at_threshold"]
        assert m["accuracy"] == 1.0 and m["f1"] == 1.0
    ens_rows = {row["id"]: row["detectors"]["deepfake_ensemble"] for row in report["samples"]}
    assert ens_rows["f1"]["score"] == pytest.approx(0.85)
    assert ens_rows["r1"]["score"] == pytest.approx(0.15)
    json.dumps(report, allow_nan=False)  # báo cáo phải JSON-able


def test_ensemble_study_rejects_bad_split_and_empty_selection(tmp_path) -> None:
    manifest = _write_dataset(tmp_path)
    with pytest.raises(ValueError, match="split"):
        ens_study.run_deepfake_ensemble_study(manifest, tmp_path, split="train")  # type: ignore[arg-type]
    dev_only = {**manifest, "samples": [dict(s, split="dev") for s in manifest["samples"]]}
    with pytest.raises(ValueError, match="no test samples"):
        ens_study.run_deepfake_ensemble_study(dev_only, tmp_path, split="test")


def test_ensemble_study_rejects_model_on_dev(tmp_path) -> None:
    manifest = _write_dataset(tmp_path)
    dev_manifest = {**manifest, "samples": [dict(s, split="dev") for s in manifest["samples"]]}
    with pytest.raises(ValueError, match="only be evaluated on test"):
        ens_study.run_deepfake_ensemble_study(dev_manifest, tmp_path, split="dev", model={"fake": True})
