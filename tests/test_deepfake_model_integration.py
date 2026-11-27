"""Runtime/model/holdout contracts; these controls do not measure real accuracy."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.evaluation.deepfake_study import run_deepfake_study
from imageguard.evaluation.deepfake_training import fit_model
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues
from imageguard.forensics.auxiliary.deepfake_model import (
    EXTRACTOR_VARIANT,
    FEATURE_NAMES,
    config_digest,
    save_model,
)
from imageguard.forensics.auxiliary.learned_deepfake import LearnedDeepfakeCues
from imageguard.services.analysis_service import AnalysisRequest, AnalysisService


def _features(level: float) -> dict:
    return dict.fromkeys(FEATURE_NAMES, level)


@pytest.fixture()
def dip_model():
    records = []
    for group in range(3):
        for label, level in (("real", 0.1), ("fake", 0.9)):
            sid = f"dev_{group}_{label}"
            records.append({"id": sid, "source_group": f"video_{group}", "label": label,
                            "split": "dev", "sha256": hashlib.sha256(sid.encode()).hexdigest(),
                            "features": _features(level + group * 0.01)})
    model, _ = fit_model(records, config_sha256=config_digest(load_thresholds().raw),
                         manifest_sha256="a" * 64, folds=3)
    return model


def _base(level: float, *, unavailable: bool = False) -> AlgorithmResult:
    return AlgorithmResult(
        method=MethodName.DEEPFAKE_CUES, verdict=Verdict.INCONCLUSIVE if unavailable else Verdict.NO_INDICATION,
        score=None if unavailable else 0.2, score_definition="synthetic heuristic control",
        evidence={**_features(level), "variant": EXTRACTOR_VARIANT, "face_detected": not unavailable,
                  "cue_unavailable": unavailable, "cue_flag": False,
                  "unavailable_reason": "no_face" if unavailable else None},
        heatmap=None if unavailable else np.ones((96, 96), np.float32),
    )


def _doc() -> ImageDocument:
    return ImageDocument(np.zeros((96, 96, 3), np.uint8), 96, 96, 3, "png", "c" * 64, 1, "test.png")


@pytest.mark.parametrize(("level", "flagged"), [(0.1, False), (0.9, True)])
def test_learned_decision_uses_selected_model_and_preserves_heuristic(level, flagged, dip_model, monkeypatch):
    base = _base(level)
    monkeypatch.setattr(DeepfakeCues, "analyze", lambda *_args: base)
    result = LearnedDeepfakeCues(dip_model).analyze(_doc(), load_thresholds())
    assert result.evidence["cue_flag"] is flagged
    assert (result.score >= 0.5) is flagged
    assert result.verdict is (Verdict.INCONCLUSIVE if flagged else Verdict.NO_INDICATION)
    assert result.evidence["heuristic_score"] == 0.2
    assert result.evidence["heuristic_cue_flag"] is False
    assert result.evidence["model_sha256"] == dip_model["model_sha256"]
    assert result.evidence["score_is_probability"] is False
    assert result.mask is None  # linear feature model does not localize a manipulated region
    assert result.evidence["heatmap_scope"] == "heuristic_texture_diagnostic_only"
    assert base.score == 0.2 and base.evidence["cue_flag"] is False


def test_learned_model_cannot_bypass_face_gate(dip_model, monkeypatch):
    monkeypatch.setattr(DeepfakeCues, "analyze", lambda *_args: _base(0.9, unavailable=True))
    result = LearnedDeepfakeCues(dip_model).analyze(_doc(), load_thresholds())
    assert result.score is None
    assert result.evidence["cue_flag"] is False
    assert result.evidence["model_status"] == "unavailable"
    assert result.evidence["unavailable_reason"] == "no_face"


def test_learned_model_rejects_unvalidated_camera_layout(dip_model, monkeypatch):
    monkeypatch.setattr(DeepfakeCues, "analyze", lambda *_args: _base(0.9))
    doc = _doc()
    doc.capture_layout = "dual_side_by_side"
    result = LearnedDeepfakeCues(dip_model).analyze(doc, load_thresholds())
    assert result.score is None and result.mask is None and result.heatmap is None
    assert result.evidence["cue_flag"] is False
    assert result.evidence["unavailable_reason"] == "unsupported_capture_layout"


def test_runtime_config_changes_do_not_silently_apply_old_model(dip_model):
    cfg = load_thresholds()
    cfg.raw["deepfake"]["min_face_side"]["value"] = 96
    with pytest.raises(ValueError, match="configuration"):
        LearnedDeepfakeCues(dip_model).analyze(_doc(), cfg)


def test_training_image_overlap_is_visible_in_app_result(dip_model, monkeypatch):
    monkeypatch.setattr(DeepfakeCues, "analyze", lambda *_args: _base(0.9))
    doc = _doc()
    doc.sha256 = dip_model["training"]["samples"][0]["sha256"]
    result = LearnedDeepfakeCues(dip_model).analyze(doc, load_thresholds())
    assert result.evidence["model_in_training_data"] is True


def _image(path: Path, level: int = 80) -> None:
    path.write_bytes(cv2.imencode(".png", np.full((96, 96, 3), level, np.uint8))[1].tobytes())


def _manifest(*, sid: str = "test_frame", group: str = "unseen_video", split: str = "test") -> dict:
    return {"version": 1, "samples": [{"id": sid, "path": "test.png", "source_group": group,
                                        "label": "fake", "split": split}]}


@pytest.mark.parametrize("overlap", ["id", "group", "bytes"])
def test_study_rejects_model_overlap_in_separate_test_manifest(tmp_path, dip_model, overlap):
    _image(tmp_path / "test.png")
    manifest = _manifest()
    if overlap == "id":
        manifest["samples"][0]["id"] = dip_model["training"]["sample_ids"][0]
    elif overlap == "group":
        manifest["samples"][0]["source_group"] = dip_model["training"]["source_groups"][0]
    else:
        changed = copy.deepcopy(dip_model)
        sha = hashlib.sha256((tmp_path / "test.png").read_bytes()).hexdigest()
        changed["training"]["samples"][0]["sha256"] = sha
        changed["training"]["image_hashes"] = sorted(row["sha256"] for row in changed["training"]["samples"])
        changed.pop("model_sha256")
        changed["model_sha256"] = config_digest(changed)
        dip_model = changed
    with pytest.raises(ValueError, match="overlaps model development"):
        run_deepfake_study(manifest, tmp_path, model=dip_model)


def test_model_study_is_test_only(tmp_path, dip_model):
    _image(tmp_path / "test.png")
    with pytest.raises(ValueError, match="only be evaluated on test"):
        run_deepfake_study(_manifest(split="dev"), tmp_path, split="dev", model=dip_model)


def test_model_study_exports_comparable_baseline_and_features(tmp_path, dip_model, monkeypatch):
    _image(tmp_path / "test.png")
    monkeypatch.setattr(DeepfakeCues, "analyze", lambda *_args: _base(0.9))
    report = run_deepfake_study(_manifest(), tmp_path, model=dip_model)
    row = report["samples"][0]
    assert row["features"] == _features(0.9)
    assert row["split"] == "test"
    assert row["cue_flag"] is True and row["heuristic_cue_flag"] is False
    assert report["summary"]["by_class"]["fake"]["n_cued"] == 1
    assert report["heuristic_baseline_summary"]["by_class"]["fake"]["n_cued"] == 0
    assert report["model_sha256"] == dip_model["model_sha256"]


def test_learned_auc_ranks_raw_margins_even_when_sigmoid_saturates():
    from imageguard.evaluation.deepfake_study import summarize_samples

    rows = [
        {"label": "real", "exclusion_reason": None, "cue_flag": True,
         "score": 1.0, "model_decision_score": 38.0},
        {"label": "fake", "exclusion_reason": None, "cue_flag": True,
         "score": 1.0, "model_decision_score": 45.0},
    ]
    assert summarize_samples(rows, ranking_key="model_decision_score")["score_roc_auc_eligible"] == 1.0
    assert summarize_samples(rows)["score_roc_auc_eligible"] == 0.5


def test_service_reloads_selected_model_and_rejects_changed_selection(tmp_path, dip_model, monkeypatch):
    path = tmp_path / "test.png"
    _image(path)
    model_path = tmp_path / "model.json"
    save_model(model_path, dip_model)
    monkeypatch.setattr(DeepfakeCues, "analyze", lambda *_args: _base(0.9))
    request = AnalysisRequest(path, methods=["deepfake"], include_exif=False,
                              deepfake_model_path=model_path, deepfake_model_sha256=dip_model["model_sha256"])
    result = AnalysisService().analyze(request).methods[0]
    assert result.evidence["model_status"] == "applied"
    request.deepfake_model_sha256 = "f" * 64
    rejected = AnalysisService().analyze(request).methods[0]
    assert rejected.verdict is Verdict.ERROR
    assert "changed since selection" in rejected.evidence["error"]


def test_training_cli_outputs_roll_back_without_overwriting_existing_file(tmp_path):
    from scripts.train_deepfake_model import _save_outputs

    model, report = tmp_path / "model.json", tmp_path / "report.json"
    report.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _save_outputs([(model, {"new": True}), (report, {"new": True})])
    assert not model.exists()
    assert report.read_text(encoding="utf-8") == "keep"


def test_cli_requires_explicit_deepfake_method_for_model(tmp_path, monkeypatch, capsys):
    from imageguard.cli import main

    monkeypatch.setattr("sys.argv", ["imageguard", "analyze", str(tmp_path / "test.png"),
                                   "--deepfake-model", str(tmp_path / "model.json"), "--no-persist"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "requires --methods deepfake" in capsys.readouterr().err


def test_cli_invalid_model_is_actionable_without_loading_image(tmp_path, monkeypatch, capsys):
    from imageguard.cli import main

    model = tmp_path / "model.json"
    model.write_text(json.dumps({"weights": "wrong schema"}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["imageguard", "analyze", str(tmp_path / "test.png"),
                                   "--methods", "deepfake", "--deepfake-model", str(model), "--no-persist"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "cannot use deepfake model" in capsys.readouterr().err
