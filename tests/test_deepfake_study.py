"""Evaluation contracts: abstentions, tied ranks, and leakage prevention."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.evaluation import deepfake_study as study


def _sample(sid: str, label: str = "real", split: str = "test", group: str | None = None) -> dict:
    return {"id": sid, "path": f"{sid}.png", "label": label, "split": split, "source_group": group or sid}


def _image(root: Path, sid: str, level: int) -> None:
    arr = np.full((96, 96, 3), level, dtype=np.uint8)
    (root / f"{sid}.png").write_bytes(cv2.imencode(".png", arr)[1].tobytes())


def _result(score: float | None, *, flagged: bool = False, face: bool = True, unavailable: str | None = None) -> AlgorithmResult:
    return AlgorithmResult(
        method=MethodName.DEEPFAKE_CUES,
        verdict=Verdict.INCONCLUSIVE,
        score=score,
        score_definition="test-only cue",
        evidence={"cue_flag": flagged, "face_detected": face,
                  "cue_unavailable": unavailable is not None, "unavailable_reason": unavailable},
    )


def test_tied_auc_is_rank_based_and_empty_class_is_undefined():
    assert study.score_roc_auc([0.1, 0.9, 0.5, 0.5], [False, True, False, True]) == 0.875
    assert study.score_roc_auc([0.5, 0.5], [True, False]) == 0.5
    assert study.score_roc_auc([0.1, 0.2], [True, True]) is None
    assert study.score_roc_auc([], []) is None
    with pytest.raises(ValueError, match="lengths"):
        study.score_roc_auc([0.1], [])
    with pytest.raises(ValueError, match="finite"):
        study.score_roc_auc([float("nan"), 0.2], [True, False])


def test_group_split_leakage_rejected_before_analysis(tmp_path):
    manifest = {"version": 1, "samples": [_sample("a", split="dev", group="video_1"), _sample("b", group="video_1")]}
    with pytest.raises(ValueError, match="both dev and test"):
        study.run_deepfake_study(manifest, tmp_path)


def test_duplicate_bytes_rejected_across_splits_even_with_different_groups(tmp_path):
    _image(tmp_path, "dev", 80)
    (tmp_path / "test.png").write_bytes((tmp_path / "dev.png").read_bytes())
    manifest = {"version": 1, "samples": [_sample("dev", split="dev"), _sample("test", "fake")]}
    with pytest.raises(ValueError, match="duplicate file content"):
        study.run_deepfake_study(manifest, tmp_path)


def test_non_regular_input_is_not_opened_by_hash_audit(tmp_path, monkeypatch):
    # Directories are portable non-files; the same guard also prevents opening
    # a named pipe/device, where a byte limit alone cannot prevent a blocked read.
    nonfile = tmp_path / "a.png"
    nonfile.mkdir()
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == nonfile:
            pytest.fail("non-file input must not be opened for hashing")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = study.run_deepfake_study({"version": 1, "samples": [_sample("a")]}, tmp_path)
    assert report["integrity"]["unhashed_ids"] == ["a"]
    assert report["summary"]["exclusions"] == {"input:PATH_INVALID": 1}


@pytest.mark.parametrize("relative", ["../escape.png", "/absolute.png", "a/../b.png", "C:/a.png", "a\\b.png", "./a.png"])
def test_paths_cannot_escape_or_ambiguously_resolve(tmp_path, relative):
    sample = {**_sample("a"), "path": relative}
    with pytest.raises(ValueError, match="relative"):
        study.validate_manifest({"version": 1, "samples": [sample]}, tmp_path)


def test_coverage_by_class_and_exclusion_denominators_are_explicit(tmp_path, monkeypatch):
    samples = [_sample("real_cue"), _sample("real_no_cue"), _sample("fake_cue", "fake"),
               _sample("fake_no_face", "fake"), _sample("missing", "fake")]
    for index, sample in enumerate(samples[:-1]):
        _image(tmp_path, sample["id"], 30 + index * 20)
    results = {
        "real_cue.png": _result(0.6, flagged=True),
        "real_no_cue.png": _result(0.2, flagged=False),
        "fake_cue.png": _result(0.9, flagged=True),
        "fake_no_face.png": _result(None, face=False, unavailable="no_face"),
    }
    monkeypatch.setattr(study.DeepfakeCues, "analyze", lambda _self, doc, _cfg: results[doc.file_name])
    report = study.run_deepfake_study({"version": 1, "samples": samples}, tmp_path)
    summary = report["summary"]
    assert summary["n_total"] == 5
    assert summary["n_eligible"] == 3
    assert summary["coverage"] == 3 / 5
    assert summary["by_class"]["real"]["n_cued"] == 1
    assert summary["by_class"]["real"]["cue_rate_eligible"] == 0.5
    assert summary["by_class"]["fake"]["coverage"] == 1 / 3
    assert summary["by_class"]["fake"]["cue_rate_eligible"] == 1.0
    assert summary["by_class"]["fake"]["cue_yield_all"] == 1 / 3
    assert summary["score_roc_auc_eligible"] == 1.0
    assert summary["exclusions"] == {"input:PATH_INVALID": 1, "no_face": 1}
    assert report["integrity"]["unhashed_ids"] == ["missing"]
    assert len(report["config_sha256"]) == 64
    json.dumps(report, allow_nan=False)


def test_inconclusive_never_becomes_a_positive_without_explicit_cue_flag(tmp_path, monkeypatch):
    _image(tmp_path, "a", 50)
    result = _result(0.9)
    result.evidence.pop("cue_flag")
    monkeypatch.setattr(study.DeepfakeCues, "analyze", lambda *_args: result)
    report = study.run_deepfake_study({"version": 1, "samples": [_sample("a")]}, tmp_path)
    assert report["summary"]["n_eligible"] == 0
    assert report["summary"]["exclusions"] == {"missing_cue_flag": 1}
    assert report["summary"]["by_class"]["real"]["cue_rate_eligible"] is None


def test_analyzer_failure_is_excluded_and_kept_in_denominator(tmp_path, monkeypatch):
    _image(tmp_path, "a", 50)

    def fail(*_args):
        raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(study.DeepfakeCues, "analyze", fail)
    report = study.run_deepfake_study({"version": 1, "samples": [_sample("a")]}, tmp_path)
    assert report["summary"]["exclusions"] == {"analyzer:RuntimeError": 1}
    assert report["summary"]["n_total"] == 1
    assert report["summary"]["score_roc_auc_eligible"] is None


def test_dual_camera_capture_is_excluded_from_single_image_protocol(tmp_path, monkeypatch):
    from PIL import Image, PngImagePlugin

    from imageguard.core.io import CAPTURE_LAYOUT_KEY

    marker = PngImagePlugin.PngInfo()
    marker.add_text(CAPTURE_LAYOUT_KEY, json.dumps({"version": 1, "layout": "dual_side_by_side",
                                                  "split_x": 96, "width": 192, "height": 96}))
    Image.new("RGB", (192, 96), (80, 80, 80)).save(tmp_path / "a.png", pnginfo=marker)

    def must_not_run(*_args):
        pytest.fail("dual capture should not be analysed as one labelled face")

    monkeypatch.setattr(study.DeepfakeCues, "analyze", must_not_run)
    report = study.run_deepfake_study({"version": 1, "samples": [_sample("a")]}, tmp_path)
    assert report["summary"]["exclusions"] == {"unsupported_capture_layout": 1}


def test_changed_file_during_read_is_rejected(tmp_path, monkeypatch):
    _image(tmp_path, "a", 50)
    original_open = study.safe_open

    def changed(path, cfg):
        doc = original_open(path, cfg)
        doc.sha256 = "changed"
        return doc

    monkeypatch.setattr(study, "safe_open", changed)
    with pytest.raises(ValueError, match="changed during evaluation"):
        study.run_deepfake_study({"version": 1, "samples": [_sample("a")]}, tmp_path)


def test_cli_does_not_overwrite_an_existing_report(tmp_path, monkeypatch):
    from scripts.evaluate_deepfake_cues import main

    _image(tmp_path, "a", 50)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "samples": [_sample("a")]}), encoding="utf-8")
    output = tmp_path / "report.json"
    output.write_text("keep existing evidence", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["evaluate_deepfake_cues", "--manifest", str(manifest),
                                    "--root", str(tmp_path), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert output.read_text(encoding="utf-8") == "keep existing evidence"
