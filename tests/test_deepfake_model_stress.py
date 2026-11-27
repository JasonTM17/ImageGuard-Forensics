"""Regression cases found by cloud runtime stress probes, not accuracy tests."""

from __future__ import annotations

import json

import pytest

from imageguard.core.config import load_thresholds
from imageguard.forensics.auxiliary import deepfake_model as runtime
from tests.test_deepfake_model import _features, _fit
from tests.test_deepfake_training_cli import _cli


@pytest.mark.parametrize("field", ["weight", "mean", "intercept", "l2"])
def test_json_integer_too_large_for_float_is_a_validation_error(tmp_path, field):
    model, _ = _fit()
    enormous = 10**1000  # valid JSON below both parser-digit and file-size limits
    if field == "weight":
        model["weights"][0] = enormous
    elif field == "mean":
        model["normalization"]["mean"][0] = enormous
    elif field == "intercept":
        model["intercept"] = enormous
    else:
        model["training"]["l2"] = enormous
    path = tmp_path / "mô_hình_số_lớn.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        runtime.load_model(path)


def test_enormous_integer_feature_is_rejected_without_overflow():
    model, _ = _fit()
    features = _features(0.5)
    features["fft_hf_peak_ratio"] = 10**1000
    with pytest.raises(ValueError, match="finite"):
        runtime.decision_function(model, features)


def test_analyze_cli_reports_invalid_large_integer_model_without_traceback(tmp_path):
    model, _ = _fit()
    model["config_sha256"] = runtime.config_digest(load_thresholds().raw)
    model.pop("model_sha256")
    model["model_sha256"] = runtime.config_digest(model)
    model["intercept"] = 10**1000
    path = tmp_path / "mô_hình.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    result = _cli("-m", "imageguard", "analyze", tmp_path / "ảnh_chưa_có.png",
                  "--methods", "deepfake", "--deepfake-model", path, "--no-persist")
    assert result.returncode == 2
    assert "cannot use deepfake model" in result.stderr
    assert "finite" in result.stderr
    assert "Traceback" not in result.stderr
