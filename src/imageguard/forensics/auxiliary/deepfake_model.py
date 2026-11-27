"""Strict JSON runtime for a small supervised model of DIP face features.

The signed logistic decision value is a learned ranking score, not a calibrated
probability of a Deepfake. No executable serialization or remote weights are used.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from numbers import Real
from pathlib import Path

import numpy as np

FEATURE_NAMES = (
    "smooth_cue", "seam_cue", "lbp_cue", "lbp_chi2", "lbp_flat_in", "lbp_flat_out",
    "freq_cue", "fft_hf_peak_ratio", "spectral_rolloff",
    "seam_hough_cue", "symmetry_cue", "hog_asym", "roi_flag_fraction",
)
FEATURE_SCHEMA = "imageguard-dip-features-v1"
EXTRACTOR_VARIANT = "dip_6pillar_deepfake_cues_v3"
MODEL_FORMAT = "imageguard-dip-logistic-v1"
SCORE_DEFINITION = "signed logistic decision score; positive is the fake training class; not a calibrated probability"
MAX_MODEL_BYTES = 8 * 1024 * 1024
MAX_TRAINING_SAMPLES = 20000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UNBOUNDED_FEATURES = {"fft_hf_peak_ratio", "spectral_rolloff"}
_MODEL_KEYS = {
    "format", "feature_schema", "feature_names", "extractor_variant", "config_sha256",
    "label_mapping", "score_definition", "normalization", "weights", "intercept",
    "training", "model_sha256",
}
_TRAINING_KEYS = {
    "algorithm", "objective", "l2", "seed", "folds", "manifest_sha256",
    "feature_records_sha256", "samples", "source_groups", "image_hashes", "sample_ids",
    "n_fitted", "n_excluded", "iterations", "converged",
}


def config_digest(value: dict) -> str:
    """Canonical SHA-256 for configuration, model and provenance JSON objects."""
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _number(value: object, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")  # noqa: TRY004 - one validation error contract
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        # JSON integers can exceed float64 while fitting within the model file
        # size limit. Keep malformed numeric input in the validation contract
        # used by the GUI and CLI instead of leaking an uncaught OverflowError.
        raise ValueError(f"{label} must be a finite number in [{low}, {high}]") from error
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{label} must be finite and in [{low}, {high}]")
    return number


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or len(value) > 512:
        raise ValueError(f"{label} must be a nonempty trimmed string of at most 512 characters")
    return value


def _feature_vector(features: dict) -> np.ndarray:
    if not isinstance(features, dict) or set(features) != set(FEATURE_NAMES):
        raise ValueError("feature names must match the complete ordered feature schema")
    # The raw spectral ratios are nonnegative and not bounded by one. A wide
    # numerical safety bound prevents malicious/extreme values overflowing fit.
    return np.array([
        _number(features[name], name, 0.0, 1e15 if name in _UNBOUNDED_FEATURES else 1.0)
        for name in FEATURE_NAMES
    ], dtype=np.float64)


def features_from_evidence(evidence: dict) -> dict[str, float]:
    """Extract features only from a usable face produced by the exact extractor."""
    if not isinstance(evidence, dict) or evidence.get("variant") != EXTRACTOR_VARIANT:
        raise ValueError("feature extractor variant mismatch")
    if evidence.get("face_detected") is not True or evidence.get("cue_unavailable") is not False:
        raise ValueError("face features are unavailable")
    values = _feature_vector({name: evidence.get(name) for name in FEATURE_NAMES})
    return dict(zip(FEATURE_NAMES, map(float, values), strict=True))


def validate_training_sample(sample: dict, *, included: bool) -> dict:
    """Normalize fit/exclusion provenance; every supplied sample must be dev."""
    if not isinstance(sample, dict):
        raise ValueError("training sample must be an object")  # noqa: TRY004 - one validation error contract
    if sample.get("split") != "dev":
        raise ValueError("training accepts dev samples only; test data must remain held out")
    label = sample.get("label")
    if label not in ("real", "fake"):
        raise ValueError("training labels must be real or fake")
    sha = sample.get("sha256")
    if included or sha is not None:
        sha = _hash(sha, "sample sha256")
    reason = None if included else _text(sample.get("exclusion_reason"), "exclusion_reason")
    return {
        "id": _text(sample.get("id"), "sample id"),
        "source_group": _text(sample.get("source_group"), "source_group"),
        "split": "dev", "label": label, "sha256": sha,
        "included": included, "exclusion_reason": reason,
    }


def _validate_training(training: dict) -> None:
    if not isinstance(training, dict) or set(training) != _TRAINING_KEYS:
        raise ValueError("training provenance keys do not match the model schema")
    if training["algorithm"] != "numpy-newton-logistic-v1":
        raise ValueError("unsupported training algorithm")
    if training["objective"] != "class-balanced mean logistic loss + l2/2 * squared weight norm":
        raise ValueError("unsupported training objective")
    _number(training["l2"], "l2", 1e-8, 1e6)
    _hash(training["manifest_sha256"], "manifest_sha256")
    _hash(training["feature_records_sha256"], "feature_records_sha256")
    for key, lower, upper in (("seed", 0, 2**32 - 1), ("folds", 2, 20),
                              ("iterations", 0, 100), ("n_fitted", 4, MAX_TRAINING_SAMPLES),
                              ("n_excluded", 0, MAX_TRAINING_SAMPLES)):
        if type(training[key]) is not int or not lower <= training[key] <= upper:
            raise ValueError(f"invalid training {key}")
    if training["converged"] is not True:
        raise ValueError("model optimizer did not converge")
    samples = training["samples"]
    if not isinstance(samples, list) or not 4 <= len(samples) <= MAX_TRAINING_SAMPLES:
        raise ValueError("invalid training sample ledger")
    ids: set[str] = set()
    hashes: set[str] = set()
    groups: set[str] = set()
    class_groups = {"real": set(), "fake": set()}
    n_fitted = 0
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {
            "id", "source_group", "split", "label", "sha256", "included", "exclusion_reason",
        } or type(sample.get("included")) is not bool:
            raise ValueError("invalid training sample ledger entry")
        normalized = validate_training_sample(sample, included=sample["included"])
        if sample != normalized:
            raise ValueError("inconsistent training sample provenance")
        if sample["id"] in ids or (sample["sha256"] is not None and sample["sha256"] in hashes):
            raise ValueError("duplicate id or image hash in training provenance")
        ids.add(sample["id"])
        groups.add(sample["source_group"])
        if sample["sha256"] is not None:
            hashes.add(sample["sha256"])
        if sample["included"]:
            n_fitted += 1
            class_groups[sample["label"]].add(sample["source_group"])
    if (training["n_fitted"] != n_fitted or training["n_excluded"] != len(samples) - n_fitted
            or any(len(value) < training["folds"] for value in class_groups.values())):
        raise ValueError("inconsistent training counts or grouped fold provenance")
    for key, values in (("source_groups", groups), ("image_hashes", hashes), ("sample_ids", ids)):
        if training[key] != sorted(values):
            raise ValueError(f"inconsistent training {key}")


def validate_model(model: dict, expected_config_sha256: str | None = None) -> dict:
    """Reject unsupported schema, malformed numbers, wrong config and edits.

    The digest detects accidental alteration; it is not a publisher signature.
    """
    if not isinstance(model, dict) or set(model) != _MODEL_KEYS:
        raise ValueError("model keys do not match the supported JSON schema")
    if (model["format"] != MODEL_FORMAT or model["feature_schema"] != FEATURE_SCHEMA
            or model["feature_names"] != list(FEATURE_NAMES)
            or model["extractor_variant"] != EXTRACTOR_VARIANT):
        raise ValueError("model format, feature schema or extractor variant mismatch")
    if (model["label_mapping"] != {"real": 0, "fake": 1}
            or any(type(value) is not int for value in model["label_mapping"].values())):
        raise ValueError("unsupported label mapping")
    if model["score_definition"] != SCORE_DEFINITION:
        raise ValueError("unsupported model score definition")
    config = _hash(model["config_sha256"], "config_sha256")
    if expected_config_sha256 is not None and config != _hash(expected_config_sha256, "expected_config_sha256"):
        raise ValueError("model configuration does not match the current extractor configuration")
    normalization = model["normalization"]
    if not isinstance(normalization, dict) or set(normalization) != {"mean", "scale"}:
        raise ValueError("invalid model normalization")
    for key, values, low, high in (
        ("mean", normalization["mean"], 0.0, 1e15),
        ("scale", normalization["scale"], 1e-12, 1e15),
        ("weights", model["weights"], -1e12, 1e12),
    ):
        if not isinstance(values, list) or len(values) != len(FEATURE_NAMES):
            raise ValueError(f"{key} must have one number per feature")
        for value in values:
            _number(value, key, low, high)
    _number(model["intercept"], "intercept", -1e12, 1e12)
    _validate_training(model["training"])
    expected_hash = config_digest({key: value for key, value in model.items() if key != "model_sha256"})
    if _hash(model["model_sha256"], "model_sha256") != expected_hash:
        raise ValueError("model SHA-256 mismatch")
    return model


def decision_function(model: dict, features: dict) -> float:
    """Return w·((x-mean)/scale)+intercept; never an authenticity probability."""
    validate_model(model)
    vector = _feature_vector(features)
    normalized = (vector - np.asarray(model["normalization"]["mean"])) / np.asarray(model["normalization"]["scale"])
    score = float(normalized @ np.asarray(model["weights"]) + model["intercept"])
    if not math.isfinite(score):
        raise ValueError("model decision score is nonfinite")
    return score


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def load_model(path: str | Path, expected_config_sha256: str | None = None) -> dict:
    """Read a bounded, inert JSON model, refusing duplicate keys and NaN/Inf."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("model path is not a regular file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_MODEL_BYTES + 1)
    if len(raw) > MAX_MODEL_BYTES:
        raise ValueError("model exceeds the JSON size limit")
    try:
        model = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except RecursionError as error:
        raise ValueError("model JSON nesting exceeds the parser limit") from error
    return validate_model(model, expected_config_sha256)


def save_model(path: str | Path, model: dict) -> None:
    """Save a validated JSON artifact without overwriting existing evidence."""
    validate_model(model)
    raw = json.dumps(model, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    if len(raw) > MAX_MODEL_BYTES:
        raise ValueError("model exceeds the JSON size limit")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
