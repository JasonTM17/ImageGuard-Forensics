"""Supervised model correctness, grouped isolation and strict JSON boundaries."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from imageguard.evaluation.deepfake_training import fit_model
from imageguard.forensics.auxiliary import deepfake_model as runtime

CONFIG = "a" * 64
MANIFEST = "b" * 64


def _features(value: float) -> dict:
    features = dict.fromkeys(runtime.FEATURE_NAMES, 0.25)
    features["smooth_cue"] = value
    features["fft_hf_peak_ratio"] = 2.0
    return features


def _rows(groups: int = 6) -> list[dict]:
    rows = []
    for group in range(groups):
        for label in ("real", "fake"):
            sid = f"group{group:02}_{label}"
            rows.append({
                "id": sid, "source_group": f"group{group:02}", "split": "dev", "label": label,
                "sha256": hashlib.sha256(sid.encode()).hexdigest(),
                "features": _features(0.15 + group * 0.005 if label == "real" else 0.85 - group * 0.005),
            })
    return rows


def _fit(rows: list[dict] | None = None, **kwargs):
    return fit_model(_rows() if rows is None else rows, config_sha256=CONFIG, manifest_sha256=MANIFEST, **kwargs)


def _reseal(model: dict) -> dict:
    model["model_sha256"] = runtime.config_digest({key: value for key, value in model.items() if key != "model_sha256"})
    return model


def test_supervised_fit_learns_direction_and_is_reproducible_for_reordered_rows():
    rows = _rows()
    model, report = _fit(rows, folds=3, seed=42, l2=0.1)
    reversed_model, reversed_report = _fit(list(reversed(rows)), folds=3, seed=42, l2=0.1)
    assert model == reversed_model
    assert report == reversed_report
    assert runtime.decision_function(model, _features(0.1)) < -1
    assert runtime.decision_function(model, _features(0.9)) > 1
    assert report["out_of_fold_metrics"]["roc_auc"] == 1
    assert report["out_of_fold_metrics"]["balanced_accuracy_at_zero"] == 1
    assert report["threshold_selection"] == "none"
    assert model["training"]["converged"] is True


def test_grouped_oof_matches_independent_refit_without_validation_samples():
    rows = _rows()
    # One source has very different spectral magnitudes; leaking its standard
    # deviation into training changes the regularized classifier.
    for row in rows:
        row["features"]["fft_hf_peak_ratio"] = 1000.0 if row["source_group"] == "group00" else 1.0
    _, report = _fit(rows, folds=3)
    oof = {row["id"]: row for row in report["out_of_fold"]}
    seen = set()
    for fold in report["folds"]:
        assert set(fold["train_groups"]).isdisjoint(fold["validation_groups"])
        train_rows = [row for row in rows if row["source_group"] in fold["train_groups"]]
        independent, _ = _fit(train_rows, folds=2)
        for row in rows:
            if row["source_group"] in fold["validation_groups"]:
                seen.add(row["id"])
                assert oof[row["id"]]["fold"] == fold["fold"]
                assert runtime.decision_function(independent, row["features"]) == pytest.approx(oof[row["id"]]["decision_score"], abs=1e-12)
    assert seen == {row["id"] for row in rows}


def test_unbalanced_pure_and_mixed_groups_keep_both_classes_in_each_fold():
    rows = _rows(5)
    # Preserve one mixed group, split all other class rows into pure groups.
    for row in rows[2:]:
        row["source_group"] = row["id"]
    _, report = _fit(rows, folds=5)
    assert report["actual_folds"] == 5
    for fold in report["folds"]:
        assert fold["metrics"]["n_real"] >= 1
        assert fold["metrics"]["n_fake"] >= 1


def test_constant_features_are_finite_and_not_perfect_classification():
    rows = _rows(3)
    for row in rows:
        row["features"] = _features(0.5)
    model, report = _fit(rows)
    assert model["normalization"]["scale"] == [1.0] * len(runtime.FEATURE_NAMES)
    assert np.all(np.isfinite(model["weights"]))
    assert report["out_of_fold_metrics"]["roc_auc"] == 0.5
    assert report["out_of_fold_metrics"]["balanced_accuracy_at_zero"] == 0.5
    assert report["actual_folds"] == 3  # explicit reduction to available groups


def test_symmetric_one_feature_fit_matches_scalar_optimum():
    rows = _rows(3)
    for row in rows:
        row["features"] = _features(0.2 if row["label"] == "real" else 0.8)
    model, _ = _fit(rows, l2=1.0)
    # Standardized x is -1/+1. Symmetry gives intercept zero; the optimum
    # satisfies lambda*w = 1/(1+exp(w)), independently of the Newton solver.
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid < 1 / (1 + np.exp(mid)):
            lo = mid
        else:
            hi = mid
    assert model["weights"][0] == pytest.approx((lo + hi) / 2, abs=1e-7)
    assert model["intercept"] == pytest.approx(0.0, abs=1e-10)
    assert max(abs(value) for value in model["weights"][1:]) < 1e-10


def test_returned_fit_is_stationary_for_independently_evaluated_objective():
    rows = _rows(5)
    rng = np.random.default_rng(271)
    for row in rows:
        for name in runtime.FEATURE_NAMES[1:]:
            row["features"][name] = float(rng.uniform(0.1, 0.9))
    regularization = 0.17
    model, _ = _fit(rows, l2=regularization)
    x = np.asarray([[row["features"][name] for name in runtime.FEATURE_NAMES] for row in rows])
    z = (x - np.asarray(model["normalization"]["mean"])) / np.asarray(model["normalization"]["scale"])
    y = np.asarray([int(row["label"] == "fake") for row in rows])
    sample_weights = np.asarray([0.5 / sum(y == label) for label in y])
    beta = np.asarray([*model["weights"], model["intercept"]])

    def loss(candidate):
        margin = z @ candidate[:-1] + candidate[-1]
        return sample_weights @ np.logaddexp(0, (1 - 2 * y) * margin) + regularization / 2 * sum(candidate[:-1] ** 2)

    step = 1e-5
    derivatives = []
    for index in range(len(beta)):
        perturbation = np.zeros_like(beta)
        perturbation[index] = step
        derivatives.append((loss(beta + perturbation) - loss(beta - perturbation)) / (2 * step))
    assert max(abs(value) for value in derivatives) < 2e-7


def test_class_balanced_constant_fit_does_not_learn_sample_prevalence():
    rows = _rows(3)
    for original in list(rows):
        if original["label"] == "real":
            for duplicate in range(3):
                sid = f"{original['id']}_different_frame_{duplicate}"
                rows.append({**original, "id": sid, "sha256": hashlib.sha256(sid.encode()).hexdigest()})
    for row in rows:
        row["features"] = _features(0.5)
    model, _ = _fit(rows)
    assert model["intercept"] == 0.0
    assert model["weights"] == [0.0] * len(runtime.FEATURE_NAMES)


def test_very_unequal_groups_stay_intact_and_keep_two_classes_in_validation():
    rows = []
    for group, n_real, n_fake in [("large", 50, 2), ("mixed", 2, 1),
                                  ("real_a", 4, 0), ("real_b", 1, 0),
                                  ("real_c", 3, 0), ("fake_a", 0, 1), ("fake_b", 0, 2)]:
        for label, count in (("real", n_real), ("fake", n_fake)):
            for index in range(count):
                sid = f"{group}_{label}_{index}"
                rows.append({"id": sid, "source_group": group, "split": "dev", "label": label,
                             "sha256": hashlib.sha256(sid.encode()).hexdigest(),
                             "features": _features(0.2 if label == "real" else 0.8)})
    _, report = _fit(rows, folds=5)
    assert report["actual_folds"] == 4
    assert len(report["out_of_fold"]) == len(rows)
    memberships = {}
    for row in report["out_of_fold"]:
        memberships.setdefault(row["source_group"], set()).add(row["fold"])
    assert all(len(assigned) == 1 for assigned in memberships.values())
    for fold in report["folds"]:
        assert fold["metrics"]["n_real"] > 0 and fold["metrics"]["n_fake"] > 0


def test_fit_rejects_model_whose_serialized_ledger_exceeds_loader_limit(monkeypatch):
    from imageguard.evaluation import deepfake_training

    monkeypatch.setattr(deepfake_training, "MAX_MODEL_BYTES", 100)
    with pytest.raises(ValueError, match="provenance exceeds the JSON size limit"):
        _fit()


@pytest.mark.parametrize("excluded", [False, True])
def test_test_samples_are_rejected_even_in_exclusion_ledger(excluded):
    row = {**_rows()[0], "split": "test", "exclusion_reason": "no_face"}
    kwargs = {"excluded_dev": [row]} if excluded else {}
    with pytest.raises(ValueError, match="dev samples only"):
        _fit(None if excluded else [row, *_rows()[1:]], **kwargs)


def test_excluded_source_hash_and_id_are_kept_but_do_not_influence_fit():
    baseline, _ = _fit()
    excluded = {
        "id": "unusable", "source_group": "excluded_group", "split": "dev", "label": "real",
        "sha256": "c" * 64, "exclusion_reason": "no_face",
    }
    model, report = _fit(excluded_dev=[excluded])
    assert model["weights"] == baseline["weights"]
    assert model["normalization"] == baseline["normalization"]
    assert "excluded_group" in model["training"]["source_groups"]
    assert "c" * 64 in model["training"]["image_hashes"]
    assert "unusable" in model["training"]["sample_ids"]
    assert report["n_excluded"] == 1


def test_duplicate_hashes_and_insufficient_class_groups_are_rejected():
    rows = _rows()
    rows[1]["sha256"] = rows[0]["sha256"]
    with pytest.raises(ValueError, match="duplicate"):
        _fit(rows)
    rows = _rows()
    for row in rows:
        if row["label"] == "fake":
            row["source_group"] = "all_fakes_one_source"
    with pytest.raises(ValueError, match="two independent source groups"):
        _fit(rows)


@pytest.mark.parametrize("kwargs", [{"l2": 0}, {"l2": float("nan")}, {"l2": True},
                                    {"folds": True}, {"folds": 1}, {"seed": -1}, {"excluded_dev": {}}])
def test_invalid_fit_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _fit(**kwargs)


def test_feature_extraction_requires_exact_usable_variant_and_feature_ranges():
    evidence = {**_features(0.5), "variant": runtime.EXTRACTOR_VARIANT,
                "face_detected": True, "cue_unavailable": False}
    assert runtime.features_from_evidence(evidence) == _features(0.5)
    for update in ({"variant": "older"}, {"face_detected": False}, {"cue_unavailable": True},
                   {"smooth_cue": 1.1}, {"freq_cue": float("nan")}, {"lbp_cue": True}):
        with pytest.raises(ValueError):
            runtime.features_from_evidence({**evidence, **update})
    evidence.pop("hog_asym")
    with pytest.raises(ValueError):
        runtime.features_from_evidence(evidence)


@pytest.mark.parametrize("mutation", [
    lambda model: model.update(extractor_variant="wrong"),
    lambda model: model.update(feature_names=list(reversed(runtime.FEATURE_NAMES))),
    lambda model: model.update(weights=[0.0]),
    lambda model: model["weights"].__setitem__(0, True),
    lambda model: model["normalization"]["scale"].__setitem__(0, 0.0),
    lambda model: model["training"].update(n_fitted=999),
    lambda model: model["training"].update(source_groups=[]),
    lambda model: model["training"]["samples"][0].update(split="test"),
    lambda model: model.update(label_mapping={"real": False, "fake": True}),
    lambda model: model.update(arbitrary_executable="os.system"),
])
def test_strict_model_schema_rejects_malformed_even_with_matching_hash(mutation):
    model, _ = _fit()
    mutation(model)
    _reseal(model)
    with pytest.raises(ValueError):
        runtime.validate_model(model)


def test_checksum_and_config_bind_the_model_to_exact_settings():
    model, _ = _fit()
    runtime.validate_model(model, CONFIG)
    with pytest.raises(ValueError, match="configuration"):
        runtime.validate_model(model, "d" * 64)
    model["weights"][0] += 0.001
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runtime.validate_model(model)


def test_json_roundtrip_preserves_decisions_and_refuses_overwrite(tmp_path):
    model, _ = _fit()
    path = tmp_path / "model.json"
    runtime.save_model(path, model)
    loaded = runtime.load_model(path, CONFIG)
    assert loaded == model
    assert runtime.decision_function(loaded, _features(0.7)) == runtime.decision_function(model, _features(0.7))
    old_bytes = path.read_bytes()
    with pytest.raises(FileExistsError):
        runtime.save_model(path, model)
    assert path.read_bytes() == old_bytes


@pytest.mark.parametrize("content", ['{"format":"one","format":"two"}', '{"value":NaN}', '{"value":Infinity}', '[]'])
def test_loader_rejects_ambiguous_or_nonfinite_json(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        runtime.load_model(path)


def test_loader_bounds_file_size_and_refuses_nonfiles(tmp_path, monkeypatch):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * 100)
    monkeypatch.setattr(runtime, "MAX_MODEL_BYTES", 20)
    with pytest.raises(ValueError, match="size limit"):
        runtime.load_model(path)
    with pytest.raises(ValueError, match="regular file"):
        runtime.load_model(tmp_path)


def test_excessive_json_nesting_is_a_validation_error_for_gui_callers(tmp_path):
    path = tmp_path / "nested.json"
    path.write_text("[" * 2000 + "]" * 2000, encoding="utf-8")
    # Python versions differ in whether the JSON parser or schema rejects the
    # nested array first; GUI callers must receive the same validation family.
    with pytest.raises(ValueError):
        runtime.load_model(path)


def test_model_and_features_are_not_mutated_by_prediction():
    model, _ = _fit()
    features = _features(0.8)
    before = copy.deepcopy(model)
    feature_before = features.copy()
    runtime.decision_function(model, features)
    assert model == before
    assert features == feature_before
    json.dumps(model, allow_nan=False)
