"""Deterministic grouped development evaluation and NumPy logistic fitting.

Only independently labelled ``dev`` rows enter this module. Standardization is
fit afresh inside each training fold; no test data or test threshold is used.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from imageguard.forensics.auxiliary.deepfake_model import (
    EXTRACTOR_VARIANT,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    MAX_MODEL_BYTES,
    MAX_TRAINING_SAMPLES,
    MODEL_FORMAT,
    SCORE_DEFINITION,
    _feature_vector,
    _hash,
    _number,
    config_digest,
    validate_model,
    validate_training_sample,
)


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float) -> dict:
    """Minimize a strictly regularized, class-balanced convex objective."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    design = np.column_stack((z, np.ones(len(z))))
    counts = np.bincount(y, minlength=2)
    if np.any(counts == 0):
        raise ValueError("each training fold must contain both real and fake labels")
    sample_weight = 1.0 / (2.0 * counts[y])
    regularization = np.diag([l2] * x.shape[1] + [0.0])
    beta = np.zeros(design.shape[1], dtype=np.float64)

    def objective(value: np.ndarray) -> float:
        logits = design @ value
        return float(sample_weight @ (np.logaddexp(0.0, logits) - y * logits)
                     + 0.5 * value @ regularization @ value)

    for iteration in range(101):
        logits = design @ beta
        probabilities = np.exp(-np.logaddexp(0.0, -logits))
        gradient = design.T @ (sample_weight * (probabilities - y)) + regularization @ beta
        if np.max(np.abs(gradient)) <= 1e-8:
            return {"mean": mean, "scale": scale, "weights": beta[:-1],
                    "intercept": float(beta[-1]), "iterations": iteration}
        if iteration == 100:
            break
        curvature = sample_weight * probabilities * (1.0 - probabilities)
        hessian = (design.T * curvature) @ design + regularization
        step = np.linalg.solve(hessian, gradient)
        current = objective(beta)
        step_size = 1.0
        descent = float(gradient @ step)
        for _ in range(30):
            candidate = beta - step_size * step
            if objective(candidate) <= current - 1e-4 * step_size * descent:
                beta = candidate
                break
            step_size *= 0.5
        else:
            raise ValueError("logistic optimizer line search did not converge")
    raise ValueError("logistic optimizer did not converge within 100 iterations")


def _grouped_folds(rows: list[dict], labels: np.ndarray, folds: int, seed: int) -> list[list[int]]:
    """Create deterministic group-disjoint folds containing both classes.

    Seed ties, distribute mixed groups first, then fill missing classes using
    single-class groups. This guarantees class coverage whenever each class
    occurs in at least ``folds`` groups, including unbalanced group sizes.
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["source_group"]].append(index)
    counts = {group: np.bincount(labels[indices], minlength=2) for group, indices in grouped.items()}
    rng = np.random.default_rng(seed)
    names = sorted(grouped)
    rng.shuffle(names)
    names.sort(key=lambda group: -len(grouped[group]))
    mixed = [group for group in names if np.all(counts[group] > 0)]
    pure = {label: [group for group in names if counts[group][1 - label] == 0] for label in (0, 1)}
    assignments: list[list[int]] = [[] for _ in range(folds)]
    fold_counts = np.zeros((folds, 2), dtype=np.int64)

    def assign(group: str, fold: int) -> None:
        assignments[fold].extend(grouped[group])
        fold_counts[fold] += counts[group]

    used: set[str] = set()
    for fold, group in enumerate(mixed[:folds]):
        assign(group, fold)
        used.add(group)
    for label in (0, 1):
        candidates = iter(pure[label])
        for fold in range(folds):
            if fold_counts[fold, label] == 0:
                group = next(candidates)
                assign(group, fold)
                used.add(group)
    totals = np.bincount(labels, minlength=2).astype(float)
    for group in names:
        if group in used:
            continue
        # Keep class fractions near their ideal share, using only dev labels.
        costs = []
        for fold in range(folds):
            candidate = fold_counts.copy()
            candidate[fold] += counts[group]
            costs.append(float(np.sum((candidate / totals - 1.0 / folds) ** 2)))
        fold = min(range(folds), key=lambda index: (costs[index], len(assignments[index]), index))
        assign(group, fold)
    return [sorted(indices) for indices in assignments]


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    # Every validation fold and the complete OOF set have both classes.
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    n_fake = int(labels.sum())
    n_real = len(labels) - n_fake
    return float((ranks[labels == 1].sum() - n_fake * (n_fake + 1) / 2) / (n_fake * n_real))


def _metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    predicted_fake = scores >= 0.0
    fake = labels == 1
    recall = float(np.mean(predicted_fake[fake]))
    fpr = float(np.mean(predicted_fake[~fake]))
    return {
        "n": len(labels), "n_real": int((~fake).sum()), "n_fake": int(fake.sum()),
        "roc_auc": _auc(scores, labels), "fake_recall_at_zero": recall,
        "real_false_positive_rate_at_zero": fpr,
        "balanced_accuracy_at_zero": (recall + 1.0 - fpr) / 2.0,
    }


def fit_model(
    records: list[dict], *, config_sha256: str, manifest_sha256: str,
    l2: float = 1.0, folds: int = 5, seed: int = 20261127,
    excluded_dev: list[dict] | None = None,
) -> tuple[dict, dict]:
    """Fit on dev only, returning a strict JSON model and honest grouped OOF report.

    Every record contains id/source_group/split/label/sha256/features. Optional
    excluded_dev rows retain source/hash provenance but never enter fitting.
    Fixed L2 and decision threshold zero are not tuned on validation or test.
    """
    _hash(config_sha256, "config_sha256")
    _hash(manifest_sha256, "manifest_sha256")
    l2 = _number(l2, "l2", 1e-8, 1e6)
    if type(folds) is not int or not 2 <= folds <= 20:
        raise ValueError("folds must be an integer from 2 to 20")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if not isinstance(records, list) or not records or (excluded_dev is not None and not isinstance(excluded_dev, list)):
        raise ValueError("records and excluded_dev must be lists")
    if len(records) + len(excluded_dev or []) > MAX_TRAINING_SAMPLES:
        raise ValueError("too many training provenance records")
    fitted = []
    for record in records:
        metadata = validate_training_sample(record, included=True)
        vector = _feature_vector(record.get("features"))
        fitted.append({**metadata, "features": dict(zip(FEATURE_NAMES, map(float, vector), strict=True))})
    fitted.sort(key=lambda row: row["id"])
    ledger = [{key: value for key, value in row.items() if key != "features"} for row in fitted]
    ledger.extend(validate_training_sample(row, included=False) for row in (excluded_dev or []))
    ledger.sort(key=lambda row: row["id"])
    sample_ids = [row["id"] for row in ledger]
    hashes = [row["sha256"] for row in ledger if row["sha256"] is not None]
    if len(set(sample_ids)) != len(sample_ids) or len(set(hashes)) != len(hashes):
        raise ValueError("duplicate sample id or image hash in dev provenance")
    x = np.array([_feature_vector(row["features"]) for row in fitted])
    y = np.array([int(row["label"] == "fake") for row in fitted], dtype=np.int64)
    class_groups = [{row["source_group"] for row in fitted if row["label"] == label} for label in ("real", "fake")]
    actual_folds = min(folds, *(len(groups) for groups in class_groups))
    if actual_folds < 2:
        raise ValueError("grouped validation requires at least two independent source groups per class")
    validation_folds = _grouped_folds(fitted, y, actual_folds, seed)
    oof = np.full(len(fitted), np.nan)
    fold_indices = np.full(len(fitted), -1, dtype=int)
    reports = []
    for fold_id, validation in enumerate(validation_folds):
        train = np.ones(len(fitted), dtype=bool)
        train[validation] = False
        fit = _fit_logistic(x[train], y[train], l2)
        scores = ((x[validation] - fit["mean"]) / fit["scale"]) @ fit["weights"] + fit["intercept"]
        oof[validation] = scores
        fold_indices[validation] = fold_id
        reports.append({
            "fold": fold_id, "n_train": int(train.sum()),
            "train_groups": sorted({row["source_group"] for index, row in enumerate(fitted) if train[index]}),
            "validation_groups": sorted({fitted[index]["source_group"] for index in validation}),
            "metrics": _metrics(scores, y[validation]),
        })
    final_fit = _fit_logistic(x, y, l2)
    model = {
        "format": MODEL_FORMAT, "feature_schema": FEATURE_SCHEMA, "feature_names": list(FEATURE_NAMES),
        "extractor_variant": EXTRACTOR_VARIANT, "config_sha256": config_sha256,
        "label_mapping": {"real": 0, "fake": 1}, "score_definition": SCORE_DEFINITION,
        "normalization": {"mean": final_fit["mean"].tolist(), "scale": final_fit["scale"].tolist()},
        "weights": final_fit["weights"].tolist(), "intercept": final_fit["intercept"],
        "training": {
            "algorithm": "numpy-newton-logistic-v1",
            "objective": "class-balanced mean logistic loss + l2/2 * squared weight norm",
            "l2": l2, "seed": seed, "folds": actual_folds, "manifest_sha256": manifest_sha256,
            "feature_records_sha256": config_digest({"records": fitted, "excluded": [row for row in ledger if not row["included"]]}),
            "samples": ledger, "source_groups": sorted({row["source_group"] for row in ledger}),
            "image_hashes": sorted(hashes), "sample_ids": sorted(sample_ids),
            "n_fitted": len(fitted), "n_excluded": len(ledger) - len(fitted),
            "iterations": final_fit["iterations"], "converged": True,
        },
    }
    model["model_sha256"] = config_digest(model)
    validate_model(model, config_sha256)
    # Match save_model's exact pretty-JSON encoding, including its newline, so
    # training cannot return an artifact that the bounded loader must reject.
    if len(json.dumps(model, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")) + 1 > MAX_MODEL_BYTES:
        raise ValueError("trained model provenance exceeds the JSON size limit")
    report = {
        "protocol": "grouped-dev-logistic-v1", "scope": "development cross-validation only; not a held-out test result",
        "score_definition": SCORE_DEFINITION, "config_sha256": config_sha256,
        "model_sha256": model["model_sha256"], "seed": seed,
        "requested_folds": folds, "actual_folds": actual_folds,
        "l2": l2, "hyperparameter_selection": "fixed", "decision_threshold": 0.0,
        "threshold_selection": "none", "n_fitted": len(fitted), "n_excluded": len(ledger) - len(fitted),
        "folds": reports, "out_of_fold_metrics": _metrics(oof, y),
        "out_of_fold": [{"id": row["id"], "source_group": row["source_group"], "label": row["label"],
                         "fold": int(fold_indices[index]), "decision_score": float(oof[index])}
                        for index, row in enumerate(fitted)],
    }
    return model, report
