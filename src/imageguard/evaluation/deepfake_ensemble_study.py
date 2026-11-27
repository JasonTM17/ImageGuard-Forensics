"""Đánh giá ensemble deepfake trên bộ ảnh có nhãn (real/fake).

Chạy từng detector riêng lẻ (cues cổ điển, GenConViT-ED, CoC Xception, DIP đã
học nếu có) VÀ ``deepfake_ensemble`` trên một split cố định, rồi báo cáo cho mỗi
detector: coverage, ROC AUC của điểm, và accuracy/precision/recall/F1 tại
ngưỡng quyết định riêng của detector đó. Không học, không chỉnh ngưỡng, không
coi điểm là xác suất — đúng contract của ``deepfake_study``.
"""

from __future__ import annotations

import math
import platform
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from imageguard.core import limits
from imageguard.core.config import ThresholdSet, load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.errors import ErrorResult
from imageguard.core.io import safe_open
from imageguard.evaluation.deepfake_study import (
    _assert_model_holdout,
    _content_hashes,
    _digest,
    _image_path,
    score_roc_auc,
    validate_manifest,
)
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues
from imageguard.forensics.auxiliary.deepfake_model import (
    config_digest,
    validate_model,
)
from imageguard.forensics.auxiliary.learned_deepfake import LearnedDeepfakeCues
from imageguard.forensics.ml.deepfake_ensemble import DeepfakeEnsemble
from imageguard.forensics.ml.deepfake_genconvit import DeepfakeGenConViT
from imageguard.forensics.ml.deepfake_screen_coc import DeepfakeScreenCoC

# key báo cáo → (factory nhận model DIP hoặc None, section ngưỡng, key ngưỡng)
DETECTOR_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("deepfake_cues", "cues", "deepfake", "score_thresh"),
    ("deepfake_ml", "ml", "deepfake_ml", "fake_thresh"),
    ("deepfake_screen", "screen", "deepfake_screen_coc", "fake_thresh"),
)
LEARNED_KEY = "learned_dip"
ENSEMBLE_KEY = "deepfake_ensemble"
LEARNED_THRESHOLD = 0.5  # LearnedDeepfakeCues: sigmoid(margin) >= 0.5 <=> margin >= 0


def _build_detectors(model: dict | None) -> dict[str, object]:
    detectors: dict[str, object] = {
        "cues": LearnedDeepfakeCues(model) if model is not None else DeepfakeCues(),
        "ml": DeepfakeGenConViT(),
        "screen": DeepfakeScreenCoC(),
    }
    return detectors


def classification_metrics(
    scores: list[float], labels: list[bool], *, threshold: float
) -> dict:
    """Accuracy/precision/recall/F1 tại một ngưỡng cố định; None khi mẫu rỗng."""
    if len(scores) != len(labels):
        raise ValueError("score and label lengths differ")
    tp = sum(1 for s, y in zip(scores, labels, strict=True) if s >= threshold and y)
    fp = sum(1 for s, y in zip(scores, labels, strict=True) if s >= threshold and not y)
    tn = sum(1 for s, y in zip(scores, labels, strict=True) if s < threshold and not y)
    fn = sum(1 for s, y in zip(scores, labels, strict=True) if s < threshold and y)
    n = len(scores)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and precision + recall > 0
        else None
    )
    return {
        "threshold": threshold,
        "n": n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / n if n else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": tn / (tn + fp) if tn + fp else None,
    }


def _summarize_detector(rows: list[dict], key: str, threshold: float) -> dict:
    """Tổng hợp một detector: coverage + AUC + metrics tại ngưỡng của nó."""
    eligible = [row for row in rows if row["detectors"][key]["exclusion_reason"] is None]
    by_class = {}
    for label in ("real", "fake"):
        all_class = [row for row in rows if row["label"] == label]
        scored = [row for row in eligible if row["label"] == label]
        by_class[label] = {
            "n_total": len(all_class),
            "n_eligible": len(scored),
            "n_excluded": len(all_class) - len(scored),
            "coverage": len(scored) / len(all_class) if all_class else None,
        }
    scores = [row["detectors"][key]["score"] for row in eligible]
    labels = [row["label"] == "fake" for row in eligible]
    return {
        "decision_threshold": threshold,
        "n_total": len(rows),
        "n_eligible": len(eligible),
        "n_excluded": len(rows) - len(eligible),
        "coverage": len(eligible) / len(rows) if rows else None,
        "exclusions": dict(sorted(Counter(
            row["detectors"][key]["exclusion_reason"]
            for row in rows
            if row["detectors"][key]["exclusion_reason"]
        ).items())),
        "by_class": by_class,
        "score_roc_auc_eligible": score_roc_auc(scores, labels) if scores else None,
        "classification_at_threshold": classification_metrics(
            scores, labels, threshold=threshold
        ) if scores else None,
    }


def run_deepfake_ensemble_study(
    manifest: dict, root: Path, *, split: str = "test",
    cfg: ThresholdSet | None = None, model: dict | None = None,
) -> dict:
    """Chạy mọi detector deepfake + ensemble trên một split có nhãn, trả báo cáo JSON-able."""
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    samples = validate_manifest(manifest, root)
    selected = [sample for sample in samples if sample["split"] == split]
    if not selected:
        raise ValueError(f"manifest has no {split} samples")
    cfg = cfg or load_thresholds()
    hashes, unhashed = _content_hashes(samples, root, cfg.limit("max_file_bytes", limits.MAX_FILE_BYTES))
    if model is not None:
        if split != "test":
            raise ValueError("a learned model may only be evaluated on test; use grouped OOF for dev")
        model = validate_model(model, expected_config_sha256=config_digest(cfg.raw))
        _assert_model_holdout(model, selected, hashes)

    detectors = _build_detectors(model)
    ensemble = DeepfakeEnsemble(model)
    report_keys = [
        *( ["learned_dip"] if model is not None else ["deepfake_cues"] ),
        "deepfake_ml", "deepfake_screen", ENSEMBLE_KEY,
    ]

    rows: list[dict] = []
    for sample in selected:
        row = {key: sample[key] for key in ("id", "label", "source_group", "split")}
        row.update(condition=sample.get("condition", "original"), sha256=hashes.get(sample["id"]),
                   input_exclusion=None, detectors={})
        doc = safe_open(_image_path(root, sample["path"]), cfg)
        if isinstance(doc, ErrorResult):
            row["input_exclusion"] = f"input:{doc.code.value}"
        elif doc.sha256 != hashes.get(sample["id"]):
            raise ValueError(f"image changed during evaluation: {sample['id']!r}")
        elif doc.capture_layout != "single":
            row["input_exclusion"] = "unsupported_capture_layout"
        else:
            for key, factory_key, _, _ in DETECTOR_SPECS:
                report_key = "learned_dip" if key == "deepfake_cues" and model is not None else key
                entry = {"score": None, "verdict": None, "exclusion_reason": None}
                try:
                    result = detectors[factory_key].analyze(doc, cfg)
                except Exception as error:  # noqa: BLE001 - giữ đủ mẫu trong mẫu số đánh giá
                    entry["exclusion_reason"] = f"analyzer:{type(error).__name__}"
                else:
                    entry["verdict"] = result.verdict.value
                    score = result.score
                    if result.verdict in {Verdict.ERROR, Verdict.NOT_APPLICABLE}:
                        entry["exclusion_reason"] = result.verdict.value
                    elif score is None or not math.isfinite(score) or not 0 <= score <= 1:
                        entry["exclusion_reason"] = "invalid_score"
                    else:
                        entry["score"] = float(score)
                row["detectors"][report_key] = entry
            entry = {"score": None, "verdict": None, "exclusion_reason": None}
            try:
                result = ensemble.analyze(doc, cfg)
            except Exception as error:  # noqa: BLE001
                entry["exclusion_reason"] = f"analyzer:{type(error).__name__}"
            else:
                entry["verdict"] = result.verdict.value
                if result.verdict in {Verdict.ERROR, Verdict.NOT_APPLICABLE}:
                    entry["exclusion_reason"] = result.verdict.value
                elif result.score is None or not math.isfinite(result.score):
                    entry["exclusion_reason"] = "invalid_score"
                else:
                    entry["score"] = float(result.score)
                entry["members_available"] = result.evidence.get("members_available")
                entry["agreement_fake_frac"] = result.evidence.get("agreement_fake_frac")
            row["detectors"][ENSEMBLE_KEY] = entry
        rows.append(row)

    usable = [row for row in rows if row["input_exclusion"] is None]
    thresholds = {
        "deepfake_cues": float(cfg.value("deepfake", "score_thresh")),
        "deepfake_ml": float(cfg.value("deepfake_ml", "fake_thresh")),
        "deepfake_screen": float(cfg.value("deepfake_screen_coc", "fake_thresh")),
        LEARNED_KEY: LEARNED_THRESHOLD,
        ENSEMBLE_KEY: float(cfg.value("deepfake_ensemble", "fake_thresh")),
    }
    detector_summaries = {
        key: _summarize_detector(usable, key, thresholds[key]) for key in report_keys
    }
    return {
        "protocol": "deepfake-ensemble-study-v1",
        "scope": (
            "frame-level detector comparison on a labelled split; scores are heuristic indices, "
            "not probabilities; ensemble combines member scores by explicit weighted mean; "
            "no authenticity verdict"
        ),
        "split": split,
        "manifest_sha256": _digest(manifest),
        "config_sha256": config_digest(cfg.raw),
        "model_sha256": None if model is None else model["model_sha256"],
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__},
        "integrity": {"n_manifest_samples": len(samples), "n_hashed": len(hashes), "unhashed_ids": unhashed},
        "detectors": detector_summaries,
        "by_condition": {
            condition: _summarize_detector(
                [row for row in usable if row["condition"] == condition], ENSEMBLE_KEY,
                thresholds[ENSEMBLE_KEY],
            )
            for condition in sorted({row["condition"] for row in usable})
        },
        "samples": rows,
    }
