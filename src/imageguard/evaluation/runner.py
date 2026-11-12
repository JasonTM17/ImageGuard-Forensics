"""Evaluation runner — chạy methods trên dataset, tái lập hoàn toàn.

Mọi report ghi provenance: {seed, config_hash, git_sha, working_tree_clean,
dataset_version,
n, n_excluded, n_anomaly}. Verdict→label: SUSPICIOUS → forged;
NO_INDICATION/INCONCLUSIVE → authentic (INCONCLUSIVE đếm riêng — độ thận trọng).
NOT_APPLICABLE và ERROR loại khỏi xét hoàn toàn (đếm n_not_applicable/n_error) —
docs/evaluation-methodology.md. GT mask sai kích thước → exclude + đếm;
GT mask rỗng trên forged → data anomaly → exclude + đếm
(docs/evaluation-methodology.md: mask thiếu/sai bị loại và đếm, không tự resize).
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.core.models import ImageDocument
from imageguard.evaluation.datasets import DatasetAdapter
from imageguard.evaluation.metrics import aggregate_pixel, image_metrics, pixel_metrics
from imageguard.forensics.base import Method


@dataclass(slots=True)
class EvalReport:
    experiment_id: str
    dataset: str
    split: str
    provenance: dict
    per_method: dict[str, dict] = field(default_factory=dict)
    per_condition: dict[str, dict] = field(default_factory=dict)

    def save(self, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        jp = out_dir / f"eval-{self.experiment_id}.json"
        cp = out_dir / f"eval-{self.experiment_id}.csv"
        jp.write_text(json.dumps({
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "split": self.split,
            "provenance": self.provenance,
            "per_method": self.per_method,
            "per_condition": self.per_condition,
        }, indent=1, default=str), encoding="utf-8")
        with cp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "metric", "value", "n"])
            for m, d in self.per_method.items():
                for k, v in _flatten(d).items():
                    w.writerow([m, k, v, d.get("n", "")])
        return jp, cp


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _git_snapshot() -> tuple[str, bool | None]:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root,
            encoding="utf-8", errors="replace", timeout=10).strip()
        clean = not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root,
            encoding="utf-8", errors="replace", timeout=10).strip()
        return sha, clean
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def run_eval(
    adapter: DatasetAdapter,
    methods: list[Method],
    cfg: ThresholdSet,
    split: str = "dev",
    seed: int = 42,
) -> EvalReport:
    per_method_acc: dict[str, dict] = {
        m.name: {"pred": [], "gt": [], "pixel": [], "n": 0,
                 "n_excluded": 0, "n_inconclusive": 0,
                 "n_not_applicable": 0, "n_error": 0,
                 "n_clean_correct": 0, "fp_area_ratios": [],
                 "durations_ms": []}
        for m in methods
    }
    per_cond: dict[str, dict] = {}

    for sample in adapter.samples(split):
        doc = safe_open(sample.image_path, cfg)
        if not isinstance(doc, ImageDocument):
            for m in methods:
                per_method_acc[m.name]["n_excluded"] += 1
            continue
        gt_forged = sample.label == "forged"
        try:
            gt_mask = _load_mask(sample.gt_mask_path, doc)
        except ValueError:
            # mask sai kích thước — docs/dataset-strategy: reject + log,
            # KHÔNG silent resize, KHÔNG để một sample hỏng giết cả eval.
            for m in methods:
                per_method_acc[m.name]["n_excluded"] += 1
            continue
        if gt_mask is None and (gt_forged or sample.gt_mask_path is not None):
            # A missing mask is optional for authentic samples, but a forged
            # sample or explicitly supplied unreadable mask has no usable GT.
            for m in methods:
                per_method_acc[m.name]["n_excluded"] += 1
            continue
        if gt_forged and gt_mask is not None and not (gt_mask > 0).any():
            for m in methods:
                per_method_acc[m.name]["n_excluded"] += 1
            continue  # data anomaly: forged file mà GT rỗng → exclude+log

        for m in methods:
            acc = per_method_acc[m.name]
            try:
                r = m.analyze(doc, cfg)
            except Exception:  # noqa: BLE001 - method isolation: method crash không giết cả eval
                acc["n_excluded"] += 1
                continue
            acc["durations_ms"].append(r.duration_ms)
            # Methodology (docs/evaluation-methodology.md): NOT_APPLICABLE và
            # ERROR loại khỏi quyết định — KHÔNG đếm thành 'authentic'.
            if r.verdict == Verdict.NOT_APPLICABLE:
                acc["n_not_applicable"] += 1
                continue
            if r.verdict == Verdict.ERROR:
                acc["n_error"] += 1
                continue
            acc["n"] += 1
            pred_forged = r.verdict == Verdict.SUSPICIOUS_INDICATION
            if r.verdict == Verdict.INCONCLUSIVE:
                acc["n_inconclusive"] += 1
            acc["pred"].append(pred_forged)
            acc["gt"].append(gt_forged)

            if gt_forged and gt_mask is not None:
                pm = pixel_metrics(r.mask, gt_mask)
                acc["pixel"].append(pm)
            else:
                # authentic: R3/R4 bookkeeping
                if r.mask is not None and (r.mask > 0).any():
                    acc["fp_area_ratios"].append(
                        float((r.mask > 0).sum()) / r.mask.size)  # R3
                else:
                    acc["n_clean_correct"] += 1  # R4

            cacc = per_cond.setdefault(
                sample.category,
                {m2.name: {"pred": [], "gt": [], "n": 0} for m2 in methods})
            cacc[m.name]["pred"].append(pred_forged)
            cacc[m.name]["gt"].append(gt_forged)
            cacc[m.name]["n"] += 1

    git_sha, working_tree_clean = _git_snapshot()
    report = EvalReport(
        experiment_id=f"{adapter.name}-{split}-{int(time.time())}-{uuid.uuid4().hex[:8]}",
        dataset=adapter.name, split=split,
        provenance={
            "seed": seed,
            "config_hash": cfg.snapshot_hash(),
            "git_sha": git_sha,
            "working_tree_clean": working_tree_clean,
            "dataset_version": adapter.version(),
        },
    )
    for name, acc in per_method_acc.items():
        im = image_metrics(acc["pred"], acc["gt"])
        px = aggregate_pixel(acc["pixel"])
        durs = acc["durations_ms"]
        report.per_method[name] = {
            **im,
            "n_scored": acc["n"], "n_excluded": acc["n_excluded"],
            "n_inconclusive": acc["n_inconclusive"],
            "n_not_applicable": acc["n_not_applicable"],
            "n_error": acc["n_error"],
            "n_clean_correct": acc["n_clean_correct"],
            "fp_area_ratio_mean": (float(np.mean(acc["fp_area_ratios"]))
                                   if acc["fp_area_ratios"] else None),
            "pixel_macro": px["macro"], "pixel_micro": px["micro"],
            "pixel_n_images": px["n_images_scored"],
            "runtime_ms_mean": float(np.mean(durs)) if durs else None,
            "runtime_ms_median": float(np.median(durs)) if durs else None,
        }
    report.per_condition = {
        cat: {name: {**image_metrics(v["pred"], v["gt"]), "n": v["n"]}
              for name, v in methods_d.items()}
        for cat, methods_d in per_cond.items()
    }
    return report


def _load_mask(path, doc: ImageDocument) -> np.ndarray | None:
    if path is None:
        return None
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    try:
        m = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    except cv2.error:
        return None
    if m is None:
        return None
    if m.shape != (doc.height, doc.width):
        raise ValueError(f"mask {path.name} dims {m.shape} != image "
                         f"{doc.height}x{doc.width} — reject")
    return m
