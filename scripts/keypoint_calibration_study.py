"""Offline study: keypoint verified-pair count vs simple extra gates (no detector changes).

Runs detect → self_match → geometric_verify exactly like ``keypoint.pipeline``,
records auxiliary features from inlier pairs, and compares rank AUC on the tuning
split. Does not modify shipped thresholds or verdict logic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from imageguard.core.config import ThresholdSet, load_thresholds  # noqa: E402
from imageguard.core.io import safe_open  # noqa: E402
from imageguard.core.models import ImageDocument  # noqa: E402
from imageguard.core.preprocess import preprocess_keypoint  # noqa: E402
from imageguard.forensics.keypoint.detectors import create_backend  # noqa: E402
from imageguard.forensics.keypoint.geo_verify import geometric_verify  # noqa: E402
from imageguard.forensics.keypoint.matchers import self_match  # noqa: E402

Label = Literal["forged", "original"]
Z_95 = 1.959963984540054
STUDY_SEED = 20261112
_FO_SUFFIX = re.compile(r"^(.+)_(F|O)\.(png|jpg|jpeg|tif|bmp)$", re.IGNORECASE)

DATASET_DIRS = {
    "synth_eval": "synth_eval",
    "tuning": "modelforge_audit_train_fixed",
    "holdout": "modelforge_audit_holdout_fixed",
    "comofod": "comofod_base",
}


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (endpoints in [0, 1])."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if successes < 0 or successes > n:
        raise ValueError("successes must satisfy 0 <= successes <= n")
    if n == 0:
        return (float("nan"), float("nan"))
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via average ranks (Mann–Whitney U), no third-party deps."""
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    n = int(labels.size)
    if n == 0:
        return float("nan")
    y = labels.astype(np.int8)
    s = scores.astype(np.float64)
    if np.any(~np.isfinite(s)):
        raise ValueError("scores must be finite")
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and s[order[j]] == s[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def spatial_spread_fraction(
    kps: list,
    pairs: np.ndarray,
    inlier_idx: np.ndarray,
    gray_shape: tuple[int, ...],
) -> float:
    """BBox area fraction covering all inlier keypoints (both endpoints) vs image."""
    h, w = int(gray_shape[0]), int(gray_shape[1])
    if h <= 0 or w <= 0:
        raise ValueError("gray_shape must be positive height and width")
    if inlier_idx.size == 0:
        return 0.0
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    sel_pairs = pairs[inlier_idx]
    kp_idx = np.unique(np.concatenate([sel_pairs[:, 0], sel_pairs[:, 1]]))
    if kp_idx.size == 0:
        return 0.0
    xy = pts[kp_idx]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    bbox_area = max(0.0, float(xmax - xmin)) * max(0.0, float(ymax - ymin))
    return bbox_area / float(w * h)


def median_inlier_pair_distance_px(
    kps: list,
    pairs: np.ndarray,
    inlier_idx: np.ndarray,
) -> float:
    if inlier_idx.size == 0:
        return 0.0
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    sel = pairs[inlier_idx]
    d = np.linalg.norm(pts[sel[:, 0]] - pts[sel[:, 1]], axis=1)
    return float(np.median(d))


@dataclass(frozen=True, slots=True)
class ImageMetrics:
    dataset: str
    split_role: str
    file_name: str
    label: Label
    n_keypoints: int
    n_verified: int
    n_clusters: int
    spatial_spread: float
    median_pair_distance_px: float
    insufficient_texture: bool


def analyze_image(doc: ImageDocument, cfg: ThresholdSet) -> ImageMetrics:
    """Mirror ``KeypointCopyMove.analyze`` through geometric_verify (no verdict)."""
    gray = preprocess_keypoint(doc, cfg)
    backend = create_backend(cfg)
    kps, desc = backend.detect(gray)
    n_kp = len(kps)
    k_min = int(cfg.value("keypoint", "k_min"))
    if n_kp < k_min or desc is None:
        return ImageMetrics(
            dataset="",
            split_role="",
            file_name=doc.file_name,
            label="original",
            n_keypoints=n_kp,
            n_verified=0,
            n_clusters=0,
            spatial_spread=0.0,
            median_pair_distance_px=0.0,
            insufficient_texture=True,
        )
    matches = self_match(kps, desc, cfg, backend.norm)
    ver = geometric_verify(matches.pairs, kps, cfg)
    n_ver = len(ver.inlier_pairs)
    spread = spatial_spread_fraction(kps, matches.pairs, ver.inlier_pairs, gray.shape)
    med_dist = median_inlier_pair_distance_px(kps, matches.pairs, ver.inlier_pairs)
    return ImageMetrics(
        dataset="",
        split_role="",
        file_name=doc.file_name,
        label="original",
        n_keypoints=n_kp,
        n_verified=n_ver,
        n_clusters=int(ver.n_clusters),
        spatial_spread=spread,
        median_pair_distance_px=med_dist,
        insufficient_texture=False,
    )


def _label_from_name(path: Path) -> Label | None:
    m = _FO_SUFFIX.match(path.name)
    if not m:
        return None
    return "forged" if m.group(2).upper() == "F" else "original"


def iter_labeled_images(root: Path) -> Iterable[tuple[Path, Label]]:
    if not root.is_dir():
        raise ValueError(f"dataset directory not found: {root.name}")
    seen: set[str] = set()
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        label = _label_from_name(path)
        if label is None:
            continue
        if path.name in seen:
            continue
        seen.add(path.name)
        yield path, label


def collect_dataset(
    data_root: Path,
    dataset_key: str,
    split_role: str,
    cfg: ThresholdSet,
) -> list[ImageMetrics]:
    sub = DATASET_DIRS[dataset_key]
    root = data_root / sub
    rows: list[ImageMetrics] = []
    for path, label in iter_labeled_images(root):
        doc = safe_open(path, cfg)
        if not isinstance(doc, ImageDocument):
            raise TypeError("safe_open must return ImageDocument")
        base = analyze_image(doc, cfg)
        rows.append(
            ImageMetrics(
                dataset=dataset_key,
                split_role=split_role,
                file_name=path.name,
                label=label,
                n_keypoints=base.n_keypoints,
                n_verified=base.n_verified,
                n_clusters=base.n_clusters,
                spatial_spread=base.spatial_spread,
                median_pair_distance_px=base.median_pair_distance_px,
                insufficient_texture=base.insufficient_texture,
            )
        )
    return rows


def _labels_and_scores(rows: list[ImageMetrics], score_fn) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([1 if r.label == "forged" else 0 for r in rows], dtype=np.int8)
    scores = np.array([float(score_fn(r)) for r in rows], dtype=np.float64)
    return labels, scores


def tune_threshold(
    rows: list[ImageMetrics],
    feature: Literal["spatial_spread", "median_pair_distance_px"],
) -> float:
    """Pick feature cutoff on tuning rows maximizing gated rank AUC."""
    if not rows:
        return float("nan")
    vals = sorted({float(getattr(r, feature)) for r in rows})
    if not vals:
        return float("nan")
    best_x = vals[0]
    best_auc = -1.0
    for x in vals:
        labels, scores = _labels_and_scores(
            rows,
            lambda r, thr=x: float(r.n_verified) if getattr(r, feature) >= thr else 0.0,
        )
        auc = rank_auc(labels, scores)
        if math.isnan(auc):
            continue
        if auc > best_auc:
            best_auc = auc
            best_x = x
    return best_x


def flag_counts(
    rows: list[ImageMetrics],
    t_susp: float,
    *,
    min_clusters: int | None = None,
    min_spread: float | None = None,
    min_med_dist: float | None = None,
) -> dict[str, Any]:
    def flagged(r: ImageMetrics) -> bool:
        if r.n_verified < t_susp:
            return False
        if min_clusters is not None and r.n_clusters < min_clusters:
            return False
        if min_spread is not None and r.spatial_spread < min_spread:
            return False
        return not (
            min_med_dist is not None and r.median_pair_distance_px < min_med_dist
        )

    forged = [r for r in rows if r.label == "forged"]
    orig = [r for r in rows if r.label == "original"]
    ff = sum(1 for r in forged if flagged(r))
    fo = sum(1 for r in orig if flagged(r))
    bf = sum(1 for r in forged if r.n_verified >= t_susp)
    bo = sum(1 for r in orig if r.n_verified >= t_susp)
    wf_f = wilson_interval(ff, len(forged))
    wf_o = wilson_interval(fo, len(orig))
    wb_f = wilson_interval(bf, len(forged))
    wb_o = wilson_interval(bo, len(orig))
    return {
        "n_forged": len(forged),
        "n_original": len(orig),
        "forged_flagged": ff,
        "original_flagged": fo,
        "forged_flagged_wilson_95": wf_f,
        "original_flagged_wilson_95": wf_o,
        "baseline_forged_flagged": bf,
        "baseline_original_flagged": bo,
        "baseline_forged_wilson_95": wb_f,
        "baseline_original_wilson_95": wb_o,
    }


def auc_report(rows: list[ImageMetrics]) -> dict[str, float]:
    if not rows:
        return {
            "n_verified_only": float("nan"),
            "gated_spread": float("nan"),
            "gated_median_dist": float("nan"),
            "gated_n_clusters_ge_2": float("nan"),
        }
    labels, base_scores = _labels_and_scores(rows, lambda r: float(r.n_verified))
    out = {"n_verified_only": rank_auc(labels, base_scores)}
    labels_c, scores_c = _labels_and_scores(
        rows, lambda r: float(r.n_verified) if r.n_clusters >= 2 else 0.0
    )
    out["gated_n_clusters_ge_2"] = rank_auc(labels_c, scores_c)
    spread_x = tune_threshold(rows, "spatial_spread")
    labels_s, scores_s = _labels_and_scores(
        rows,
        lambda r, thr=spread_x: float(r.n_verified) if r.spatial_spread >= thr else 0.0,
    )
    out["gated_spread"] = rank_auc(labels_s, scores_s)
    out["tuned_spread_min"] = spread_x
    dist_x = tune_threshold(rows, "median_pair_distance_px")
    labels_d, scores_d = _labels_and_scores(
        rows,
        lambda r, thr=dist_x: float(r.n_verified) if r.median_pair_distance_px >= thr else 0.0,
    )
    out["gated_median_dist"] = rank_auc(labels_d, scores_d)
    out["tuned_median_pair_distance_px_min"] = dist_x
    return out


def dataset_file_counts(data_root: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for key, sub in DATASET_DIRS.items():
        root = data_root / sub
        n_f = n_o = 0
        if root.is_dir():
            for path in root.iterdir():
                if not path.is_file():
                    continue
                lab = _label_from_name(path)
                if lab == "forged":
                    n_f += 1
                elif lab == "original":
                    n_o += 1
        counts[key] = {"forged_files": n_f, "original_files": n_o, "total_labeled": n_f + n_o}
    return counts


def run_study(data_root: Path) -> dict[str, Any]:
    if not data_root.is_dir():
        raise ValueError("data_root must be an existing directory")
    cfg = load_thresholds()
    t_susp = float(cfg.value("keypoint", "t_susp"))

    tuning = collect_dataset(data_root, "tuning", "tune", cfg)
    holdout = collect_dataset(data_root, "holdout", "holdout", cfg)
    synth = collect_dataset(data_root, "synth_eval", "exploratory", cfg)
    comofod = collect_dataset(data_root, "comofod", "comofod_check", cfg)

    tuning_auc = auc_report(tuning)
    spread_thr = float(tuning_auc.get("tuned_spread_min", float("nan")))
    dist_thr = float(tuning_auc.get("tuned_median_pair_distance_px_min", float("nan")))

    def summarize_split(name: str, rows: list[ImageMetrics]) -> dict[str, Any]:
        base = flag_counts(rows, t_susp)
        cluster = flag_counts(rows, t_susp, min_clusters=2)
        spread = flag_counts(rows, t_susp, min_spread=spread_thr)
        dist = flag_counts(rows, t_susp, min_med_dist=dist_thr)
        return {
            "split": name,
            "flag_baseline_t_susp": base,
            "gate_n_clusters_ge_2": cluster,
            "gate_spread_ge_tuned": spread,
            "gate_median_pair_distance_px_ge_tuned": dist,
            "rank_auc_on_split": auc_report(rows) if name == "tuning" else None,
        }

    payload: dict[str, Any] = {
        "study_seed": STUDY_SEED,
        "shipped_t_susp": t_susp,
        "thresholds_unchanged": True,
        "not_a_casia_verdict": True,
        "dataset_file_counts": dataset_file_counts(data_root),
        "tuning_rank_auc": tuning_auc,
        "tuned_gates_from_tuning_only": {
            "spatial_spread_min": spread_thr,
            "median_pair_distance_px_min": dist_thr,
            "n_clusters_min": 2,
        },
        "splits": [
            summarize_split("tuning", tuning),
            summarize_split("holdout", holdout),
            summarize_split("synth_eval", synth),
            summarize_split("comofod", comofod),
        ],
        "per_image_tuning": [asdict(r) for r in tuning],
        "per_image_holdout": [asdict(r) for r in holdout],
    }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Keypoint false-cue offline calibration study")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("IMAGEGUARD_DATA", "data")),
        help="Root containing synth_eval, modelforge_*, comofod_base (not committed paths)",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "reports" / "keypoint-calibration-study.json",
        help="Write full metrics JSON (default under reports/, gitignored)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(STUDY_SEED)
    report = run_study(args.data_root.resolve())
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "out_json": args.out_json.name,
        "dataset_file_counts": report["dataset_file_counts"],
        "tuning_rank_auc": report["tuning_rank_auc"],
        "tuned_gates": report["tuned_gates_from_tuning_only"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
