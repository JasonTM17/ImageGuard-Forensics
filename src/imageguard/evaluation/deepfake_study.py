"""Evaluate exploratory face cues without turning abstentions into negatives.

This module measures cue behavior on independently labelled images. It does not
train a classifier, tune a threshold, or interpret the cue score as probability.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections import Counter
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from imageguard.core import limits
from imageguard.core.config import ThresholdSet, load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.errors import ErrorResult
from imageguard.core.io import safe_open
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues


def _digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _image_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (path.is_absolute() or "\\" in relative or ":" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("image paths must be relative POSIX paths inside the dataset root")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("image path escapes dataset root")
    return resolved


def validate_manifest(manifest: dict, root: Path) -> list[dict]:
    """Validate declared splits before running any detector or loading pixels."""
    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("expected a version 1 manifest object")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest.samples must be a nonempty list")
    ids: set[str] = set()
    paths: set[Path] = set()
    group_splits: dict[str, str] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("each sample must be an object")  # noqa: TRY004 - manifest errors share one CLI error contract
        for key in ("id", "path", "label", "split", "source_group"):
            if not isinstance(sample.get(key), str) or not sample[key].strip():
                raise ValueError(f"each sample requires a nonempty string {key}")
        if sample["label"] not in {"real", "fake"} or sample["split"] not in {"dev", "test"}:
            raise ValueError("labels must be real/fake and splits must be dev/test")
        if "condition" in sample and (not isinstance(sample["condition"], str) or not sample["condition"].strip()):
            raise ValueError("condition must be a nonempty string when supplied")
        path = _image_path(root, sample["path"])
        if sample["id"] in ids or path in paths:
            raise ValueError("duplicate sample id or image path")
        ids.add(sample["id"])
        paths.add(path)
        group, split = sample["source_group"], sample["split"]
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"source_group {group!r} appears in both dev and test")
        group_splits[group] = split
    return samples


def _content_hashes(samples: list[dict], root: Path, max_bytes: int) -> tuple[dict[str, str], list[str]]:
    """Audit file-byte duplicates across ALL splits, with bounded reads."""
    hashes: dict[str, str] = {}
    seen: dict[str, str] = {}
    unhashed: list[str] = []
    for sample in samples:
        path = _image_path(root, sample["path"])
        try:
            if not path.is_file() or path.stat().st_size > max_bytes:
                unhashed.append(sample["id"])
                continue
            with path.open("rb") as stream:
                remaining = max_bytes + 1
                digest = hashlib.sha256()
                while remaining > 0:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    remaining -= len(block)
                    digest.update(block)
                if remaining == 0:
                    unhashed.append(sample["id"])
                    continue
        except OSError:
            unhashed.append(sample["id"])
            continue
        sha = digest.hexdigest()
        if sha in seen:
            raise ValueError(f"duplicate file content: {seen[sha]!r} and {sample['id']!r}")
        seen[sha] = sample["id"]
        hashes[sample["id"]] = sha
    return hashes, unhashed


def score_roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    """Rank-sum AUC with average tied ranks; None when either class is absent."""
    if len(scores) != len(labels):
        raise ValueError("score and label lengths differ")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("AUC scores must be finite")
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels, strict=True))
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ordered[start:end])
        start = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def summarize_samples(rows: list[dict]) -> dict:
    """Keep excluded samples visible, especially for asymmetric face coverage."""
    eligible = [row for row in rows if row["exclusion_reason"] is None]
    by_class = {}
    for label in ("real", "fake"):
        all_class = [row for row in rows if row["label"] == label]
        scored = [row for row in eligible if row["label"] == label]
        flagged = sum(row["cue_flag"] for row in scored)
        by_class[label] = {
            "n_total": len(all_class),
            "n_eligible": len(scored),
            "n_excluded": len(all_class) - len(scored),
            "coverage": len(scored) / len(all_class) if all_class else None,
            "n_cued": flagged,
            "n_uncued": len(scored) - flagged,
            "cue_rate_eligible": flagged / len(scored) if scored else None,
            "cue_yield_all": flagged / len(all_class) if all_class else None,
        }
    return {
        "n_total": len(rows),
        "n_eligible": len(eligible),
        "n_excluded": len(rows) - len(eligible),
        "coverage": len(eligible) / len(rows) if rows else None,
        "exclusions": dict(sorted(Counter(row["exclusion_reason"] for row in rows if row["exclusion_reason"]).items())),
        "by_class": by_class,
        "score_roc_auc_eligible": score_roc_auc(
            [row["score"] for row in eligible],
            [row["label"] == "fake" for row in eligible],
        ),
    }


def run_deepfake_study(
    manifest: dict, root: Path, *, split: str = "test", cfg: ThresholdSet | None = None,
) -> dict:
    """Run shipped face cues at one fixed configuration on a labelled split."""
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    samples = validate_manifest(manifest, root)
    selected = [sample for sample in samples if sample["split"] == split]
    if not selected:
        raise ValueError(f"manifest has no {split} samples")
    cfg = cfg or load_thresholds()
    hashes, unhashed = _content_hashes(samples, root, cfg.limit("max_file_bytes", limits.MAX_FILE_BYTES))
    analyzer = DeepfakeCues()
    rows: list[dict] = []
    for sample in selected:
        row = {key: sample[key] for key in ("id", "label", "source_group")}
        row.update(condition=sample.get("condition", "original"), sha256=hashes.get(sample["id"]),
                   score=None, cue_flag=None, verdict=None, exclusion_reason=None)
        doc = safe_open(_image_path(root, sample["path"]), cfg)
        if isinstance(doc, ErrorResult):
            row["exclusion_reason"] = f"input:{doc.code.value}"
        else:
            # Reject mutation between audit and decode rather than publish a
            # report whose manifest hashes describe different image bytes.
            if doc.sha256 != hashes.get(sample["id"]):
                raise ValueError(f"image changed during evaluation: {sample['id']!r}")
            if doc.capture_layout != "single":
                row["exclusion_reason"] = "unsupported_capture_layout"
                rows.append(row)
                continue
            try:
                result = analyzer.analyze(doc, cfg)
            except Exception as error:  # noqa: BLE001 - preserve every sample in evaluation denominator
                row["exclusion_reason"] = f"analyzer:{type(error).__name__}"
            else:
                row["verdict"] = result.verdict.value
                row["face_detected"] = result.evidence.get("face_detected", False)
                if result.evidence.get("cue_unavailable"):
                    row["exclusion_reason"] = result.evidence.get("unavailable_reason") or "cue_unavailable"
                elif not row["face_detected"]:
                    row["exclusion_reason"] = "no_face"
                elif result.verdict not in {Verdict.INCONCLUSIVE, Verdict.NO_INDICATION}:
                    row["exclusion_reason"] = f"verdict:{result.verdict.value}"
                elif result.score is None or not math.isfinite(result.score) or not 0 <= result.score <= 1:
                    row["exclusion_reason"] = "invalid_score"
                elif type(result.evidence.get("cue_flag")) is not bool:
                    row["exclusion_reason"] = "missing_cue_flag"
                else:
                    row["score"] = result.score
                    row["cue_flag"] = result.evidence["cue_flag"]
        rows.append(row)
    return {
        "protocol": "deepfake-cue-study-v1",
        "scope": "frame-level exploratory cue evaluation; scores are not probabilities; no authenticity verdict",
        "split": split,
        "manifest_sha256": _digest(manifest),
        "config_sha256": _digest(cfg.raw),
        "config": cfg.raw,
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__},
        "integrity": {"n_manifest_samples": len(samples), "n_hashed": len(hashes), "unhashed_ids": unhashed},
        "summary": summarize_samples(rows),
        "by_condition": {condition: summarize_samples([row for row in rows if row["condition"] == condition])
                         for condition in sorted({row["condition"] for row in rows})},
        "samples": rows,
    }
