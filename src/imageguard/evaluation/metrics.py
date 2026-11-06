"""Evaluation metrics — image-level + pixel-level với empty-mask rules R1–R5.

R1: pixel-metrics chỉ tính trên GT non-empty (ảnh authentic không có gì để đo).
R2: GT non-empty + pred empty → recall=0, F1=0, IoU=0, precision=N/A.
R3: GT empty + pred non-empty → image-level FP + fp_area_ratio riêng.
R4: cả hai empty → TN image-level, loại khỏi pixel aggregate + n_clean_correct.
R5: split rỗng 1 lớp → metric N/A (None). KHÔNG BAO GIỜ silent F1=1.
"""

from __future__ import annotations

import numpy as np


def image_metrics(pred: list[bool], gt: list[bool]) -> dict:
    """forged=positive. Trả counts + rates; N/A = None theo R5."""
    tp = sum(p and g for p, g in zip(pred, gt))
    fp = sum(p and not g for p, g in zip(pred, gt))
    tn = sum(not p and not g for p, g in zip(pred, gt))
    fn = sum(not p and g for p, g in zip(pred, gt))
    n = len(gt)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec)) else (0.0 if prec == 0.0 or rec == 0.0 else None)
    spec = tn / (tn + fp) if (tn + fp) else None
    acc = (tp + tn) / n if n else None
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n,
            "precision": prec, "recall": rec, "f1": f1,
            "specificity": spec, "accuracy": acc}


def pixel_metrics(pred_mask: np.ndarray | None, gt_mask: np.ndarray) -> dict:
    """Per-image pixel metrics. gt empty → R1 skip (caller phân loại R3/R4)."""
    gt = gt_mask > 0
    if not gt.any():
        return {"skipped": "R1: gt empty"}
    pred = (pred_mask > 0) if pred_mask is not None else np.zeros_like(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"mask dims {pred.shape} != gt {gt.shape} — reject, không silent resize")
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    # R2: pred empty → recall=0, f1=0, iou=0, precision N/A=None
    if not pred.any():
        return {"skipped": None, "precision": None, "recall": 0.0,
                "f1": 0.0, "iou": 0.0, "tp": 0, "fp": 0, "fn": int(gt.sum()),
                "rule": "R2"}
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if prec is not None and (prec + rec) else None
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else None
    return {"skipped": None, "precision": prec, "recall": rec, "f1": f1,
            "iou": iou, "tp": tp, "fp": fp, "fn": fn, "rule": None}


def aggregate_pixel(per_image: list[dict]) -> dict:
    """Macro (primary) + micro (secondary) trên GT non-empty images."""
    scored = [m for m in per_image if m.get("skipped") is None]
    n_px = len(scored)
    macro = {k: (float(np.mean([m[k] for m in scored if m[k] is not None]))
                 if any(m[k] is not None for m in scored) else None)
             for k in ("precision", "recall", "f1", "iou")} if n_px else \
        {"precision": None, "recall": None, "f1": None, "iou": None}
    tp = sum(m["tp"] for m in scored)
    fp = sum(m["fp"] for m in scored)
    fn = sum(m["fn"] for m in scored)
    micro_p = tp / (tp + fp) if (tp + fp) else None
    micro_r = tp / (tp + fn) if (tp + fn) else None
    if tp == 0 and fp == 0 and fn > 0:
        micro_f1 = 0.0
    else:
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) \
            if micro_p is not None and micro_r is not None and (micro_p + micro_r) else None
    micro_iou = tp / (tp + fp + fn) if (tp + fp + fn) else None
    return {"n_images_scored": n_px,
            "macro": macro,
            "micro": {"precision": micro_p, "recall": micro_r,
                      "f1": micro_f1, "iou": micro_iou}}


def boundary_tolerance(pred_mask: np.ndarray, gt_mask: np.ndarray, k: int = 3) -> dict:
    """Variant khoan dung biên: dilate GT ±k px rồi đo lại — báo RIÊNG."""
    import cv2

    if gt_mask.size == 0 or not (gt_mask > 0).any():
        return {"skipped": "R1: gt empty"}
    dil = cv2.dilate((gt_mask > 0).astype(np.uint8), np.ones((2 * k + 1,) * 2, np.uint8))
    return pixel_metrics(pred_mask, dil * 255)
