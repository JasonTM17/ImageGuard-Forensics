"""ModelBench — benchmark detector trên ảnh chỉnh sửa BẰNG MODEL (modelforge).

Trả lời câu hỏi capstone: "vùng bị model chỉnh/dịch chuyển có bị phát hiện
và localize được không?" — bằng số đo, không suy diễn.

Metrics per sample × method:
- verdict (enum)
- IoU(mask, GT) — localization cứng
- boundary-F1@tol — dung sai mép 8px: vùng GT thật có mép feather/seam,
  IoU thuần quá khắt khe với mask xác suất
- noise-σ separation: median σ_residual(in GT) vs (out GT) + Mann-Whitney U
  p-value — vùng model-sinh có "vân tay nhiễu" khác nền không? Đây là cue
  mạnh nhất lý thuyết chống inpainting generative.

Aggregate: per kind × method — detection rate, mean IoU, mean boundary-F1.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.forensics.base import Method


# ---------- metrics ----------

def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else 0.0


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 8) -> float:
    """F1 với dung sai tol px quanh mask — đo 'có chạm đúng vùng không'
    thay vì trùng pixel tuyệt đối (mask model-feather không thẳng mép)."""
    p, g = (pred > 0).astype(np.uint8), (gt > 0).astype(np.uint8)
    if g.sum() == 0:
        return float("nan")
    if p.sum() == 0:
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1,) * 2)
    p_d = cv2.dilate(p, k).astype(bool)
    g_d = cv2.dilate(g, k).astype(bool)
    tp_p = (p.astype(bool) & g_d).sum()          # pred hits near GT
    tp_g = (g.astype(bool) & p_d).sum()          # GT covered near pred
    prec = tp_p / max(p.sum(), 1)
    rec = tp_g / max(g.sum(), 1)
    return float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0


def noise_sigma_map(gray: np.ndarray, k: int = 5, bs: int = 16) -> np.ndarray:
    """σ residual per pixel (downsample theo block bs) — dùng cho
    in-vs-out statistical test."""
    r = gray.astype(np.float32) - cv2.medianBlur(gray, k).astype(np.float32)
    h, w = r.shape
    bh, bw = h // bs, w // bs
    r2 = r[:bh * bs, :bw * bs].reshape(bh, bs, bw, bs)
    return r2.std(axis=(1, 3))  # (bh, bw)


def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided p-value xấp xỉ chuẩn (n>8 mỗi bên → normal approx đủ tốt
    cho báo cáo khám phá; không dùng cho quyết định y khoa)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if n1 < 4 or n2 < 4:
        return float("nan")
    allv = np.concatenate([x, y])
    ranks = np.empty(len(allv))
    order = np.argsort(allv, kind="stable")
    ranks[order] = np.arange(1, len(allv) + 1)
    # mid-rank cho ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        csum = np.zeros(len(cnt))
        np.add.at(csum, inv, ranks)
        ranks = csum[inv] / cnt[inv]
    u1 = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2)
    mu, sd = n1 * n2 / 2, math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sd == 0:
        return float("nan")
    z = (u1 - mu) / sd
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


# ---------- runner ----------

def analyze_sample(
    doc, gt: np.ndarray, methods: list[Method], cfg: ThresholdSet
) -> dict:
    row: dict = {"methods": {}}
    gray = cv2.cvtColor(doc.array, cv2.COLOR_BGR2GRAY)
    sig = noise_sigma_map(gray)
    bh, bw = sig.shape
    bs = gray.shape[0] // bh
    gt_blk = cv2.resize(gt.astype(np.uint8), (bw, bh),
                        interpolation=cv2.INTER_NEAREST) > 0
    sig_in, sig_out = sig[gt_blk], sig[~gt_blk]
    row["noise_sigma"] = {
        "in_median": float(np.median(sig_in)) if len(sig_in) else None,
        "out_median": float(np.median(sig_out)) if len(sig_out) else None,
        "mwu_p": mann_whitney_u(sig_in, sig_out),
    }
    for m in methods:
        r = m.analyze(doc, cfg)
        ent = {"verdict": r.verdict.value, "score": r.score}
        if r.mask is not None and r.mask.shape == gt.shape:
            ent["iou"] = iou(r.mask, gt)
            ent["bf1"] = boundary_f1(r.mask, gt)
        row["methods"][m.name] = ent
    return row


def run_bench(
    set_dir: Path,
    methods: list[Method],
    cfg: ThresholdSet,
) -> dict:
    """Đọc meta.json của modelforge set → full report dict."""
    set_dir = Path(set_dir)
    meta = json.loads((set_dir / "meta.json").read_text(encoding="utf-8"))
    samples = []
    n_missing = 0
    for e in meta:
        ext = ".jpg" if e.get("format") == "jpeg" else ".png"
        fp = set_dir / f"{e['id']}_F{ext}"
        doc = safe_open(fp, cfg)
        if hasattr(doc, "code"):
            continue
        gt_path = set_dir / f"{e['id']}_M.png"
        gt_img = cv2.imread(str(gt_path), 0)
        if gt_img is None:
            # GT mask thiếu/hỏng — bỏ qua sample, ghi nhận để báo cáo trung thực.
            n_missing += 1
            continue
        gt = gt_img > 0
        row = analyze_sample(doc, gt, methods, cfg)
        row.update(id=e["id"], kind=e["kind"], format=e.get("format", "png"),
                   gt_frac=float(gt.mean()))
        samples.append(row)
    return {"set": str(set_dir), "n": len(samples),
            "n_skipped_missing_gt": n_missing, "samples": samples,
            "aggregate": aggregate(samples)}


def aggregate(samples: list[dict]) -> dict:
    agg: dict = {}
    for s in samples:
        kind = s["kind"]
        for name, m in s["methods"].items():
            cell = agg.setdefault(kind, {}).setdefault(
                name, {"n": 0, "suspicious": 0, "inconclusive": 0,
                       "ious": [], "bf1s": []})
            cell["n"] += 1
            if m["verdict"] == Verdict.SUSPICIOUS_INDICATION.value:
                cell["suspicious"] += 1
            elif m["verdict"] == Verdict.INCONCLUSIVE.value:
                cell["inconclusive"] += 1
            if "iou" in m:
                cell["ious"].append(m["iou"])
                cell["bf1s"].append(m["bf1"])
    for kind, meths in agg.items():
        for name, c in meths.items():
            c["detection_rate"] = round(c["suspicious"] / c["n"], 3)
            c["mean_iou"] = round(float(np.mean(c["ious"])), 3) if c["ious"] else None
            c["mean_bf1"] = round(float(np.mean(c["bf1s"])), 3) if c["bf1s"] else None
            del c["ious"], c["bf1s"]
    return agg


def render_markdown(res: dict) -> str:
    skipped = res.get("n_skipped_missing_gt")
    skip_s = f", skipped {skipped} missing-GT" if skipped else ""
    lines = [
        "# ModelForge Detection Bench",
        "",
        f"Set: `{res['set']}` — n={res['n']} (mỗi dòng: edit-kind × method{skip_s})",
        "",
        "| kind | method | n | SUSPICIOUS | INCONCLUSIVE | det-rate | IoU | bF1@8 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for kind, meths in sorted(res["aggregate"].items()):
        for name, c in sorted(meths.items()):
            iou_s = f"{c['mean_iou']:.3f}" if c["mean_iou"] is not None else "—"
            bf_s = f"{c['mean_bf1']:.3f}" if c["mean_bf1"] is not None else "—"
            lines.append(f"| {kind} | {name} | {c['n']} | {c['suspicious']} "
                         f"| {c['inconclusive']} | {c['detection_rate']} "
                         f"| {iou_s} | {bf_s} |")
    lines += ["", "## Noise-σ separation (in-GT vs out-GT)", "",
              "| sample | kind | σ_in | σ_out | MWU p |", "|---|---|---|---|---|"]
    for s in res["samples"]:
        ns = s["noise_sigma"]
        p = ns["mwu_p"]
        p_s = f"{p:.4f}" if p == p else "—"
        in_s = f"{ns['in_median']:.2f}" if ns['in_median'] is not None else "—"
        out_s = f"{ns['out_median']:.2f}" if ns['out_median'] is not None else "—"
        lines.append(f"| {s['id']} | {s['kind']} | "
                     f"{in_s} | {out_s} | {p_s} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="modelbench")
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--methods", default="keypoint,block,noise,freq,splicing,ela")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    from imageguard.core.config import load_thresholds
    from imageguard.forensics.registry import load_method

    cfg = load_thresholds()
    methods = [load_method(s.strip()) for s in a.methods.split(",")]
    res = run_bench(a.set, methods, cfg)
    md = render_markdown(res)
    if a.out:
        a.out = Path(a.out)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        (a.out.with_suffix(".json")).write_text(
            json.dumps(res, indent=1), encoding="utf-8")
        a.out.with_suffix(".md").write_text(md, encoding="utf-8")
        print(f"saved {a.out.with_suffix('.md')}")
    else:
        print(md)


if __name__ == "__main__":
    main()
