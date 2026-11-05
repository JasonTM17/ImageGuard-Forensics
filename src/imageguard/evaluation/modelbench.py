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

import hashlib
import json
import math
import platform
import subprocess
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.forensics.base import Method

# Repo root (src/imageguard/evaluation/modelbench.py → 3 levels up).
_REPO_ROOT = Path(__file__).resolve().parents[3]

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


def localization_scores(pred: np.ndarray | None, gt: np.ndarray) -> dict:
    """Score every non-empty GT, counting a missing/invalid mask as a miss."""
    status = ("missing" if pred is None else
              "valid" if pred.shape == gt.shape else "shape_mismatch")
    result = {"mask_present": pred is not None, "mask_status": status,
              "iou": None, "bf1": None}
    if not (gt > 0).any():
        return result
    evaluated = pred if status == "valid" else np.zeros_like(gt)
    result["iou"] = iou(evaluated, gt)
    result["bf1"] = boundary_f1(evaluated, gt)
    return result


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
    n = n1 + n2
    # Hiệu chỉnh phương sai theo ties — không hiệu chỉnh thì p lệch khi có
    # nhiều block σ trùng giá trị (vùng phẳng) hoặc khi mọi giá trị trùng nhau.
    tie_sum = float((cnt.astype(float) ** 3 - cnt).sum())
    var = (n1 * n2 / 12.0) * ((n + 1) - tie_sum / (n * (n - 1)))
    if var <= 0:
        return float("nan")
    z = (u1 - n1 * n2 / 2) / math.sqrt(var)
    return float(math.erfc(abs(z) / math.sqrt(2)))


# ---------- runner ----------

def analyze_sample(
    doc, gt: np.ndarray, methods: list[Method], cfg: ThresholdSet
) -> dict:
    row: dict = {"methods": {}}
    gray = cv2.cvtColor(doc.array, cv2.COLOR_BGR2GRAY)
    sig = noise_sigma_map(gray)
    bh, bw = sig.shape
    if bh == 0 or bw == 0:
        # Ảnh nhỏ hơn block 16px → không có block nào → không có thống kê
        # in-vs-out; vẫn chấm các method, báo N/A thay vì crash cv2.resize.
        row["noise_sigma"] = {"in_median": None, "out_median": None,
                              "mwu_p": None}
    else:
        gt_blk = cv2.resize(gt.astype(np.uint8), (bw, bh),
                            interpolation=cv2.INTER_NEAREST) > 0
        sig_in, sig_out = sig[gt_blk], sig[~gt_blk]
        p = mann_whitney_u(sig_in, sig_out)
        row["noise_sigma"] = {
            "in_median": float(np.median(sig_in)) if len(sig_in) else None,
            "out_median": float(np.median(sig_out)) if len(sig_out) else None,
            # N/A (n quá nhỏ / một lớp) → None trong JSON, không phải NaN vô định.
            "mwu_p": p if math.isfinite(p) else None,
        }
    for m in methods:
        r = m.analyze(doc, cfg)
        ent = {"verdict": r.verdict.value, "score": r.score,
               "params_hash": r.params_hash}
        ent.update(localization_scores(r.mask, gt))
        row["methods"][m.name] = ent
    return row


def run_bench(
    set_dir: Path,
    methods: list[Method],
    cfg: ThresholdSet,
) -> dict:
    """Đọc meta.json của modelforge set → full report dict."""
    set_dir = Path(set_dir)
    manifest_path = set_dir / "meta.json"
    manifest_bytes = manifest_path.read_bytes()
    meta = json.loads(manifest_bytes)
    samples = []
    n_missing = 0
    n_unreadable = 0
    for e in meta:
        ext = ".jpg" if e.get("format") == "jpeg" else ".png"
        fp = set_dir / f"{e['id']}_F{ext}"
        doc = safe_open(fp, cfg)
        if hasattr(doc, "code"):
            n_unreadable += 1
            continue
        gt_path = set_dir / f"{e['id']}_M.png"
        gt_img = _imread_unicode(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_img is None:
            # GT mask thiếu/hỏng — bỏ qua sample, ghi nhận để báo cáo trung thực.
            n_missing += 1
            continue
        gt = gt_img > 0
        row = analyze_sample(doc, gt, methods, cfg)
        row.update(id=e["id"], kind=e["kind"], format=e.get("format", "png"),
                   gt_frac=float(gt.mean()))
        samples.append(row)
    originals, n_missing_originals = _run_original_controls(
        set_dir, meta, methods, cfg)
    git_sha, git_clean = _git_snapshot()
    run_id = (f"modelbench-{set_dir.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
              f"{uuid.uuid4().hex[:8]}")
    return {"experiment_id": run_id, "set": set_dir.name, "n": len(samples),
            "n_skipped_missing_gt": n_missing,
            "n_skipped_unreadable": n_unreadable,
            "original_controls": {
                "n": len(originals),
                "n_skipped": n_missing_originals,
                "samples": originals,
                "aggregate": aggregate_original_controls(originals),
            },
            "provenance": {
                "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "dataset_sha256": _dataset_sha256(set_dir, meta),
                "source_tree_sha256": _source_tree_sha256(),
                "config_hash": cfg.snapshot_hash(),
                "git_sha": git_sha,
                "working_tree_clean": git_clean,
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "generator_backends": sorted({str(e.get("backend", "unknown"))
                                               for e in meta}),
                "seeds": sorted({e.get("seed") for e in meta
                                 if e.get("seed") is not None}),
                "generator_model_sha256": sorted({e["model_sha256"] for e in meta
                                                  if e.get("model_sha256")}) or None,
                "methods": [m.name for m in methods],
            },
            "samples": samples,
            "aggregate": aggregate(samples)}


def _imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    """cv2.imread không đọc được đường dẫn non-ASCII trên Windows — đọc bytes
    rồi imdecode thay vì để mất sample âm thầm."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


def _dataset_sha256(set_dir: Path, meta: list[dict]) -> str:
    """Fingerprint manifest plus original, edited, and GT bytes in stable order."""
    digest = hashlib.sha256()
    paths = {set_dir / "meta.json"}
    for entry in meta:
        stem = entry["id"]
        ext = ".jpg" if entry.get("format") == "jpeg" else ".png"
        paths.update({set_dir / f"{stem}_O.png",
                      set_dir / f"{stem}_F{ext}",
                      set_dir / f"{stem}_M.png"})
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        if not path.is_file():
            digest.update(b"<missing>")
            continue
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256() -> str:
    """Fingerprint source modules and thresholds used by a local report."""
    package = Path(__file__).resolve().parents[1]
    root = _REPO_ROOT
    files = sorted(package.rglob("*.py"))
    files.append(root / "configs" / "thresholds.yaml")
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _run_original_controls(
    set_dir: Path,
    meta: list[dict],
    methods: list[Method],
    cfg: ThresholdSet,
) -> tuple[list[dict], int]:
    """Measure cue masks/verdicts on each untouched source image."""
    samples = []
    skipped = 0
    for entry in meta:
        image_path = set_dir / f"{entry['id']}_O.png"
        doc = safe_open(image_path, cfg)
        if hasattr(doc, "code"):
            skipped += 1
            continue
        results = {}
        for method in methods:
            result = method.analyze(doc, cfg)
            status = ("missing" if result.mask is None else
                      "valid" if result.mask.shape == doc.array.shape[:2]
                      else "shape_mismatch")
            area = (float((result.mask > 0).mean()) if status == "valid"
                    else 0.0 if status == "missing" else None)
            results[method.name] = {
                "verdict": result.verdict.value,
                "score": result.score,
                "params_hash": result.params_hash,
                "mask_present": result.mask is not None,
                "mask_status": status,
                "mask_area_ratio": area,
            }
        samples.append({"id": entry["id"], "kind": entry["kind"],
                        "methods": results})
    return samples, skipped


def aggregate_original_controls(samples: list[dict]) -> dict:
    """Summarize abstentions and non-empty cue masks on untouched images."""
    agg: dict[str, dict] = {}
    for sample in samples:
        for name, result in sample["methods"].items():
            cell = agg.setdefault(name, {
                "n": 0, "no_indication": 0, "inconclusive": 0,
                "suspicious": 0, "error": 0, "not_applicable": 0,
                "n_mask": 0, "n_mask_invalid": 0,
                "n_nonempty_mask": 0, "mask_areas": [],
            })
            cell["n"] += 1
            verdict = result["verdict"]
            if verdict == Verdict.NO_INDICATION.value:
                cell["no_indication"] += 1
            elif verdict == Verdict.INCONCLUSIVE.value:
                cell["inconclusive"] += 1
            elif verdict == Verdict.SUSPICIOUS_INDICATION.value:
                cell["suspicious"] += 1
            elif verdict == Verdict.ERROR.value:
                cell["error"] += 1
            elif verdict == Verdict.NOT_APPLICABLE.value:
                cell["not_applicable"] += 1
            if result["mask_present"]:
                cell["n_mask"] += 1
            if result["mask_status"] == "shape_mismatch":
                cell["n_mask_invalid"] += 1
            area = result["mask_area_ratio"]
            if area is not None:
                cell["mask_areas"].append(area)
                if area > 0:
                    cell["n_nonempty_mask"] += 1
    for cell in agg.values():
        cell["n_area_scored"] = len(cell["mask_areas"])
        cell["nonempty_mask_rate"] = (
            cell["n_nonempty_mask"] / cell["n"] if cell["n"] else None)
        cell["mean_mask_area_ratio"] = (
            float(np.mean(cell["mask_areas"])) if cell["mask_areas"] else None)
        cell["max_mask_area_ratio"] = (
            float(np.max(cell["mask_areas"])) if cell["mask_areas"] else None)
        del cell["mask_areas"]
    return agg


def _git_snapshot() -> tuple[str, bool | None]:
    """SHA + dirty flag của repo, độc lập với cwd; có timeout và decode UTF-8
    để không treo hay crash với đường dẫn non-ASCII trên Windows."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT,
            encoding="utf-8", errors="replace", timeout=10).strip()
        clean = not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_REPO_ROOT,
            encoding="utf-8", errors="replace", timeout=10).strip()
        return sha, clean
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def aggregate(samples: list[dict]) -> dict:
    agg: dict = {}
    for s in samples:
        kind = s["kind"]
        for name, m in s["methods"].items():
            cell = agg.setdefault(kind, {}).setdefault(
                name, {"n": 0, "suspicious": 0, "inconclusive": 0,
                       "n_mask": 0, "n_mask_invalid": 0,
                       "ious": [], "bf1s": []})
            cell["n"] += 1
            if m["verdict"] == Verdict.SUSPICIOUS_INDICATION.value:
                cell["suspicious"] += 1
            elif m["verdict"] == Verdict.INCONCLUSIVE.value:
                cell["inconclusive"] += 1
            if m.get("mask_present"):
                cell["n_mask"] += 1
            if m.get("mask_status") == "shape_mismatch":
                cell["n_mask_invalid"] += 1
            if m.get("iou") is not None:
                cell["ious"].append(m["iou"])
                cell["bf1s"].append(m["bf1"])
    for kind, meths in agg.items():
        for name, c in meths.items():
            c["detection_rate"] = round(c["suspicious"] / c["n"], 3)
            c["n_iou_scored"] = len(c["ious"])
            c["mean_iou"] = round(float(np.mean(c["ious"])), 3) if c["ious"] else None
            c["mean_bf1"] = round(float(np.mean(c["bf1s"])), 3) if c["bf1s"] else None
            del c["ious"], c["bf1s"]
    return agg


def render_markdown(res: dict) -> str:
    skip_bits = []
    if res.get("n_skipped_missing_gt"):
        skip_bits.append(f"{res['n_skipped_missing_gt']} missing-GT")
    if res.get("n_skipped_unreadable"):
        skip_bits.append(f"{res['n_skipped_unreadable']} unreadable")
    skip_s = f", skipped {', '.join(skip_bits)}" if skip_bits else ""
    lines = [
        "# ModelForge Detection Bench",
        "",
        f"Experiment: `{res.get('experiment_id', 'unrecorded')}`",
        f"Set: `{res['set']}` — n={res['n']} (mỗi dòng: edit-kind × method{skip_s})",
        "",
        "| kind | method | n | masks | IoU scored | SUSPICIOUS | INCONCLUSIVE | det-rate | IoU (all scored) | bF1@8 (all scored) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for kind, meths in sorted(res["aggregate"].items()):
        for name, c in sorted(meths.items()):
            iou_s = f"{c['mean_iou']:.3f}" if c["mean_iou"] is not None else "—"
            bf_s = f"{c['mean_bf1']:.3f}" if c["mean_bf1"] is not None else "—"
            mask_n = c.get("n_mask", 0)
            score_n = c.get("n_iou_scored", 0)
            lines.append(f"| {kind} | {name} | {c['n']} | {mask_n}/{c['n']} "
                         f"| {score_n}/{c['n']} | {c['suspicious']} "
                         f"| {c['inconclusive']} | {c['detection_rate']} "
                         f"| {iou_s} | {bf_s} |")
    provenance = res.get("provenance")
    if provenance:
        model_sha = ", ".join(provenance.get("generator_model_sha256") or [])
        lines += ["", "## Provenance", "",
                  f"- Config hash: `{provenance.get('config_hash', 'unknown')}`",
                  (f"- Git: `{provenance.get('git_sha', 'unknown')}` "
                   f"(working tree clean: `{provenance.get('working_tree_clean')}`)"),
                  f"- Dataset SHA-256: `{provenance.get('dataset_sha256', 'unknown')}`",
                  f"- Manifest SHA-256: `{provenance.get('dataset_manifest_sha256', 'unknown')}`",
                  f"- Source tree SHA-256: `{provenance.get('source_tree_sha256', 'unknown')}`",
                  f"- Backend(s): `{', '.join(provenance.get('generator_backends', []))}`",
                  f"- Seed(s): `{', '.join(map(str, provenance.get('seeds', [])))}`",
                  (f"- Runtime: Python `{provenance.get('python')}`, OpenCV `"
                   f"{provenance.get('opencv')}`, NumPy `{provenance.get('numpy')}`"),
                  f"- Generator model SHA-256: `{model_sha}`"]
    controls = res.get("original_controls")
    if controls:
        lines += ["", "## Untouched original controls", "",
                  f"N={controls['n']}, skipped={controls['n_skipped']}", "",
                  "| method | NO_INDICATION | INCONCLUSIVE | SUSPICIOUS | ERROR | NOT_APPLICABLE | non-empty masks | invalid masks | area scored | mean mask area | max mask area |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, cell in sorted(controls["aggregate"].items()):
            mean_area = cell["mean_mask_area_ratio"]
            max_area = cell["max_mask_area_ratio"]
            mean_s = f"{mean_area:.3%}" if mean_area is not None else "—"
            max_s = f"{max_area:.3%}" if max_area is not None else "—"
            lines.append(
                f"| {name} | {cell['no_indication']} | {cell['inconclusive']} "
                f"| {cell['suspicious']} | {cell['error']} "
                f"| {cell['not_applicable']} | {cell['n_nonempty_mask']}/{cell['n']} "
                f"| {cell['n_mask_invalid']} | {cell['n_area_scored']}/{cell['n']} "
                f"| {mean_s} | {max_s} |")
    lines += ["", "## Noise-σ separation (in-GT vs out-GT)", "",
              "| sample | kind | σ_in | σ_out | MWU p |", "|---|---|---|---|---|"]
    for s in res["samples"]:
        ns = s["noise_sigma"]
        p = ns["mwu_p"]
        p_s = ("—" if p is None else
               f"{p:.4g}" if p > 0 else "<5e-324")
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
