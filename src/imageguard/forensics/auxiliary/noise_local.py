"""Noise-σ REGION detector — vùng σ thấp bất thường (model-fill mịn hơn nền).

Motivation đo được (docs/modelforge.md): vùng LaMa-inpaint có σ-residual
thấp hơn nền với p<0.001 trên mọi mẫu erase/move, nhưng cả global-MAD
(median toàn ảnh, IoU≈0.02) lẫn local-MAD (láng giềng hỗn hợp → MAD phình
→ z chìm, n_low≈0) đều thất bại localize. Lý do đúng: cue là MỘT PHÍA —
vùng sinh-mới MỊN hơn, không phải "lệch 2 phía so với median".

Thuật toán v3:
  σ_b   = std(residual) per block B×B (integral image)
  t     = Otsu(σ) — nếu phân phối σ lưỡng đỉnh, Otsu tự tách 2 mode
  flag  = σ < t   (vùng đuôi thấp = ứng viên fill mịn)
  giữ lại connected components ≥ min_comp block (vùng thật, không nhiễu lẻ)
  evidence thêm seam-ring: local-MAD z_cao (mép blend nhiễu bất thường).

Otsu chọn được vì nó thích nghi với độ tách hai mode thay vì ngưỡng cứng;
khi ảnh đơn đỉnh, đuôi thấp chỉ là vài % lẻ → component filter loại bỏ.

Giới hạn: vùng phẳng tự nhiên (bầu trời, tường) có σ thấp THẬT → false
positive theo đúng bản chất; Otsu yếu khi vùng chỉnh < vài % diện tích;
aux signal — KHÔNG trả SUSPICIOUS.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.forensics.auxiliary.noise_map import NoiseMap
from imageguard.forensics.base import cancelled

SCORE_DEF = ("tỉ lệ block có |z| local-MAD ≥ ngưỡng (z=(σ−med_l)/1.4826·mad_l, "
             "baseline = láng giềng block)")


class NoiseLocal:
    name = "noise_local"

    def analyze(
        self,
        doc: ImageDocument,
        cfg: ThresholdSet,
        cancel: threading.Event | None = None,
    ) -> AlgorithmResult:
        t0 = time.perf_counter()
        lim = [
            "block nhiễu/mịn bất thường CỤC BỘ có thể là chuyển vùng texture "
            "tự nhiên (vật phẳng cạnh nền nhiễu) — không đồng nghĩa chỉnh sửa",
            "độ chính xác giảm ở viền ảnh (median láng giềng nhân bản mép)",
            "signal khám phá — corroborate bằng detector conclusive",
        ]
        k = int(cfg.value("noise", "median_k"))
        bs = int(cfg.value("noise", "block"))
        kl = int(cfg.value("noise", "local_k"))
        tz = float(cfg.value("noise", "local_z"))
        min_frac = float(cfg.value("noise", "local_min_frac"))

        gray = cv2.cvtColor(doc.array, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim)
        r = gray - cv2.medianBlur(gray, k).astype(np.float32)

        h, w = r.shape
        bh, bw = h // bs, w // bs
        if bh < kl + 2 or bw < kl + 2:
            return self._done(
                Verdict.NOT_APPLICABLE, None, t0,
                [f"ảnh quá nhỏ cho block {bs}px + láng giềng {kl}"] + lim)

        integ2 = cv2.integral(r * r)[1:, 1:]
        integ1 = cv2.integral(r)[1:, 1:]
        ys, xs = np.mgrid[0:bh, 0:bw] * bs
        e2 = (integ2[ys + bs - 1, xs + bs - 1] - integ2[ys, xs + bs - 1]
              - integ2[ys + bs - 1, xs] + integ2[ys, xs]) / (bs * bs)
        e1 = (integ1[ys + bs - 1, xs + bs - 1] - integ1[ys, xs + bs - 1]
              - integ1[ys + bs - 1, xs] + integ1[ys, xs]) / (bs * bs)
        sigma = np.sqrt(np.clip(e2 - e1 * e1, 0, None)).astype(np.float32)

        # Cue chính: vùng σ THẤP hơn q×median toàn ảnh — model-fill mịn
        # ~40-70% (đo thật: σ_in 2-7 vs σ_out 9.5-11.5, q=0.6 an toàn giữa
        # hai cụm). Một phía + tỉ lệ cứng thay vì Otsu (vùng edit ~1% lưới
        # → Otsu không tách được mode).
        med_g = float(np.median(sigma))
        q = float(cfg.value("noise", "low_sigma_ratio"))
        t_sig = q * med_g
        low = (sigma < t_sig).astype(np.uint8)
        # lọc component: vùng fill thật liền khối, nhiễu lẻ rải rác
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(low, 8)
        keep = np.zeros_like(low)
        kept_area = 0
        min_comp = int(cfg.value("noise", "min_region_blocks"))
        for i in range(1, n_lbl):
            if stats[i, cv2.CC_STAT_AREA] >= min_comp:
                keep[lbl == i] = 1
                kept_area += int(stats[i, cv2.CC_STAT_AREA])
        frac = kept_area / max(bh * bw, 1)

        # Cue phụ (evidence/heatmap): seam-ring local-MAD high-z.
        med_l = cv2.medianBlur(sigma, kl)
        mad_l = cv2.medianBlur(np.abs(sigma - med_l), kl) + 1e-6
        z_hi = (sigma - med_l) / (1.4826 * mad_l)
        n_hi = int((z_hi >= tz).sum())

        # heatmap: độ "mịn hơn baseline" — 0.5 tại ngưỡng q·median
        heat = np.clip((med_g - sigma) / (2 * med_g * q + 1e-6) + 0.5, 0, 1)
        heat_img = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize((keep * 255).astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST)
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim)

        verdict = (Verdict.INCONCLUSIVE if frac >= min_frac
                   else Verdict.NO_INDICATION)
        if verdict == Verdict.INCONCLUSIVE:
            lim.append(
                f"{frac:.2%} block trong vùng σ-thấp "
                f"(t={t_sig:.2f}={q}·median {med_g:.2f})"
                " — ứng viên fill mịn, cần corroborate")
        return self._done(
            verdict, frac, t0, lim,
            mask=mask if frac > 0 else None, heatmap=heat_img,
            evidence={"low_sigma_t": round(t_sig, 3), "ratio": q,
                      "sigma_median": round(med_g, 3),
                      "kept_blocks": kept_area, "seam_ring_blocks": n_hi,
                      "block": bs},
            params_hash=cfg.snapshot_hash())

    def _done(self, verdict, score, t0, limitations, evidence=None,
              mask=None, heatmap=None, params_hash=""):
        return AlgorithmResult(
            method=MethodName.NOISE_INCONSISTENCY, verdict=verdict,
            score=score, score_definition=SCORE_DEF, mask=mask,
            heatmap=heatmap,
            evidence={"variant": "otsu_low_sigma", **(evidence or {})},
            limitations=limitations, params_hash=params_hash,
            duration_ms=int((time.perf_counter() - t0) * 1000))
