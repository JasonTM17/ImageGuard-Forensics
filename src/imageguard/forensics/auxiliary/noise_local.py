"""Noise-σ REGION detector — vùng σ thấp bất thường (model-fill mịn hơn nền).

Motivation đo được (docs/modelforge.md): vùng LaMa-inpaint có σ-residual
thấp hơn nền với p<0.001 trên mọi mẫu erase/move, nhưng cả global-MAD
(median toàn ảnh, IoU≈0.02) lẫn local-MAD (láng giềng hỗn hợp → MAD phình
→ z chìm, n_low≈0) đều thất bại localize. Lý do đúng: cue là MỘT PHÍA —
vùng sinh-mới MỊN hơn, không phải "lệch 2 phía so với median".

Thuật toán v4:
  σ_b   = std(residual) per block B×B (integral image)
  t1    = q  · median(σ) — ứng viên: σ thấp hơn nền (q≈0.6)
  t2    = q2 · median(σ) — component-mean σ < t2 mới giữ (q2≈0.45):
          LaMa-fill mịn SÂU hơn flat tự nhiên; đo GT-comp 3.2-4.9 vs
          natural-flat 4.5-5.7; tầng 2 giảm nhưng KHÔNG triệt tiêu cue trên ảnh gốc
          (holdout 1/6 control còn mask khác rỗng — docs/modelforge.md).
  flag  = component đủ lớn (≥min_region_blocks) và đủ mịn
  evidence thêm seam-ring: local-MAD z_cao (đã đo: KHÔNG tách được
  edited vs natural — giữ làm evidence mô tả, không dùng để lọc).

q=0.6/q2=0.45 chọn từ bộ seed 2026, validate held-out seed 7777 —
ngưỡng thực nghiệm tạm, chưa hiệu chỉnh trên ảnh thật (docs/modelforge.md).

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
from imageguard.core.preprocess import (
    aux_limitation,
    aux_working_copy,
    scaled_block,
    scaled_kernel,
    upscale_to_doc,
)
from imageguard.forensics.base import cancelled

SCORE_DEF = ("tỉ lệ block có σ < q·median toàn ảnh sau lọc component "
             "(vùng residual mịn bất thường — ứng viên model-fill)")


class NoiseLocal:
    """Vùng σ residual **thấp** bất thường (ứng viên inpaint/model-fill).

    Hai tầng connected-components + ngưỡng q, q2 (chủ đề: noise /
    segmentation). **Aux**: INCONCLUSIVE hoặc NO_INDICATION — không SUSPICIOUS.
    """

    name = "noise_local"

    def analyze(
        self,
        doc: ImageDocument,
        cfg: ThresholdSet,
        cancel: threading.Event | None = None,
    ) -> AlgorithmResult:
        """Cờ block có σ < q·median; giữ component đủ lớn và mean σ < q2·median.

        Score = tỉ lệ block trong vùng giữ; seam-ring z_hi chỉ evidence.
        """
        t0 = time.perf_counter()
        lim = [
            (
                "block nhiễu/mịn bất thường CỤC BỘ có thể là chuyển vùng texture "
                "tự nhiên (vật phẳng cạnh nền nhiễu) — không đồng nghĩa chỉnh sửa"
            ),
            "độ chính xác giảm ở viền ảnh (median láng giềng nhân bản mép)",
            "signal khám phá — corroborate bằng detector conclusive",
        ]
        k = int(cfg.value("noise", "median_k"))
        bs = int(cfg.value("noise", "block"))
        kl = int(cfg.value("noise", "local_k"))
        # cv2.medianBlur với CV_32F chỉ hỗ trợ ksize 3 và 5 — giá trị khác
        # raise AssertionError. Clamp [3,5] và ép số lẻ.
        kl = min(5, max(3, kl))
        if kl % 2 == 0:
            kl = 5 if kl == 4 else 3
        tz = float(cfg.value("noise", "local_z"))
        min_frac = float(cfg.value("noise", "local_min_frac"))

        # Ảnh lớn → phân tích trên bản downscale, kernel/block giữ kích thước vật lý.
        work, scale = aux_working_copy(doc, cfg)
        aux_lim = aux_limitation(scale)
        if aux_lim:
            lim.append(aux_lim)
        k_eff = scaled_kernel(k, scale)
        bs_eff = scaled_block(bs, scale)
        kl_eff = scaled_kernel(kl, scale)
        # Ngưỡng tính theo block; block và working copy cùng scale nên số block
        # tối thiểu không đổi khi downscale.
        min_comp = int(cfg.value("noise", "min_region_blocks"))

        gray = cv2.cvtColor(work.array, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim)
        r = gray - cv2.medianBlur(gray, k_eff).astype(np.float32)

        h, w = r.shape
        bh, bw = h // bs_eff, w // bs_eff
        if bh < kl_eff + 2 or bw < kl_eff + 2:
            return self._done(
                Verdict.NOT_APPLICABLE, None, t0,
                [f"ảnh quá nhỏ cho block {bs_eff}px + láng giềng {kl_eff}"] + lim)

        # cv2.integral trả integ(y,x) = Σ r[:y, :x] (h+1×w+1) — KHÔNG slice [1:,1:].
        integ2 = cv2.integral(r * r)
        integ1 = cv2.integral(r)
        ys, xs = np.mgrid[0:bh, 0:bw] * bs_eff
        e2 = (integ2[ys + bs_eff, xs + bs_eff] - integ2[ys, xs + bs_eff]
              - integ2[ys + bs_eff, xs] + integ2[ys, xs]) / (bs_eff * bs_eff)
        e1 = (integ1[ys + bs_eff, xs + bs_eff] - integ1[ys, xs + bs_eff]
              - integ1[ys + bs_eff, xs] + integ1[ys, xs]) / (bs_eff * bs_eff)
        sigma = np.sqrt(np.clip(e2 - e1 * e1, 0, None)).astype(np.float32)

        # Cue chính: vùng σ THẤP hơn q×median toàn ảnh — model-fill mịn
        # ~40-70% (đo thật: σ_in 2-7 vs σ_out 9.5-11.5, q=0.6 an toàn giữa
        # hai cụm). Một phía + tỉ lệ cứng thay vì Otsu (vùng edit ~1% lưới
        # → Otsu không tách được mode).
        med_g = float(np.median(sigma))
        q = float(cfg.value("noise", "low_sigma_ratio"))
        t_sig = q * med_g
        low = (sigma < t_sig).astype(np.uint8)
        # Lọc component hai tầng: (1) đủ lớn → loại nhiễu lẻ;
        # (2) component-mean σ < q2·median — LaMa-fill mịn SÂU hơn flat
        # tự nhiên (đo: GT-comp σ≈3.2-4.9 vs natural-flat ≈4.5-5.7).
        # Tầng 2 giảm flag-rate trên ảnh gốc (không phải 0% tuyệt đối — xem docs/modelforge.md).
        q2 = float(cfg.value("noise", "region_sigma_ratio"))
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(low, 8)
        keep = np.zeros_like(low)
        kept_area = 0
        for i in range(1, n_lbl):
            if stats[i, cv2.CC_STAT_AREA] < min_comp:
                continue
            cm = lbl == i
            if sigma[cm].mean() < q2 * med_g:
                keep[cm] = 1
                kept_area += int(stats[i, cv2.CC_STAT_AREA])
        frac = kept_area / max(bh * bw, 1)

        # Cue phụ (evidence/heatmap): seam-ring local-MAD high-z.
        med_l = cv2.medianBlur(sigma, kl_eff)
        mad_l = cv2.medianBlur(np.abs(sigma - med_l), kl_eff) + 1e-6
        z_hi = (sigma - med_l) / (1.4826 * mad_l)
        n_hi = int((z_hi >= tz).sum())

        # Heatmap tăng khi σ thấp hơn median; giá trị 0.5 tại σ = median.
        heat = np.clip((med_g - sigma) / (2 * med_g * q + 1e-6) + 0.5, 0, 1)
        heat_img = upscale_to_doc(heat, doc, cv2.INTER_LINEAR)
        mask = upscale_to_doc((keep * 255).astype(np.uint8), doc,
                              cv2.INTER_NEAREST)
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
                      "block": bs_eff, "scale": round(scale, 4),
                      "min_region_blocks": min_comp},
            params_hash=cfg.snapshot_hash())

    def _done(self, verdict, score, t0, limitations, evidence=None,
              mask=None, heatmap=None, params_hash=""):
        return AlgorithmResult(
            method=MethodName.NOISE_LOCAL, verdict=verdict,
            score=score, score_definition=SCORE_DEF, mask=mask,
            heatmap=heatmap,
            evidence={"variant": "two_tier_low_sigma", **(evidence or {})},
            limitations=limitations, params_hash=params_hash,
            duration_ms=int((time.perf_counter() - t0) * 1000))
