"""Noise inconsistency — vùng ảnh nguồn khác nhau có "vân tay nhiễu" khác nhau.

Nguyên lý: sensor noise (PRNU-like, thực tế chỉ đo σ residual) gần như đồng
nhất trong ảnh thật. Vùng bị ghép/làm mịn/denoise cục bộ → σ noise lệch.
Thuật toán:
  residual r = I − medianBlur(I, k)   (lọc trung vị tách texture mịn khỏi noise)
  với mỗi block B×B: σ_b = std(r trong block)
  outlier: |σ_b − median(σ)| > m · MAD(σ)
     MAD = median(|σ − median(σ)|) — robust, không bị vùng cực trị kéo như mean/std.

Giới hạn CỨNG (ghi vào result): vùng texture mạnh tự nhiên có σ cao (không
phải giả); JPEG blockiness làm méo residual; ảnh đã denoise mất dấu vết.
Signal này KHÔNG BAO GIỜ trả SUSPICIOUS — tối đa INCONCLUSIVE.
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

SCORE_DEF = "tỉ lệ block có σ residual là outlier MAD (|σ−median| > m·MAD)"


class NoiseMap:
    name = MethodName.NOISE_INCONSISTENCY.value

    def analyze(
        self,
        doc: ImageDocument,
        cfg: ThresholdSet,
        cancel: threading.Event | None = None,
    ) -> AlgorithmResult:
        t0 = time.perf_counter()
        lim = [
            ("σ residual cao có thể do texture/sắc cạnh tự nhiên — không đồng "
             "nghĩa vùng bị chỉnh sửa"),
            ("JPEG compression làm méo residual; ảnh đã denoise/sharpen mất "
             "dấu vết nhiễu gốc"),
            "signal khám phá — cần corroborate bằng detector conclusive",
        ]
        k = int(cfg.value("noise", "median_k"))
        bs = int(cfg.value("noise", "block"))
        m = float(cfg.value("noise", "outlier_mad"))
        min_frac = float(cfg.value("noise", "min_outlier_frac"))
        # Ảnh lớn → phân tích trên bản downscale, kernel giữ đúng kích thước vật lý.
        work, scale = aux_working_copy(doc, cfg)
        aux_lim = aux_limitation(scale)
        if aux_lim:
            lim.append(aux_lim)
        k_eff = scaled_kernel(k, scale)
        bs_eff = scaled_block(bs, scale)

        gray = cv2.cvtColor(work.array, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim)
        r = gray - cv2.medianBlur(gray, k_eff).astype(np.float32)

        h, w = r.shape
        bh, bw = h // bs_eff, w // bs_eff
        if bh < 2 or bw < 2:
            return self._done(
                Verdict.NOT_APPLICABLE, None, t0,
                [f"ảnh {w}×{h} quá nhỏ cho block {bs_eff}px"] + lim)

        # σ per block qua tích phân nhanh: E[x²]−E[x]² trên integral image.
        # cv2.integral trả integ(y,x) = Σ r[:y, :x] (shape h+1×w+1) — KHÔNG được
        # slice [1:,1:] (sẽ lệch 1px và thu nhỏ cửa sổ còn (bs-1)²).
        integ = cv2.integral(r * r)
        integ1 = cv2.integral(r)
        ys, xs = np.mgrid[0:bh, 0:bw] * bs_eff
        e2 = (integ[ys + bs_eff, xs + bs_eff] - integ[ys, xs + bs_eff]
              - integ[ys + bs_eff, xs] + integ[ys, xs]) / (bs_eff * bs_eff)
        e1 = (integ1[ys + bs_eff, xs + bs_eff] - integ1[ys, xs + bs_eff]
              - integ1[ys + bs_eff, xs] + integ1[ys, xs]) / (bs_eff * bs_eff)
        sigma = np.sqrt(np.clip(e2 - e1 * e1, 0, None))

        med = float(np.median(sigma))
        mad = float(np.median(np.abs(sigma - med))) + 1e-6
        outlier = np.abs(sigma - med) > m * mad
        frac = float(outlier.mean())
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim)

        heat = np.zeros((h, w), np.float32)
        mask = np.zeros((h, w), np.uint8)
        heat_blk = np.clip(np.abs(sigma - med) / (m * mad + 1e-6), 0, 1)
        for by in range(bh):
            for bx in range(bw):
                y0, x0 = by * bs_eff, bx * bs_eff
                heat[y0:y0 + bs_eff, x0:x0 + bs_eff] = heat_blk[by, bx]
                if outlier[by, bx]:
                    mask[y0:y0 + bs_eff, x0:x0 + bs_eff] = 255

        verdict = (Verdict.INCONCLUSIVE if frac >= min_frac
                   else Verdict.NO_INDICATION)
        if verdict == Verdict.INCONCLUSIVE:
            lim.append(
                f"{frac:.2%} block là outlier σ (median={med:.2f}, MAD={mad:.2f})"
                " — cần corroborate, không đủ để nghi ngờ")
        # Upscale về đúng kích thước ảnh gốc để UI/eval dùng chung mask.
        mask = upscale_to_doc(mask, doc, cv2.INTER_NEAREST)
        heat = upscale_to_doc(heat, doc, cv2.INTER_LINEAR)
        return self._done(
            verdict, frac, t0, lim,
            mask=mask if frac > 0 else None, heatmap=heat,
            evidence={"sigma_median": round(med, 3), "sigma_mad": round(mad, 3),
                      "outlier_frac": round(frac, 4), "block": bs_eff,
                      "scale": round(scale, 4),
                      "n_outlier_blocks": int(outlier.sum())},
            params_hash=cfg.snapshot_hash())

    def _done(self, verdict, score, t0, limitations, evidence=None,
              mask=None, heatmap=None, params_hash=""):
        return AlgorithmResult(
            method=MethodName.NOISE_INCONSISTENCY, verdict=verdict,
            score=score, score_definition=SCORE_DEF, mask=mask,
            heatmap=heatmap, evidence=evidence or {}, limitations=limitations,
            params_hash=params_hash,
            duration_ms=int((time.perf_counter() - t0) * 1000))
