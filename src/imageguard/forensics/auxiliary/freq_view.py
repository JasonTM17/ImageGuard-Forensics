"""Frequency-domain cues — exploratory, không standalone verdict.

Hai cue:
1. FFT magnitude spectrum (log-scale, fftshift) — artifact để người dùng
   xem pattern lưới (resampling periodicity) hoặc dải tần bất thường.
2. JPEG 8×8 blockiness: chênh gradient TB tại đường biên bội-số-8 vs các
   cột/hàng khác. Ảnh JPEG có "ridge" mỗi 8px; ảnh đã resize/re-save có thể
   làm lệch → cue double-compression / không-phải-JPEG-gốc.

Giới hạn: blockiness cao ở ảnh JPEG bình thường (đúng bản chất nén); chỉ là
cue khám phá — KHÔNG trả SUSPICIOUS.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.core.preprocess import aux_limitation, aux_working_copy, upscale_to_doc
from imageguard.forensics.base import cancelled

SCORE_DEF = "blockiness = chênh |∇| tại biên 8×8 trừ |∇| nền, chuẩn hoá [0,1]"


class FreqView:
    """Cue tần số: FFT log-magnitude (artifact) + chỉ số JPEG blockiness.

    Chủ đề: frequency (FFT), JPEG grid 8×8. **Aux**: không SUSPICIOUS;
    JPEG → luôn NO_INDICATION (blockiness kỳ vọng); ảnh khác format có thể
    INCONCLUSIVE nếu blockiness cao.
    """

    name = MethodName.FREQUENCY_ANALYSIS.value

    def analyze(
        self,
        doc: ImageDocument,
        cfg: ThresholdSet,
        cancel: threading.Event | None = None,
    ) -> AlgorithmResult:
        """fftshift(fft2) chuẩn hoá làm heatmap; blockiness = |∇| biên 8px − nội.

        Verdict không đẩy fusion — exploratory cho người xem spectrum.
        """
        t0 = time.perf_counter()
        lim = [
            "blockiness cao là bản chất nén JPEG — không phải dấu hiệu sửa",
            "spectrum chỉ là artifact xem-mắt; không tự diễn giải thành vùng giả",
            "signal khám phá — không ảnh hưởng verdict tổng",
        ]
        t = float(cfg.value("freq", "blockiness_t"))

        # Ảnh lớn → phân tích trên bản downscale (FFT complex128 rất nặng);
        # cửa sổ blockiness đổi tỉ lệ thuận để giữ đúng kích thước vật lý.
        work, scale = aux_working_copy(doc, cfg)
        aux_lim = aux_limitation(scale)
        if aux_lim:
            lim.append(aux_lim)
        win = max(2, round(8 * scale))  # cửa sổ blockiness (px đã downscale)

        gray = cv2.cvtColor(work.array, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim)

        # --- cue 1: FFT spectrum artifact (heatmap kênh hiển thị) ---
        f = np.fft.fftshift(np.fft.fft2(gray))
        spec = np.log1p(np.abs(f))
        spec = (spec - spec.min()) / (np.ptp(spec) + 1e-9)
        spec = upscale_to_doc(spec.astype(np.float32), doc, cv2.INTER_LINEAR)

        # --- cue 2: JPEG blockiness ---
        # |∇x| TB theo cột; biên block = cột % win == win-1 (trừ cột 0).
        gx = np.abs(np.diff(gray, axis=1)).mean(axis=0)
        edge_cols = np.arange(win - 1, gray.shape[1] - 1, win)  # biên sau block
        inner = np.setdiff1d(np.arange(len(gx)), edge_cols)
        gx_b = float(gx[edge_cols].mean()) if len(edge_cols) else 0.0
        gx_i = float(gx[inner].mean()) + 1e-9
        blockiness = float(np.clip((gx_b - gx_i) / gx_i, 0, 1))

        is_jpeg = doc.image_format == "jpeg"
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim)
        ev = {"blockiness": round(blockiness, 4), "boundary_grad": round(gx_b, 3),
              "inner_grad": round(gx_i, 3), "format": doc.image_format,
              "window": win, "scale": round(scale, 4)}
        if is_jpeg:
            verdict = Verdict.NO_INDICATION
            lim.append(
                f"blockiness={blockiness:.3f} trên JPEG — vệt nén kỳ vọng; "
                "đọc spectrum để phát hiện pattern lưới bất thường")
        else:
            # Non-JPEG mà có blockiness → gợi ý đã qua JPEG → cue đáng chú ý.
            verdict = (Verdict.INCONCLUSIVE if blockiness >= t
                       else Verdict.NO_INDICATION)
            if blockiness >= t:
                lim.append(
                    f"blockiness={blockiness:.3f} ≥ {t} trên ảnh "
                    f"{doc.image_format} — gợi ý ảnh từng qua nén JPEG")
        return self._done(
            verdict, blockiness, t0, lim, heatmap=spec,
            evidence=ev, params_hash=cfg.snapshot_hash())

    def _done(self, verdict, score, t0, limitations, evidence=None,
              heatmap=None, params_hash=""):
        return AlgorithmResult(
            method=MethodName.FREQUENCY_ANALYSIS, verdict=verdict,
            score=score, score_definition=SCORE_DEF, heatmap=heatmap,
            evidence=evidence or {}, limitations=limitations,
            params_hash=params_hash,
            duration_ms=int((time.perf_counter() - t0) * 1000))
