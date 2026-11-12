"""Splicing experiment — vùng ghép TỪ ẢNH KHÁC (khác copy-move same-image).

Ý tưởng: vùng ghép từ nguồn khác thường mang noise signature + JPEG history
khác phần còn lại → noise-outlier map VÀ ELA hot-zone nên trùng nhau tại đó.
Cue: overlap(noise_outlier, ela_hot) / ela_hot ≥ ngưỡng → corroboration.

Khác copy-move: copy-move tìm HAI vùng giống nhau trong CÙNG ảnh (keypoint
self-match / block clone); splicing tìm vùng "lạ" về thống kê so với nền —
không cần twin.

Giới hạn CỨNG: thí nghiệm L2, chưa benchmark trên splicing dataset; phụ
thuộc ELA (ảnh non-JPEG → ELA NOT_APPLICABLE → method suy giảm chỉ còn
noise cue); KHÔNG trả SUSPICIOUS — tối đa INCONCLUSIVE.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.forensics.auxiliary.ela import ELA
from imageguard.forensics.auxiliary.noise_map import NoiseMap
from imageguard.forensics.base import cancelled

SCORE_DEF = "tỉ lệ diện tích ELA-hot trùng noise-outlier (corroboration)"


class SplicingCues:
    """Thí nghiệm ghép ảnh: giao mask noise-outlier × ELA-hot.

    Khác copy-move (không tìm twin). **Aux**: tối đa INCONCLUSIVE khi
    overlap_frac đủ lớn; thiếu mask ELA/noise → INCONCLUSIVE degraded.
    """

    name = "splicing_cues"

    def analyze(
        self,
        doc: ImageDocument,
        cfg: ThresholdSet,
        cancel: threading.Event | None = None,
    ) -> AlgorithmResult:
        """Gọi NoiseMap + ELA; overlap / diện tích ELA-hot là score.

        Không sinh SUSPICIOUS — chỉ gợi ý corroboration cho reviewer.
        """
        t0 = time.perf_counter()
        lim = [
            "signal thí nghiệm L2 — chưa benchmark trên splicing dataset",
            ("chỉ là cross-cue noise×ELA; vùng khớp nhau có thể do texture "
             "tự nhiên gây cả hai cùng nóng"),
            "KHÔNG trả SUSPICIOUS — corroboration cue cho reviewer",
        ]
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim)

        noise = NoiseMap().analyze(doc, cfg, cancel)
        ela = ELA().analyze(doc, cfg, cancel)
        ev = {"noise_verdict": noise.verdict.value,
              "ela_verdict": ela.verdict.value}
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim)

        if noise.mask is None or ela.mask is None:
            why = ("ELA không áp dụng (non-JPEG)" if ela.mask is None
                   else "noise map không có outlier")
            ev["cue_unavailable"] = True
            return self._done(
                Verdict.INCONCLUSIVE, None, t0,
                [f"cue suy giảm: {why} — không đủ cặp tín hiệu để đối chiếu"]
                + lim, evidence=ev, params_hash=cfg.snapshot_hash())

        inter = np.logical_and(noise.mask > 0, ela.mask > 0).sum()
        hot = max(int((ela.mask > 0).sum()), 1)
        frac = float(inter / hot)
        mask = np.zeros_like(ela.mask)
        mask[np.logical_and(noise.mask > 0, ela.mask > 0)] = 255
        ev.update({"overlap_frac": round(frac, 4),
                   "overlap_px": int(inter), "ela_hot_px": int(hot)})

        t = float(cfg.value("splicing", "overlap_frac"))
        verdict = Verdict.INCONCLUSIVE if frac >= t else Verdict.NO_INDICATION
        if verdict == Verdict.INCONCLUSIVE:
            lim.append(
                f"{frac:.1%} vùng ELA-hot trùng noise-outlier ≥ {t:.0%} — "
                "corroboration đáng xem thủ công, không phải kết luận")
        return self._done(
            verdict, frac, t0, lim, mask=mask, evidence=ev,
            params_hash=cfg.snapshot_hash())

    def _done(self, verdict, score, t0, limitations, evidence=None,
              mask=None, params_hash=""):
        return AlgorithmResult(
            method=MethodName.SPLICING_CUES,
            verdict=verdict, score=score, score_definition=SCORE_DEF,
            mask=mask, evidence=evidence or {},
            limitations=limitations, params_hash=params_hash,
            duration_ms=int((time.perf_counter() - t0) * 1000))
