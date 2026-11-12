"""Block-based copy-move pipeline.

score = matched_area_fraction = diện tích mask / diện tích ảnh.
Verdict: frac ≥ min_area_frac → SUSPICIOUS; 0 < frac < ngưỡng → INCONCLUSIVE;
không có block đủ texture → INCONCLUSIVE; có block nhưng không shift nào qua
voting → NO_INDICATION.

Method này chậm O(H·W·B²) — chạy trên working copy cap block.max_side,
mask upscale về ảnh gốc. Không phát hiện được rotate/scale (DCT block
không bất biến hình học) — đó là trade-off đã biết của trường phái này.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.core.preprocess import to_gray
from imageguard.forensics.base import cancelled
from imageguard.forensics.block.dct_blocks import extract_features
from imageguard.forensics.block.voting import match_pairs, vote_mask
from imageguard.forensics.keypoint.mask_builder import upscale_mask

SCORE_DEF = "matched_area_fraction = diện tích mask / diện tích ảnh (shift-voted)"


class BlockCopyMove:
    """Detector copy-move kiểu Fridrich–Popescu: DCT block + shift voting.

    Method **conclusive**: có thể trả ``SUSPICIOUS_INDICATION`` khi
    ``matched_area_fraction ≥ min_area_frac`` (fusion đếm như detector chính).
    Không phát hiện xoay/scale — chỉ tịnh tiến trong cùng ảnh.
    """

    name = MethodName.COPY_MOVE_BLOCK.value

    def analyze(
        self,
        doc: ImageDocument,
        cfg: ThresholdSet,
        cancel: threading.Event | None = None,
    ) -> AlgorithmResult:
        """Chạy đủ bước: scale → DCT feature → match → vote → verdict.

        Verdict (đúng như mã): INCONCLUSIVE nếu không đủ block texture;
        SUSPICIOUS nếu diện tích mask đủ lớn; INCONCLUSIVE nếu mask yếu;
        NO_INDICATION nếu không có shift nào vượt voting.
        """
        t0 = time.perf_counter()
        limitations: list[str] = []

        max_side = int(cfg.value("block", "max_side"))
        gray = to_gray(doc.array)
        h, w = gray.shape
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            gray = cv2.resize(
                gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
            limitations.append(
                f"block analysis trên bản scale={scale:.2f} — vùng copy nhỏ hơn "
                "block có thể mất"
            )
        gh, gw = gray.shape
        B = int(cfg.value("block", "block_size"))
        if min(gh, gw) < B:
            return self._done(
                Verdict.INCONCLUSIVE, 0.0, t0,
                [f"working copy {gw}×{gh} < block {B}px — không trích được block"],
            )

        feats, pos = extract_features(
            gray, B, int(cfg.value("block", "n_coeffs")),
            flat_std_eps=float(cfg.value("block", "flat_std_eps")),
        )
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, 0.0, t0, ["cancelled after DCT"])

        pairs = match_pairs(
            feats, pos,
            float(cfg.value("block", "sim_thresh")),
            float(cfg.value("block", "min_block_dist")),
        )
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, 0.0, t0, ["cancelled after match"])

        mask, heat, stats = vote_mask(
            pairs, pos, B, gray.shape, int(cfg.value("block", "min_shift_votes"))
        )
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, 0.0, t0, ["cancelled after voting"])
        mask = upscale_mask(mask, scale, (doc.height, doc.width))
        frac = float(np.count_nonzero(mask)) / mask.size
        if scale < 1.0:
            heat = cv2.resize(heat, (doc.width, doc.height))

        min_frac = float(cfg.value("block", "min_area_frac"))
        if len(feats) == 0:
            verdict = Verdict.INCONCLUSIVE
            limitations.append(
                "không đủ bằng chứng: không có block DCT đủ texture để đối chiếu"
            )
        elif frac >= min_frac:
            verdict = Verdict.SUSPICIOUS_INDICATION
        elif frac > 0:
            verdict = Verdict.INCONCLUSIVE
            limitations.append(
                f"mask chỉ phủ {frac:.3%} < min_area_frac={min_frac} — tín hiệu yếu"
            )
        else:
            verdict = Verdict.NO_INDICATION

        limitations.append(
            "block-DCT chỉ phát hiện translate copy-move — rotate/scale/flip "
            "làm hệ số DCT đổi → không match (giới hạn trường phái)"
        )
        return self._done(
            verdict, frac, t0, limitations, mask=mask, heatmap=heat,
            evidence={
                "n_blocks": len(feats),
                "n_candidate_pairs": len(pairs),
                "n_shifts_kept": stats["n_shifts_kept"],
                "top_shift_votes": stats.get("top_shift_votes", 0),
                "work_size": [gw, gh],
            },
            params_hash=cfg.snapshot_hash(),
        )

    def _done(self, verdict, score, t0, limitations, evidence=None, mask=None,
              heatmap=None, params_hash=""):
        return AlgorithmResult(
            method=MethodName.COPY_MOVE_BLOCK,
            verdict=verdict,
            score=score,
            score_definition=SCORE_DEF,
            mask=mask,
            heatmap=heatmap,
            evidence=evidence or {},
            limitations=limitations,
            params_hash=params_hash,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
