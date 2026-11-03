"""Pipeline copy-move keypoint: detect → self-match → cluster+RANSAC → mask
→ verdict. Mỗi stage check cooperative cancel; mọi ngưỡng từ thresholds.yaml.

Verdict (định nghĩa trong thresholds.yaml, section keypoint):
  verified_pairs ≥ t_susp  → SUSPICIOUS_INDICATION
  verified_pairs < t_clean → NO_INDICATION
  giữa hai ngưỡng          → INCONCLUSIVE
  keypoints < k_min        → INCONCLUSIVE (insufficient texture — lý do ghi
                             trong limitations, đây là "không đủ bằng chứng",
                             KHÔNG phải "không có dấu hiệu").
"""

from __future__ import annotations

import threading
import time

import numpy as np

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.core.preprocess import preprocess_keypoint
from imageguard.forensics.base import cancelled
from imageguard.forensics.keypoint.detectors import create_backend
from imageguard.forensics.keypoint.geo_verify import geometric_verify
from imageguard.forensics.keypoint.mask_builder import build_mask, upscale_mask
from imageguard.forensics.keypoint.matchers import self_match
from imageguard.forensics.keypoint.viz import draw_pairs

SCORE_DEF = "số cặp keypoint match qua geometric verify (RANSAC affine inliers)"


class KeypointCopyMove:
    name = MethodName.COPY_MOVE_KEYPOINT.value

    def analyze(
        self,
        doc: ImageDocument,
        cfg: ThresholdSet,
        cancel: threading.Event | None = None,
    ) -> AlgorithmResult:
        t0 = time.perf_counter()
        limitations: list[str] = []

        gray = preprocess_keypoint(doc, cfg)
        if cancelled(cancel):
            return self._result(Verdict.INCONCLUSIVE, 0, t0, ["cancelled before detect"])

        backend = create_backend(cfg)
        kps, desc = backend.detect(gray)
        n_kp = len(kps)
        if cancelled(cancel):
            return self._result(Verdict.INCONCLUSIVE, 0, t0, ["cancelled after detect"])

        k_min = int(cfg.value("keypoint", "k_min"))
        if n_kp < k_min or desc is None:
            limitations.append(
                f"insufficient texture: {n_kp} keypoints < k_min={k_min} — "
                "ảnh quá phẳng/mờ để keypoint method kết luận"
            )
            return self._result(
                Verdict.INCONCLUSIVE, 0, t0, limitations,
                evidence={"n_keypoints": n_kp},
            )

        matches = self_match(kps, desc, cfg, backend.norm)
        if cancelled(cancel):
            return self._result(Verdict.INCONCLUSIVE, 0, t0, ["cancelled after match"])

        ver = geometric_verify(matches.pairs, kps, cfg)
        n_ver = len(ver.inlier_pairs)
        score = float(n_ver)

        t_susp = float(cfg.value("keypoint", "t_susp"))
        t_clean = float(cfg.value("keypoint", "t_clean"))
        if n_ver >= t_susp:
            verdict = Verdict.SUSPICIOUS_INDICATION
        elif n_ver < t_clean:
            verdict = Verdict.NO_INDICATION
        else:
            verdict = Verdict.INCONCLUSIVE
            limitations.append(
                f"verified pairs {n_ver} nằm giữa ngưỡng [{t_clean}, {t_susp}) — "
                "có tín hiệu nhưng chưa đủ mạnh để kết luận nghi vấn"
            )

        mask = heat = pairs_img = None
        if n_ver:
            mask, heat = build_mask(kps, matches.pairs, ver.inlier_pairs, gray.shape)
            pairs_img = draw_pairs(doc.array, kps, matches.pairs,
                                   ver.inlier_pairs, doc.scale_factor)
            mask = upscale_mask(mask, doc.scale_factor, (doc.height, doc.width))
            if doc.scale_factor < 1.0:
                heat_full = np.zeros((doc.height, doc.width), np.float32)
                import cv2
                heat = cv2.resize(heat, (doc.width, doc.height))
                heat = heat_full + heat
            coverage = float(np.count_nonzero(mask)) / mask.size
            if coverage > 0.5:
                limitations.append(
                    f"mask coverage {coverage:.1%} > 50% — nghi texture lặp tự nhiên "
                    "(gạch/cửa sổ) hơn copy-move; diễn giải thận trọng"
                )
                verdict = Verdict.INCONCLUSIVE
            limitations.append(
                "mask = union(source-side + dest-side) — keypoint không xác định "
                "được hướng copy (đâu là gốc, đâu là bản dán)"
            )
        if doc.resolution_reduced:
            limitations.append(
                f"phân tích trên working copy scale={doc.scale_factor:.2f} — "
                "vùng copy rất nhỏ có thể bị mất sau downscale"
            )

        return self._result(
            verdict, score, t0, limitations,
            mask=mask, heatmap=heat,
            evidence={
                "n_keypoints": n_kp,
                "n_pairs_raw": matches.stats.get("knn_raw", 0),
                "n_pairs_filtered": matches.stats.get("unique_pairs", 0),
                "n_verified": n_ver,
                "n_clusters": ver.n_clusters,
                "backend": str(cfg.value("keypoint", "backend")),
                "pairs_image": pairs_img,
            },
            params_hash=cfg.snapshot_hash(),
        )

    def _result(self, verdict, score, t0, limitations, evidence=None, mask=None,
                heatmap=None, params_hash=""):
        return AlgorithmResult(
            method=MethodName.COPY_MOVE_KEYPOINT,
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
