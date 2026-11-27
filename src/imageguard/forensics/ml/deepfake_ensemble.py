"""Deepfake ensemble — một bộ nhận diện tổng hợp nhiều detector.

``deepfake_ensemble`` chạy đồng thời các detector deepfake đã có trong dự án
và gộp điểm theo **trung bình có trọng số minh bạch** (không phải hộp đen):

- ``deepfake_cues``  — bộ 6 trụ cột xử lí ảnh cổ điển (index 0–1), hoặc
  ``LearnedDeepfakeCues`` khi người dùng nạp mô hình DIP đã học có nhãn;
- ``deepfake_ml``    — GenConViT-ED ONNX, ``fake_frac`` 0–1 trên crop mặt;
- ``deepfake_screen``— CoC Xception ONNX, ``fake_prob`` 0–1 (domain screen-recapture).

Mọi điểm đều đã là hướng "fake" trong [0,1]. Detector không khả dụng cho ảnh
(nhận diện lỗi, không có mặt, thiếu model) bị loại khỏi trung bình và được
ghi rõ trong ``evidence.members``. Điểm tổng hợp vẫn là **chỉ số heuristic**,
không phải xác suất đã hiệu chuẩn; verdict theo ``fake_thresh``/``real_thresh``
của section ``deepfake_ensemble``. Cần tối thiểu ``min_members`` detector khả
dụng, nếu không verdict là INCONCLUSIVE (không đủ cơ sở); không có detector
nào chạy được thì NOT_APPLICABLE.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues
from imageguard.forensics.auxiliary.learned_deepfake import LearnedDeepfakeCues
from imageguard.forensics.base import cancelled
from imageguard.forensics.ml.deepfake_genconvit import DeepfakeGenConViT
from imageguard.forensics.ml.deepfake_screen_coc import DeepfakeScreenCoC

SCORE_DEF = (
    "trung bình có trọng số các điểm fake [0,1] của các detector deepfake khả dụng "
    "(cues/DIP học, GenConViT-ED, CoC Xception) — chỉ số heuristic tổng hợp, "
    "không phải xác suất đã hiệu chuẩn"
)

# key evidence → (weight key trong thresholds, factory member)
MEMBER_SPECS: tuple[tuple[str, str, Any], ...] = (
    ("deepfake_cues", "weight_deepfake_cues", "cues"),
    ("deepfake_ml", "weight_deepfake_ml", "ml"),
    ("deepfake_screen", "weight_deepfake_screen", "screen"),
)


def member_score(result: AlgorithmResult) -> float | None:
    """Điểm fake [0,1] của một member; None nếu member không dùng được cho ảnh này."""
    if result.verdict in {Verdict.ERROR, Verdict.NOT_APPLICABLE}:
        return None
    score = result.score
    if score is None or not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    score = float(score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def combine_members(
    members: dict[str, float],
    weights: dict[str, float],
    *,
    min_members: int,
    fake_thresh: float,
    real_thresh: float,
) -> dict[str, Any]:
    """Gộp điểm các member khả dụng — trung bình có trọng số, mọi trọng số đều ghi rõ trong evidence.

    Trả dict với ``combined`` (None nếu không có member), ``verdict``,
    ``agreement_fake_frac`` (tỉ lệ member có điểm ≥ 0.5) và ``weights_used``.
    """
    used = {key: float(weights[key]) for key in members if float(weights.get(key, 0.0)) > 0.0}
    total_w = sum(used.values())
    combined = (
        sum(members[key] * used[key] for key in used) / total_w if total_w > 0 else None
    )
    agreement = (
        sum(1.0 for s in members.values() if s >= 0.5) / len(members) if members else None
    )
    if combined is None:
        verdict = Verdict.NOT_APPLICABLE
    elif len(members) < min_members or total_w <= 0:
        verdict = Verdict.INCONCLUSIVE
    elif combined >= fake_thresh:
        verdict = Verdict.SUSPICIOUS_INDICATION
    elif combined <= real_thresh:
        verdict = Verdict.NO_INDICATION
    else:
        verdict = Verdict.INCONCLUSIVE
    return {
        "combined": combined,
        "verdict": verdict,
        "agreement_fake_frac": agreement,
        "members_available": len(members),
        "weights_used": used,
    }


class DeepfakeEnsemble:
    """Method registry entry — spec ``deepfake_ensemble``.

    ``model`` là DIP model JSON đã load/validate sẵn (opt-in); khi có, member
    ``deepfake_cues`` được thay bằng ``LearnedDeepfakeCues`` và dùng trọng số
    ``weight_learned_dip`` thay vì ``weight_deepfake_cues``.
    """

    name = "deepfake_ensemble"

    def __init__(self, model: dict | None = None) -> None:
        self.model = model

    def _build_members(self) -> list[tuple[str, str, Any]]:
        cues = (
            LearnedDeepfakeCues(self.model) if self.model is not None else DeepfakeCues()
        )
        factories = {"cues": cues, "ml": DeepfakeGenConViT(), "screen": DeepfakeScreenCoC()}
        if self.model is not None:
            return [
                ("learned_dip", "weight_learned_dip", factories["cues"]),
                ("deepfake_ml", "weight_deepfake_ml", factories["ml"]),
                ("deepfake_screen", "weight_deepfake_screen", factories["screen"]),
            ]
        return [(key, wkey, factories[factory]) for key, wkey, factory in MEMBER_SPECS]

    def analyze(
        self,
        doc: ImageDocument,
        cfg: ThresholdSet,
        cancel: threading.Event | None = None,
    ) -> AlgorithmResult:
        t0 = time.perf_counter()
        lim = [
            "điểm tổng hợp là trung bình có trọng số các chỉ số heuristic — không phải xác suất đã hiệu chuẩn",
            "mỗi detector có domain riêng (GenConViT đa miền, CoC chỉ screen-recapture); ensemble không cover generator mới",
            "SUSPICIOUS_INDICATION là dấu hiệu nghi vấn cần đối chiếu — không phải bằng chứng giả mạo",
        ]
        evidence: dict[str, Any] = {
            "variant": "deepfake_ensemble_v1",
            "learned_model": self.model is not None,
        }

        weights = {
            wkey: float(cfg.value("deepfake_ensemble", wkey))
            for _, wkey, _ in MEMBER_SPECS
        }
        weights["weight_learned_dip"] = float(cfg.value("deepfake_ensemble", "weight_learned_dip"))
        min_members = int(cfg.value("deepfake_ensemble", "min_members"))
        fake_thresh = float(cfg.value("deepfake_ensemble", "fake_thresh"))
        real_thresh = float(cfg.value("deepfake_ensemble", "real_thresh"))

        members_detail: dict[str, Any] = {}
        scores: dict[str, float] = {}
        mask: np.ndarray | None = None
        mask_from: str | None = None
        built = self._build_members()
        weight_by_member = {key: weights[wkey] for key, wkey, _ in built}
        for key, wkey, instance in built:
            if cancelled(cancel):
                members_detail[key] = {"available": False, "reason": "cancelled"}
                continue
            try:
                r = instance.analyze(doc, cfg, cancel)
            except Exception as exc:  # noqa: BLE001 - một member crash không được huỷ cả ensemble
                members_detail[key] = {
                    "available": False, "reason": f"error:{type(exc).__name__}",
                }
                continue
            score = member_score(r)
            entry: dict[str, Any] = {
                "available": score is not None,
                "verdict": r.verdict.value,
                "score": round(score, 4) if score is not None else None,
                "weight": weights.get(wkey, 0.0),
            }
            if score is None:
                entry["reason"] = (
                    r.verdict.value if r.verdict in {Verdict.ERROR, Verdict.NOT_APPLICABLE}
                    else "invalid_score"
                )
            else:
                scores[key] = score
                if r.mask is not None and mask is None:
                    mask = r.mask
                    mask_from = key
            members_detail[key] = entry

        combined = combine_members(
            scores, weight_by_member,
            min_members=min_members, fake_thresh=fake_thresh, real_thresh=real_thresh,
        )
        verdict = combined["verdict"]
        score = combined["combined"]

        evidence.update(
            members=members_detail,
            weights=weights,
            min_members=min_members,
            fake_thresh=fake_thresh,
            real_thresh=real_thresh,
            members_available=combined["members_available"],
            agreement_fake_frac=(
                round(combined["agreement_fake_frac"], 4)
                if combined["agreement_fake_frac"] is not None else None
            ),
            mask_from=mask_from,
        )
        if self.model is not None:
            evidence["model_sha256"] = self.model.get("model_sha256")

        if verdict is Verdict.NOT_APPLICABLE:
            lim.insert(0, "không detector deepfake nào chạy được trên ảnh này — ensemble không áp dụng")
        elif verdict is Verdict.INCONCLUSIVE and combined["members_available"] < min_members:
            lim.insert(
                0,
                f"chỉ {combined['members_available']}/{min_members} detector khả dụng — "
                "không đủ cơ sở kết luận",
            )
        elif verdict is Verdict.SUSPICIOUS_INDICATION:
            lim.insert(
                0,
                f"ensemble ghi nhận dấu hiệu deepfake tổng hợp mạnh "
                f"(combined={score:.2f} ≥ {fake_thresh:.2f}, agreement={combined['agreement_fake_frac']:.2f}) "
                "— cần đối chiếu chuyên gia",
            )
        elif verdict is Verdict.INCONCLUSIVE:
            lim.insert(
                0,
                f"điểm tổng hợp vùng xám (combined={score:.2f} trong "
                f"({real_thresh:.2f}, {fake_thresh:.2f})) — không đủ cơ sở kết luận",
            )

        return AlgorithmResult(
            method=MethodName.DEEPFAKE_ENSEMBLE,
            verdict=verdict,
            score=round(score, 4) if score is not None else None,
            score_definition=SCORE_DEF,
            mask=mask,
            evidence=evidence,
            limitations=lim,
            params_hash=cfg.snapshot_hash(),
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
