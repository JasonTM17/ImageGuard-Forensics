"""Opt-in, locally trained linear decision on the existing DIP measurements.

The face/quality gates remain mandatory. The learned index is a sigmoid of a
linear margin, not a calibrated probability or a manipulation localization.
"""

from __future__ import annotations

import math
import threading
from dataclasses import replace

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues
from imageguard.forensics.auxiliary.deepfake_model import (
    config_digest,
    decision_function,
    features_from_evidence,
    validate_model,
)

SCORE_DEFINITION = (
    "chỉ số sigmoid của biên quyết định logistic trên 13 đặc trưng xử lý ảnh; "
    "trọng số học từ tập dev có nhãn, ngưỡng cố định 0.5 — "
    "chưa hiệu chuẩn xác suất, không phải xác suất Deepfake"
)


class LearnedDeepfakeCues:
    """Use an explicitly selected model; never silently substitute bad weights."""

    name = "deepfake_cues"

    def __init__(self, model: dict) -> None:
        self.model = validate_model(model)

    def analyze(
        self, doc: ImageDocument, cfg: ThresholdSet,
        cancel: threading.Event | None = None,
    ) -> AlgorithmResult:
        model = validate_model(self.model, expected_config_sha256=config_digest(cfg.raw))
        base = DeepfakeCues().analyze(doc, cfg, cancel)
        evidence = {
            **base.evidence,
            "decision_source": "learned_dip",
            "model_sha256": model["model_sha256"],
            "feature_schema": model["feature_schema"],
            "heuristic_score": base.score,
            "heuristic_cue_flag": base.evidence.get("cue_flag", False),
            "model_status": "unavailable",
        }
        if base.evidence.get("cue_unavailable") or base.score is None:
            return replace(base, evidence=evidence)
        if doc.capture_layout != "single":
            evidence.update(cue_unavailable=True, cue_flag=False,
                            unavailable_reason="unsupported_capture_layout")
            return replace(
                base, verdict=Verdict.INCONCLUSIVE, score=None, mask=None, heatmap=None,
                evidence=evidence,
                limitations=["mô hình đã học chỉ hỗ trợ ảnh đơn; ảnh ghép camera chưa được đánh giá"]
                + base.limitations,
            )

        features = features_from_evidence(base.evidence)
        margin = decision_function(model, features)
        index = (1.0 / (1.0 + math.exp(-margin)) if margin >= 0
                 else math.exp(margin) / (1.0 + math.exp(margin)))
        flagged = margin >= 0.0
        seen = any(sample.get("sha256") == doc.sha256
                   for sample in model["training"]["samples"])
        evidence.update(
            model_status="applied", cue_flag=flagged, model_decision_score=margin,
            model_in_training_data=seen, decision_threshold=0.5,
            decision_rule="sigmoid(linear_margin) >= 0.5",
            calibration_status="not_probability_calibrated", score_is_probability=False,
            heatmap_scope="heuristic_texture_diagnostic_only",
        )
        limitations = [
            "đang dùng trọng số học từ dữ liệu dev; chất lượng phụ thuộc nguồn ảnh và nhãn huấn luyện",
            "chỉ số mô hình không phải xác suất; cần đối chiếu với tập test tách theo nguồn",
            "heatmap chỉ minh họa texture của nhánh xử lý ảnh; mô hình này không định vị vùng giả mạo",
        ]
        if seen:
            limitations.insert(0, "ảnh này có trong dữ liệu phát triển của mô hình; không dùng kết quả để báo cáo test")
        limitations.extend(lim for lim in base.limitations
                           if not lim.startswith("phát hiện bất thường mịn/LBP/phổ"))
        return replace(
            base, verdict=Verdict.INCONCLUSIVE if flagged else Verdict.NO_INDICATION,
            score=index, score_definition=SCORE_DEFINITION, mask=None,
            evidence=evidence, limitations=limitations,
        )
