"""The analysis card keeps independent cue meaning visible beyond truncated metrics."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.ui import strings
from imageguard.ui.widgets.verdict_panel import MethodCard


def test_card_explains_spatial_score_and_independent_burst(qtbot) -> None:
    evidence = {f"metric_{i}": i for i in range(8)}
    evidence.update({
        "cue_flag": False,
        "ref_status": "difference",
        "reference_cue_flag": True,
        "temporal_reason": "Vùng mặt đứng yên; độ tĩnh không cộng vào điểm.",
    })
    result = AlgorithmResult(
        method=MethodName.DEEPFAKE_CUES,
        verdict=Verdict.INCONCLUSIVE,
        score=0.2,
        score_definition="Điểm kết hợp các dấu hiệu xử lí ảnh",
        evidence=evidence,
    )
    card = MethodCard(result)
    qtbot.addWidget(card)
    text = "\n".join(label.text() for label in card.findChildren(QLabel))

    assert "0.200/1" in text
    assert "không phải xác suất thật/giả" in text
    assert strings.DEEPFAKE_CUE_FLAG_OFF in text
    assert strings.DEEPFAKE_REFERENCE_STATUS["difference"] in text
    assert evidence["temporal_reason"] in text


def test_unavailable_card_does_not_describe_clearance(qtbot) -> None:
    result = AlgorithmResult(
        method=MethodName.DEEPFAKE_CUES,
        verdict=Verdict.INCONCLUSIVE,
        evidence={"cue_unavailable": True, "cue_flag": False},
    )
    card = MethodCard(result)
    qtbot.addWidget(card)
    text = "\n".join(label.text() for label in card.findChildren(QLabel))

    assert strings.DEEPFAKE_CUE_UNAVAILABLE in text
    assert strings.DEEPFAKE_CUE_FLAG_OFF not in text
    assert "Điểm dấu hiệu:" not in text
