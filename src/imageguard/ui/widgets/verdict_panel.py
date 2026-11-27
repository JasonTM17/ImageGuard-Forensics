"""Verdict banner + per-method evidence cards."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
)

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.ui import strings
from imageguard.ui.a11y import apply_a11y


class VerdictBanner(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(self)
        self._label = QLabel("—")
        self._label.setStyleSheet("font-size:20px; font-weight:bold;")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._label)
        self._sub = QLabel("")
        self._sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub.setWordWrap(True)
        lay.addWidget(self._sub)
        apply_a11y(self, strings.A11Y_VERDICT_BANNER)

    def set_verdict(self, verdict: Verdict, signals: list[str]) -> None:
        v = verdict.value
        self._label.setText(strings.VERDICT_LABEL.get(v, v))
        color = strings.VERDICT_COLOR.get(v, "#444")
        self.setStyleSheet(f"background:{color}22; border:2px solid {color};"
                           "border-radius:6px;")
        self._label.setStyleSheet(
            f"font-size:20px; font-weight:bold; color:{color};")
        self._sub.setText(
            strings.VERDICT_SIGNALS_PREFIX + ", ".join(signals) if signals else "")


class MethodCard(QFrame):
    """Evidence card cho một AlgorithmResult: verdict + score + definition +
    duration + chỉ số evidence + limitations — score luôn đi kèm định nghĩa."""

    def __init__(self, result: AlgorithmResult, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        title = strings.METHOD_LABEL.get(result.method.value, result.method.value)
        verdict_txt = strings.VERDICT_LABEL.get(result.verdict.value,
                                                result.verdict.value)
        head = QLabel(f"<b>{title}</b> — {verdict_txt} · {result.duration_ms} ms")
        head.setWordWrap(True)
        lay.addWidget(head)
        apply_a11y(self, strings.FMT_A11Y_METHOD_CARD.format(title=title))
        if result.score is not None:
            score_text = (
                strings.DEEPFAKE_PREVIEW_SCORE.format(score=result.score)
                if result.method is MethodName.DEEPFAKE_CUES
                else f"score = {result.score:.4g}"
            )
            score_lbl = QLabel(f"{score_text}\n({result.score_definition})")
            score_lbl.setWordWrap(True)
            lay.addWidget(score_lbl)
        if result.method is MethodName.DEEPFAKE_CUES:
            for text in _deepfake_context_lines(result.evidence):
                context_lbl = QLabel(text)
                context_lbl.setTextFormat(Qt.TextFormat.PlainText)
                context_lbl.setWordWrap(True)
                lay.addWidget(context_lbl)
        ev_items = _format_evidence_summary(result.evidence)
        if ev_items:
            ev_lbl = QLabel(f"Chỉ số: {ev_items}")
            ev_lbl.setWordWrap(True)
            ev_lbl.setStyleSheet("color:#1f2937; font-size:12px;")
            lay.addWidget(ev_lbl)
        for lim in result.limitations[:4]:
            lim_lbl = QLabel(f"• {lim}")
            lim_lbl.setWordWrap(True)
            lim_lbl.setStyleSheet("color:#555;")
            lay.addWidget(lim_lbl)


def _deepfake_context_lines(evidence: dict) -> list[str]:
    """Keep applicability and independent acquisition cues visible beyond first-N fields."""
    lines = []
    if evidence.get("cue_unavailable"):
        lines.append(strings.DEEPFAKE_CUE_UNAVAILABLE)
    elif "cue_flag" in evidence:
        lines.append(
            strings.DEEPFAKE_CUE_FLAG_ON if evidence["cue_flag"] else strings.DEEPFAKE_CUE_FLAG_OFF
        )
    ref_text = strings.DEEPFAKE_REFERENCE_STATUS.get(evidence.get("ref_status"))
    if ref_text:
        lines.append(ref_text)
    temporal_reason = evidence.get("temporal_reason")
    if isinstance(temporal_reason, str) and temporal_reason:
        lines.append(temporal_reason)
    return lines


def _format_evidence_summary(evidence: dict | None) -> str:
    """Tóm tắt tối đa 6 trường scalar từ ``evidence`` (bỏ ảnh/khóa nội bộ)."""
    if not evidence:
        return ""
    skip_keys = {"variant", "pairs_image", "block", "scale"}
    parts: list[str] = []
    for k, v in evidence.items():
        if k in skip_keys:
            continue
        if isinstance(v, (bool, int, str)):
            parts.append(f"{k}={v}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        elif isinstance(v, (list, tuple)) and len(v) <= 4 and all(
            isinstance(x, (int, float)) for x in v
        ):
            parts.append(f"{k}={list(v)}")
        if len(parts) >= 6:
            break
    return " · ".join(parts)
