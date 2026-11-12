"""Verdict banner + per-method evidence cards."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
)

from imageguard.core.enums import Verdict
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
    limitations — score luôn đi kèm định nghĩa."""

    def __init__(self, result: AlgorithmResult, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        title = strings.METHOD_LABEL.get(result.method.value, result.method.value)
        verdict_txt = strings.VERDICT_LABEL.get(result.verdict.value,
                                                result.verdict.value)
        head = QLabel(f"<b>{title}</b> — {verdict_txt}")
        head.setWordWrap(True)
        lay.addWidget(head)
        apply_a11y(self, strings.FMT_A11Y_METHOD_CARD.format(title=title))
        if result.score is not None:
            score_lbl = QLabel(f"score = {result.score:.4g}  "
                               f"({result.score_definition})")
            score_lbl.setWordWrap(True)
            lay.addWidget(score_lbl)
        for lim in result.limitations[:4]:
            lim_lbl = QLabel(f"• {lim}")
            lim_lbl.setWordWrap(True)
            lim_lbl.setStyleSheet("color:#555;")
            lay.addWidget(lim_lbl)
