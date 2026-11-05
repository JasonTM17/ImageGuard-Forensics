"""Verdict banner + disclaimer bar + per-method evidence cards."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from imageguard.core.enums import Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.ui import strings


class DisclaimerBar(QFrame):
    """Disclaimer cố định — luôn hiển thị, không bao giờ ẩn được."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("background:#fef3c7; color:#713f12; padding:4px;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lbl = QLabel(strings.DISCLAIMER)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)


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

    def set_verdict(self, verdict: Verdict, signals: list[str]) -> None:
        v = verdict.value
        self._label.setText(strings.VERDICT_LABEL.get(v, v))
        color = strings.VERDICT_COLOR.get(v, "#444")
        self.setStyleSheet(f"background:{color}22; border:2px solid {color};"
                           "border-radius:6px;")
        self._label.setStyleSheet(
            f"font-size:20px; font-weight:bold; color:{color};")
        self._sub.setText(
            "Signals: " + ", ".join(signals) if signals else "")


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
        if result.score is not None:
            lay.addWidget(QLabel(f"score = {result.score:.4g}  "
                                 f"({result.score_definition})"))
        for lim in result.limitations[:4]:
            l = QLabel(f"• {lim}")
            l.setWordWrap(True)
            l.setStyleSheet("color:#555;")
            lay.addWidget(l)
