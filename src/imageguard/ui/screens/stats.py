"""Stats screen — đọc DB thật qua HistoryService.stats(), nhãn "thống kê phiên"."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from imageguard.services.history_service import HistoryService
from imageguard.ui import strings


class StatsScreen(QWidget):
    def __init__(self, history: HistoryService, parent=None) -> None:
        super().__init__(parent)
        self._svc = history
        lay = QVBoxLayout(self)
        self.lbl_head = QLabel("")
        self.lbl_head.setStyleSheet("font-size:16px;")
        lay.addWidget(self.lbl_head)
        self.lbl_label = QLabel("")  # "thống kê phiên — không phải benchmark"
        self.lbl_label.setStyleSheet("color:#666;")
        lay.addWidget(self.lbl_label)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Verdict", "Số lượng"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.table)
        self.btn = QPushButton("Làm mới")
        self.btn.clicked.connect(self.refresh)
        lay.addWidget(self.btn)
        self.refresh()

    def refresh(self) -> None:
        s = self._svc.stats()
        self.lbl_head.setText(
            f"{s['n_analyses']} phân tích · "
            f"avg duration: {s['avg_duration_ms'] or 0:.0f} ms")
        self.lbl_label.setText(s.get("label", ""))
        rows = s.get("by_verdict", {})
        self.table.setRowCount(len(rows))
        for i, (v, n) in enumerate(rows.items()):
            self.table.setItem(i, 0, QTableWidgetItem(
                strings.VERDICT_LABEL.get(v, v)))
            self.table.setItem(i, 1, QTableWidgetItem(str(n)))
