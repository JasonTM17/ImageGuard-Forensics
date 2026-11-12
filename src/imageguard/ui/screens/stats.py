"""Stats screen — đọc DB thật qua HistoryService.stats(), nhãn "thống kê phiên"."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from imageguard.services.history_service import HistoryService
from imageguard.ui import strings
from imageguard.ui.a11y import apply_a11y
from imageguard.ui.screens.history import _table_cell


def _configure_stats_table(table: QTableWidget) -> None:
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)


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
        apply_a11y(self.table, strings.A11Y_STATS_TABLE)
        self.table.setHorizontalHeaderLabels([
            strings.STATS_COL_VERDICT,
            strings.STATS_COL_COUNT,
        ])
        _configure_stats_table(self.table)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.table)
        self.btn = QPushButton(strings.BTN_REFRESH)
        apply_a11y(self.btn, strings.A11Y_STATS_REFRESH)
        self.btn.clicked.connect(self.refresh)
        lay.addWidget(self.btn)
        self.refresh()

    def refresh(self) -> None:
        try:
            s = self._svc.stats()
        except Exception as e:  # noqa: BLE001 - ranh giới UI: báo lỗi cho người dùng, không crash
            QMessageBox.warning(
                self, "Lỗi cơ sở dữ liệu",
                f"Không đọc được thống kê phiên: {type(e).__name__}: {e}")
            return
        self.lbl_head.setText(strings.FMT_STATS_SUMMARY.format(
            n=s["n_analyses"],
            avg_ms=s["avg_duration_ms"] or 0,
        ))
        self.lbl_label.setText(s.get("label", ""))
        rows = s.get("by_verdict", {})
        self.table.setRowCount(len(rows))
        for i, (v, n) in enumerate(rows.items()):
            label = strings.VERDICT_LABEL.get(v, v)
            self.table.setItem(i, 0, _table_cell(label))
            self.table.setItem(i, 1, _table_cell(str(n)))
        self.table.resizeColumnsToContents()
