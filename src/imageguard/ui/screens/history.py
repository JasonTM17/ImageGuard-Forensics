"""History screen — bảng analyses từ SQLite + reopen + delete."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from imageguard.services.history_service import HistoryService
from imageguard.ui import strings


class HistoryScreen(QWidget):
    def __init__(self, history: HistoryService, parent=None) -> None:
        super().__init__(parent)
        self._svc = history
        lay = QVBoxLayout(self)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Thời gian", "Ảnh", "Verdict", "Methods", "ID"])
        self.table.setColumnHidden(4, True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        lay.addWidget(self.table)
        btns = QHBoxLayout()
        self.btn_reopen = QPushButton(strings.BTN_REOPEN)
        self.btn_delete = QPushButton(strings.BTN_DELETE)
        self.btn_delete.clicked.connect(self._delete)
        btns.addWidget(self.btn_reopen)
        btns.addWidget(self.btn_delete)
        btns.addStretch(1)
        lay.addLayout(btns)
        self.lbl_empty = QLabel(strings.EMPTY_HISTORY)
        lay.addWidget(self.lbl_empty)
        self.refresh()

    def refresh(self) -> None:
        rows = self._svc.list()
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(r["created_at"][:19]))
            self.table.setItem(i, 1, QTableWidgetItem(r["image_name"]))
            self.table.setItem(i, 2, QTableWidgetItem(
                strings.VERDICT_LABEL.get(r["verdict"], r["verdict"])))
            self.table.setItem(i, 3, QTableWidgetItem(str(r["n_methods"])))
            self.table.setItem(i, 4, QTableWidgetItem(r["id"]))
        self.lbl_empty.setVisible(not rows)

    def _selected_id(self) -> str | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 4)
        return item.text() if item else None

    def _delete(self) -> None:
        aid = self._selected_id()
        if not aid:
            return
        if QMessageBox.question(self, strings.BTN_DELETE,
                                strings.CONFIRM_DELETE) == QMessageBox.StandardButton.Yes:
            self._svc.delete(aid)
            self.refresh()
