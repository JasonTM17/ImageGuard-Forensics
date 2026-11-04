"""History screen — bảng analyses từ SQLite + reopen + delete."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
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
from imageguard.ui.widgets.image_canvas import ImageCanvas, ndarray_to_qpixmap


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
        self.btn_reopen.clicked.connect(self._reopen)
        self.btn_export = QPushButton("Xuất báo cáo")
        self.btn_export.clicked.connect(self._export)
        self.btn_delete = QPushButton(strings.BTN_DELETE)
        self.btn_delete.clicked.connect(self._delete)
        btns.addWidget(self.btn_reopen)
        btns.addWidget(self.btn_export)
        btns.addWidget(self.btn_delete)
        btns.addStretch(1)
        lay.addLayout(btns)
        self.lbl_empty = QLabel(strings.EMPTY_HISTORY)
        lay.addWidget(self.lbl_empty)
        self.refresh()

    def refresh(self) -> None:
        try:
            rows = self._svc.list()
        except Exception as e:
            rows = []
            QMessageBox.warning(
                self, "Lỗi cơ sở dữ liệu",
                f"Không đọc được lịch sử phân tích: {type(e).__name__}: {e}")
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
            try:
                removed = self._svc.delete(aid)
            except Exception as e:
                QMessageBox.warning(
                    self, "Lỗi xoá",
                    f"Không xoá được bản ghi: {type(e).__name__}: {e}")
                return
            if removed is False:
                QMessageBox.information(
                    self, "Không tìm thấy bản ghi",
                    "Bản ghi đã bị xoá hoặc không tồn tại trong lịch sử.")
            self.refresh()

    def _reopen(self) -> None:
        aid = self._selected_id()
        if not aid:
            return
        try:
            d = self._svc.get(aid)
        except Exception as e:
            QMessageBox.warning(
                self, "Lỗi cơ sở dữ liệu",
                f"Không đọc được chi tiết phân tích: {type(e).__name__}: {e}")
            return
        if d is None:
            return
        dlg = AnalysisDetailDialog(d, self._svc, self)
        dlg.exec()

    def _export(self) -> None:
        aid = self._selected_id()
        if not aid:
            return
        from imageguard.services.report_service import ReportService
        default_name = f"report_{aid[:8]}.md"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Xuất báo cáo phân tích", default_name, "Markdown (*.md)"
        )
        if not save_path:
            return
        rep_svc = ReportService(self._svc.store)
        try:
            res = rep_svc.export_markdown(aid, Path(save_path))
        except Exception as e:
            QMessageBox.warning(self, "Xuất thất bại",
                                f"Không xuất được báo cáo:\n{type(e).__name__}: {e}")
            return
        if not res:
            QMessageBox.warning(self, "Xuất thất bại",
                                "Không đọc được report.json hoặc không ghi được file đích.")
            return
        QMessageBox.information(
            self, "Thành công", f"Đã xuất báo cáo phân tích thành công:\n{save_path}"
        )


class AnalysisDetailDialog(QDialog):
    """Xem lại phân tích đã lưu: tóm tắt methods + artifact images."""

    def __init__(self, detail: dict, svc: HistoryService, parent=None) -> None:
        super().__init__(parent)
        self._svc = svc
        self._aid = detail["id"]
        self._arts = detail["artifacts"]
        self.setWindowTitle(f"{detail['image_name']} — {detail['created_at'][:19]}")
        self.resize(900, 620)
        lay = QVBoxLayout(self)
        v = detail["verdict"]
        top_h = QHBoxLayout()
        top_h.addWidget(QLabel(
            f"<b>{strings.VERDICT_LABEL.get(v, v)}</b> · "
            f"{len(detail['methods'])} methods · {detail['duration_ms']} ms · "
            f"config `{detail['config_hash']}`"), 1)
        btn_exp = QPushButton("Xuất báo cáo")
        btn_exp.clicked.connect(self._export)
        top_h.addWidget(btn_exp)
        lay.addLayout(top_h)
        for m in detail["methods"]:
            score = (f" — score {m['score']:.4g} ({m['score_definition']})"
                     if m["score"] is not None else "")
            lbl = QLabel(f"• {m['method']}: "
                         f"{strings.VERDICT_LABEL.get(m['verdict'], m['verdict'])}"
                         f"{score}")
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
        self.canvas = ImageCanvas()
        self.combo = QComboBox()
        self.combo.addItem("Chọn artifact…", None)
        for a in self._arts:
            self.combo.addItem(f"{a['kind']} — {a['method']}", a["rel_path"])
        self.combo.currentIndexChanged.connect(self._show_art)
        self.lbl_art_err = QLabel("")
        self.lbl_art_err.setStyleSheet("color:#b91c1c;")
        self.lbl_art_err.setWordWrap(True)
        lay.addWidget(self.combo)
        lay.addWidget(self.lbl_art_err)
        lay.addWidget(self.canvas, 1)

    def _show_art(self) -> None:
        rel = self.combo.currentData()
        if not rel or not rel.lower().endswith((".png", ".jpg", ".bmp")):
            self.lbl_art_err.setText("")
            self.canvas.set_overlay(None)
            return
        try:
            data = self._svc.store.read(rel)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception as e:
            self.lbl_art_err.setText(
                f"Không hiển thị artifact '{rel}': {type(e).__name__}: {e}")
            return
        if img is None:
            self.lbl_art_err.setText(f"Artifact '{rel}' hỏng hoặc không đọc được.")
            return
        self.lbl_art_err.setText("")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self.canvas.set_image(img)

    def _export(self) -> None:
        from imageguard.services.report_service import ReportService
        default_name = f"report_{self._aid[:8]}.md"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Xuất báo cáo phân tích", default_name, "Markdown (*.md)"
        )
        if not save_path:
            return
        rep_svc = ReportService(self._svc.store)
        try:
            res = rep_svc.export_markdown(self._aid, Path(save_path))
        except Exception as e:
            QMessageBox.warning(self, "Xuất thất bại",
                                f"Không xuất được báo cáo:\n{type(e).__name__}: {e}")
            return
        if not res:
            QMessageBox.warning(self, "Xuất thất bại",
                                "Không đọc được report.json hoặc không ghi được file đích.")
            return
        QMessageBox.information(
            self, "Thành công", f"Đã xuất báo cáo phân tích thành công:\n{save_path}"
        )
