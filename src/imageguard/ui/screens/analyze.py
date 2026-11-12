"""Analyze screen — open ảnh → chọn methods → run/cancel → results inline.

Layout: trái = canvas preview + overlay toggle; phải = controls (file,
metadata, methods, progress, verdict banner + method cards khi xong).
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from imageguard.core.errors import ErrorResult
from imageguard.core.models import ForensicReport
from imageguard.services.analysis_service import AnalysisRequest, AnalysisService
from imageguard.ui import strings
from imageguard.ui.a11y import apply_a11y
from imageguard.ui.widgets.image_canvas import (
    ImageCanvas,
    heat_to_qpixmap,
    ndarray_to_qpixmap,
)
from imageguard.ui.widgets.verdict_panel import MethodCard, VerdictBanner
from imageguard.ui.workers import AnalysisWorker, make_pool


class AnalyzeScreen(QWidget):
    def __init__(self, service: AnalysisService, parent=None) -> None:
        super().__init__(parent)
        self._svc = service
        self._pool = make_pool()
        self._worker: AnalysisWorker | None = None
        self._cancel_event: threading.Event | None = None
        self._path: Path | None = None
        self._report: ForensicReport | None = None
        self._doc = None
        self._build()

    def _build(self) -> None:
        split = QSplitter(self)
        lay = QVBoxLayout(self)
        lay.addWidget(split)

        self.canvas = ImageCanvas()
        self.canvas.image_dropped.connect(self.load_path)
        split.addWidget(self.canvas)

        right = QWidget()
        rl = QVBoxLayout(right)
        self.btn_open = QPushButton(strings.BTN_OPEN)
        apply_a11y(self.btn_open, strings.A11Y_ANALYZE_OPEN)
        self.btn_open.clicked.connect(self._pick_file)
        rl.addWidget(self.btn_open)
        self.lbl_file = QLabel(strings.FMT_HINT)
        self.lbl_file.setWordWrap(True)
        rl.addWidget(self.lbl_file)
        self.lbl_meta = QLabel("")
        self.lbl_meta.setWordWrap(True)
        rl.addWidget(self.lbl_meta)

        methods_box = QGroupBox(strings.LBL_METHODS)
        methods_grid = QGridLayout(methods_box)
        self.chk_kp = QCheckBox(strings.METHOD_LABEL["copy_move_keypoint"])
        self.chk_kp.setChecked(True)
        self.chk_bl = QCheckBox(strings.METHOD_LABEL["copy_move_block"])
        self.chk_ela = QCheckBox(strings.METHOD_LABEL["ela"])
        self.chk_ela.setChecked(True)
        self.chk_noise = QCheckBox(strings.METHOD_LABEL["noise_inconsistency"])
        self.chk_nloc = QCheckBox(strings.METHOD_LABEL["noise_local"])
        self.chk_freq = QCheckBox(strings.METHOD_LABEL["frequency_analysis"])
        self.chk_spl = QCheckBox(strings.METHOD_LABEL["splicing_cues"])
        method_checks = (
            ("copy_move_keypoint", self.chk_kp),
            ("copy_move_block", self.chk_bl),
            ("ela", self.chk_ela),
            ("noise_inconsistency", self.chk_noise),
            ("noise_local", self.chk_nloc),
            ("frequency_analysis", self.chk_freq),
            ("splicing_cues", self.chk_spl),
        )
        for i, (method_key, c) in enumerate(method_checks):
            label = strings.METHOD_LABEL[method_key]
            a11y_name = strings.FMT_A11Y_ANALYZE_METHOD.format(
                group=strings.A11Y_ANALYZE_METHODS,
                label=label,
            )
            apply_a11y(c, a11y_name)
            methods_grid.addWidget(c, i // 2, i % 2)
        rl.addWidget(methods_box)

        self.btn_run = QPushButton(strings.BTN_RUN)
        apply_a11y(self.btn_run, strings.A11Y_ANALYZE_RUN)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton(strings.BTN_CANCEL)
        apply_a11y(self.btn_cancel, strings.A11Y_ANALYZE_CANCEL)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        rl.addWidget(self.btn_run)
        rl.addWidget(self.btn_cancel)
        self.progress = QProgressBar()
        apply_a11y(self.progress, strings.A11Y_ANALYZE_PROGRESS)
        self.progress.setRange(0, 100)
        rl.addWidget(self.progress)
        self.lbl_err = QLabel("")
        self.lbl_err.setStyleSheet("color:#b91c1c;")
        self.lbl_err.setWordWrap(True)
        rl.addWidget(self.lbl_err)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#4b5563;")
        rl.addWidget(self.lbl_status)

        self.banner = VerdictBanner()
        self.banner.hide()
        rl.addWidget(self.banner)
        self.overlay_box = QComboBox()
        apply_a11y(self.overlay_box, strings.A11Y_ANALYZE_OVERLAY)
        self.overlay_box.addItem(strings.OVERLAY_NONE, None)
        self.overlay_box.currentIndexChanged.connect(self._apply_overlay)
        self.overlay_box.hide()
        rl.addWidget(self.overlay_box)
        self.btn_export = QPushButton(strings.BTN_EXPORT_REPORT)
        apply_a11y(self.btn_export, strings.A11Y_ANALYZE_EXPORT)
        self.btn_export.clicked.connect(self._export_report)
        self.btn_export.hide()
        rl.addWidget(self.btn_export)

        self.cards_area = QScrollArea()
        apply_a11y(self.cards_area, strings.A11Y_ANALYZE_CARDS)
        self.cards_area.setMinimumHeight(180)
        self.cards_area.setWidgetResizable(True)
        self.cards_inner = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_inner)
        self.cards_area.setWidget(self.cards_inner)
        rl.addWidget(self.cards_area, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        self.setAcceptDrops(True)

    def _show_busy_open(self) -> None:
        self.lbl_status.setText(strings.MSG_BUSY_CANNOT_OPEN)

    # --- file ---
    def _pick_file(self) -> None:
        if self._worker is not None:
            self._show_busy_open()
            return  # không mở file mới khi analysis đang chạy
        f, _ = QFileDialog.getOpenFileName(
            self, strings.BTN_OPEN, "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
        if f:
            self.load_path(Path(f))

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        if self._worker is not None:
            self._show_busy_open()
            return  # không mở file mới khi analysis đang chạy
        urls = e.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.is_file():
                e.acceptProposedAction()
                self.load_path(path)

    def _clear_results(self) -> None:
        """Xoá toàn bộ UI kết quả cũ — banner/cards/overlay/export/report."""
        self._report = None
        self.banner.hide()
        self.overlay_box.blockSignals(True)
        self.overlay_box.clear()
        self.overlay_box.addItem(strings.OVERLAY_NONE, None)
        self.overlay_box.blockSignals(False)
        self.overlay_box.hide()
        self.btn_export.hide()
        for i in reversed(range(self.cards_layout.count())):
            item = self.cards_layout.itemAt(i)
            w = item.widget() if item else None
            self.cards_layout.removeItem(item)  # gỡ item (kể cả stretch)
            if w is not None:
                w.deleteLater()
        self.canvas.set_overlay(None)

    def _set_running(self, running: bool) -> None:
        self.btn_open.setEnabled(not running)
        self.btn_run.setEnabled(not running and self._path is not None)
        self.btn_cancel.setEnabled(running)
        if not running:
            self.progress.setFormat("%p%")
        for c in (self.chk_kp, self.chk_bl, self.chk_ela,
                  self.chk_noise, self.chk_nloc, self.chk_freq, self.chk_spl):
            c.setEnabled(not running)

    def load_path(self, path: Path) -> None:
        if self._worker is not None:  # đang phân tích: bỏ qua, tránh lệch ảnh/kết quả
            self._show_busy_open()
            return
        self._clear_results()
        self.lbl_err.setText("")
        self.lbl_status.setText("")
        self.lbl_file.setText(path.name)
        self.lbl_meta.setText("")
        # Preview nhanh qua safe_open (header probe + decode đã chặn bomb).
        from imageguard.core.config import load_thresholds
        from imageguard.core.io import safe_open

        out = safe_open(path, load_thresholds())
        if isinstance(out, ErrorResult):
            self._path = None
            self._doc = None
            self.canvas.clear_image()
            self.lbl_err.setText(f"{out.code.value}: {out.message}")
            self.btn_run.setEnabled(False)
            return
        self._path = path
        self._doc = out
        self.canvas.set_image(out.array)
        ex = out.exif.get("present", {})
        ex_txt = ", ".join(f"{k}={v}" for k, v in list(ex.items())[:3]) or "—"
        self.lbl_meta.setText(
            f"{out.width}×{out.height} · {out.image_format} · "
            f"{out.file_bytes // 1024} KB · EXIF: {ex_txt}")
        self.btn_run.setEnabled(True)

    # --- run ---
    def _methods(self) -> list[str]:
        m = []
        if self.chk_kp.isChecked():
            m.append("keypoint")
        if self.chk_bl.isChecked():
            m.append("block")
        if self.chk_ela.isChecked():
            m.append("ela")
        if self.chk_noise.isChecked():
            m.append("noise")
        if self.chk_nloc.isChecked():
            m.append("noise_local")
        if self.chk_freq.isChecked():
            m.append("freq")
        if self.chk_spl.isChecked():
            m.append("splicing")
        return m

    def _run(self) -> None:
        if not self._path or self._worker is not None:
            return
        methods = self._methods()
        if not methods:
            self.lbl_err.setText(strings.ERR_SELECT_METHOD)
            return
        self._clear_results()  # ẩn banner/export/cards cũ trước khi run mới
        self.lbl_err.setText("")
        self.progress.setValue(0)
        self._cancel_event = threading.Event()
        req = AnalysisRequest(
            path=self._path, methods=methods, cancel=self._cancel_event)
        self._worker = AnalysisWorker(self._svc, req)
        self._worker.signals.progress.connect(self._on_progress)
        self._worker.signals.result.connect(self._on_result)
        self._worker.signals.failed.connect(self._on_failed)
        self._set_running(True)
        self._pool.start(self._worker)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText(strings.STATUS_ANALYZE_CANCELLING)

    def _on_progress(self, frac: float, stage: str) -> None:
        self.progress.setValue(int(frac * 100))
        label_stage = strings.METHOD_LABEL.get(stage, stage)
        self.progress.setFormat(f"%p% — {label_stage}")

    def _on_failed(self, msg: str) -> None:
        self._worker = None
        self._set_running(False)
        self._clear_results()
        self.lbl_err.setText(msg)

    def _on_result(self, out) -> None:
        self._worker = None
        self._set_running(False)
        if isinstance(out, ErrorResult):
            self._clear_results()
            self.lbl_err.setText(f"{out.code.value}: {out.message}")
            return
        self._clear_results()
        self._report = out
        friendly_signals = [
            strings.METHOD_LABEL.get(m.method.value, m.method.value)
            for m in out.methods if m.verdict.value == "suspicious_indication"
        ]
        self.banner.set_verdict(out.overall_verdict, friendly_signals)
        self.banner.show()
        self.overlay_box.blockSignals(True)
        self.overlay_box.clear()
        self.overlay_box.addItem(strings.OVERLAY_NONE, None)
        for m in out.methods:
            self.cards_layout.addWidget(MethodCard(m))
            method_label = strings.METHOD_LABEL.get(
                m.method.value, m.method.value
            )
            if m.mask is not None:
                self.overlay_box.addItem(
                    strings.OVERLAY_MASK_FMT.format(label=method_label),
                    ("mask", m))
            if m.heatmap is not None:
                self.overlay_box.addItem(
                    strings.OVERLAY_HEATMAP_FMT.format(label=method_label),
                    ("heat", m))
            if m.evidence.get("pairs_image") is not None:
                self.overlay_box.addItem(
                    strings.OVERLAY_PAIRS_FMT.format(label=method_label),
                    ("pairs", m))
        self.cards_layout.addStretch(1)
        self.overlay_box.blockSignals(False)
        self.overlay_box.show()
        self.btn_export.show()
        self._apply_overlay()
        if self._cancel_event is not None and self._cancel_event.is_set():
            self.lbl_status.setText(strings.STATUS_ANALYZE_CANCELLED)

    def _export_report(self) -> None:
        if self._report is None:
            return
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        from imageguard.services.report_service import ReportService

        default_name = f"report_{self._path.stem if self._path else 'image'}_{self._report.report_id[:8]}.md"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Xuất báo cáo phân tích", default_name, "Markdown (*.md)"
        )
        if not save_path:
            return
        rep_svc = ReportService(self._svc.store)
        try:
            res = rep_svc.export_markdown(self._report.report_id, Path(save_path))
        except Exception as e:  # noqa: BLE001 - ranh giới UI: báo lỗi cho người dùng, không crash
            QMessageBox.warning(self, "Xuất thất bại",
                                f"Không xuất được báo cáo:\n{type(e).__name__}: {e}")
            return
        if not res:
            QMessageBox.warning(self, "Xuất thất bại",
                                "Không đọc được report.json hoặc không ghi được file đích. "
                                "Xem logs/artifacts của phiên phân tích.")
            return
        QMessageBox.information(
            self, "Thành công", f"Đã xuất báo cáo phân tích thành công:\n{save_path}"
        )

    def _apply_overlay(self) -> None:
        data = self.overlay_box.currentData()
        if data is None or self._doc is None:
            self.lbl_err.setText("")
            self.canvas.set_overlay(None)
            return
        kind, m = data
        shape = self._doc.array.shape[:2]
        if kind == "mask" and m.mask is not None:
            if m.mask.shape != shape:
                self.canvas.set_overlay(None)
                self.lbl_err.setText(strings.FMT_OVERLAY_SKIP_DIMS.format(
                    kind="mask", dims=m.mask.shape, shape=shape))
                return
            bgr = self._doc.array.copy()
            bgr[m.mask > 0] = (0, 0, 255)
            blend = cv2.addWeighted(self._doc.array, 0.6, bgr, 0.4, 0)
            self.canvas.set_overlay(ndarray_to_qpixmap(blend), alpha=1.0)
        elif kind == "heat" and m.heatmap is not None:
            if m.heatmap.shape[:2] != shape:
                self.canvas.set_overlay(None)
                self.lbl_err.setText(strings.FMT_OVERLAY_SKIP_DIMS.format(
                    kind="heatmap", dims=m.heatmap.shape[:2], shape=shape))
                return
            self.canvas.set_overlay(
                heat_to_qpixmap(m.heatmap, shape), alpha=0.55)
        elif kind == "pairs" and m.evidence.get("pairs_image") is not None:
            pi = m.evidence["pairs_image"]
            if pi.shape[:2] == shape:
                self.canvas.set_overlay(ndarray_to_qpixmap(pi), alpha=1.0)
            else:
                self.canvas.set_overlay(None)
                self.lbl_err.setText(strings.FMT_OVERLAY_SKIP_DIMS.format(
                    kind="pairs", dims=pi.shape[:2], shape=shape))
