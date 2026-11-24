"""Analyze screen — open ảnh → chọn methods → run/cancel → results inline.

Layout: trái = canvas preview + overlay toggle; phải = controls (file,
metadata, methods, progress, verdict banner + method cards khi xong).
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
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
from imageguard.forensics.localize.heatmap import overlay_mask
from imageguard.ingest.session_reference import SessionReferenceStore
from imageguard.services.analysis_service import AnalysisRequest, AnalysisService
from imageguard.ui import strings
from imageguard.ui.a11y import apply_a11y
from imageguard.ui.widgets.camera_dialog import CameraCaptureDialog
from imageguard.ui.widgets.image_canvas import (
    ImageCanvas,
    heat_to_qpixmap,
    ndarray_to_qpixmap,
)
from imageguard.ui.widgets.sample_dialog import SampleGeneratorDialog
from imageguard.ui.widgets.verdict_panel import MethodCard, VerdictBanner
from imageguard.ui.workers import AnalysisWorker, make_pool


class AnalyzeScreen(QWidget):
    send_to_enhance = Signal(Path)

    def __init__(self, service: AnalysisService, parent=None) -> None:
        super().__init__(parent)
        self._svc = service
        self._pool = make_pool()
        self._worker: AnalysisWorker | None = None
        self._cancel_event: threading.Event | None = None
        self._path: Path | None = None
        self._report: ForensicReport | None = None
        self._doc = None
        self._ref_store = SessionReferenceStore()
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
        file_row = QHBoxLayout()
        self.btn_open = QPushButton(strings.BTN_OPEN)
        apply_a11y(self.btn_open, strings.A11Y_ANALYZE_OPEN)
        self.btn_open.clicked.connect(self._pick_file)
        self.btn_camera = QPushButton(strings.CAMERA_BTN_OPEN)
        apply_a11y(self.btn_camera, strings.A11Y_CAMERA_OPEN)
        self.btn_camera.clicked.connect(self._capture_camera)
        file_row.addWidget(self.btn_open, 1)
        file_row.addWidget(self.btn_camera, 1)
        rl.addLayout(file_row)

        self.btn_demo_sample = QPushButton(strings.BTN_DEMO_SAMPLE)
        apply_a11y(self.btn_demo_sample, strings.A11Y_DEMO_SAMPLE)
        self.btn_demo_sample.clicked.connect(self._open_demo_sample)
        rl.addWidget(self.btn_demo_sample)

        ref_row = QHBoxLayout()
        self.btn_set_ref = QPushButton(strings.ANALYZE_BTN_SET_REFERENCE)
        apply_a11y(self.btn_set_ref, strings.ANALYZE_BTN_SET_REFERENCE)
        self.btn_set_ref.clicked.connect(self._set_reference)
        self.btn_clear_ref = QPushButton(strings.ANALYZE_BTN_CLEAR_REFERENCE)
        apply_a11y(self.btn_clear_ref, strings.ANALYZE_BTN_CLEAR_REFERENCE)
        self.btn_clear_ref.clicked.connect(self._clear_reference)
        ref_row.addWidget(self.btn_set_ref, 1)
        ref_row.addWidget(self.btn_clear_ref, 1)
        rl.addLayout(ref_row)
        self.lbl_ref = QLabel("")
        self.lbl_ref.setWordWrap(True)
        rl.addWidget(self.lbl_ref)

        self.lbl_file = QLabel(strings.FMT_HINT)
        self.lbl_file.setWordWrap(True)
        rl.addWidget(self.lbl_file)
        self.lbl_meta = QLabel("")
        self.lbl_meta.setWordWrap(True)
        rl.addWidget(self.lbl_meta)

        methods_box = QGroupBox(strings.LBL_METHODS)
        methods_outer = QVBoxLayout(methods_box)
        preset_row = QHBoxLayout()
        self.btn_preset_default = QPushButton(strings.BTN_PRESET_DEFAULT)
        self.btn_preset_camera = QPushButton(strings.BTN_PRESET_CAMERA)
        self.btn_preset_all = QPushButton(strings.BTN_PRESET_ALL)
        self.btn_preset_default.clicked.connect(
            lambda: self._apply_method_preset(["keypoint", "block", "ela"])
        )
        self.btn_preset_camera.clicked.connect(
            lambda: self._apply_method_preset(["ela", "noise_local", "freq", "deepfake"])
        )
        self.btn_preset_all.clicked.connect(
            lambda: self._apply_method_preset(
                ["keypoint", "block", "ela", "noise", "noise_local", "freq", "splicing", "deepfake"]
            )
        )
        preset_row.addWidget(self.btn_preset_default)
        preset_row.addWidget(self.btn_preset_camera)
        preset_row.addWidget(self.btn_preset_all)
        methods_outer.addLayout(preset_row)

        methods_grid = QGridLayout()
        self.chk_kp = QCheckBox(strings.METHOD_LABEL["copy_move_keypoint"])
        self.chk_kp.setChecked(True)
        self.chk_bl = QCheckBox(strings.METHOD_LABEL["copy_move_block"])
        self.chk_ela = QCheckBox(strings.METHOD_LABEL["ela"])
        self.chk_ela.setChecked(True)
        self.chk_noise = QCheckBox(strings.METHOD_LABEL["noise_inconsistency"])
        self.chk_nloc = QCheckBox(strings.METHOD_LABEL["noise_local"])
        self.chk_freq = QCheckBox(strings.METHOD_LABEL["frequency_analysis"])
        self.chk_spl = QCheckBox(strings.METHOD_LABEL["splicing_cues"])
        self.chk_df = QCheckBox(strings.METHOD_LABEL["deepfake_cues"])
        self.chk_df.setChecked(False)
        method_checks = (
            ("copy_move_keypoint", self.chk_kp),
            ("copy_move_block", self.chk_bl),
            ("ela", self.chk_ela),
            ("noise_inconsistency", self.chk_noise),
            ("noise_local", self.chk_nloc),
            ("frequency_analysis", self.chk_freq),
            ("splicing_cues", self.chk_spl),
            ("deepfake_cues", self.chk_df),
        )
        for i, (method_key, c) in enumerate(method_checks):
            label = strings.METHOD_LABEL[method_key]
            a11y_name = strings.FMT_A11Y_ANALYZE_METHOD.format(
                group=strings.A11Y_ANALYZE_METHODS,
                label=label,
            )
            apply_a11y(c, a11y_name)
            methods_grid.addWidget(c, i // 2, i % 2)
        methods_outer.addLayout(methods_grid)
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

        self.btn_send_enhance = QPushButton(strings.BTN_SEND_TO_ENHANCE)
        self.btn_send_enhance.setEnabled(False)
        self.btn_send_enhance.clicked.connect(self._emit_send_to_enhance)
        rl.addWidget(self.btn_send_enhance)

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

        self._refresh_ref_status()
        self.setAcceptDrops(True)

    # --- session reference ---
    def _refresh_ref_status(self) -> None:
        has_ref = self._ref_store.get_reference() is not None
        self.lbl_ref.setText(
            strings.ANALYZE_REFERENCE_READY
            if has_ref
            else strings.ANALYZE_REFERENCE_MISSING
        )
        idle = self._worker is None
        self.btn_set_ref.setEnabled(idle and self._path is not None)
        self.btn_clear_ref.setEnabled(idle and has_ref)

    def _set_reference(self) -> None:
        if self._worker is not None or self._path is None:
            return
        try:
            self._ref_store.set_reference(self._path)
        except Exception as e:  # noqa: BLE001 - ranh giới UI: báo lỗi ẩn path, không crash
            self.lbl_err.setText(
                f"{strings.MSG_EXCEPTION_PATH_REDACTED} ({type(e).__name__})"
            )
        self._refresh_ref_status()

    def _clear_reference(self) -> None:
        if self._worker is not None:
            return
        try:
            self._ref_store.clear()
        except Exception as e:  # noqa: BLE001
            self.lbl_err.setText(
                f"{strings.MSG_EXCEPTION_PATH_REDACTED} ({type(e).__name__})"
            )
        self._refresh_ref_status()

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

    def _capture_camera(self) -> None:
        if self._worker is not None:
            self._show_busy_open()
            return
        dlg = CameraCaptureDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.captured_path is not None:
            # Nguyên tắc: khung hình gốc từ camera tự động kích hoạt nhánh kiểm tra Deepfake trước.
            self.chk_df.setChecked(True)
            self.load_path(dlg.captured_path)
            if self._path is not None and self._ref_store.get_reference() is None:
                self.lbl_ref.setText(strings.ANALYZE_CAMERA_NEXT_STEP)

    def _open_demo_sample(self) -> None:
        if self._worker is not None:
            self._show_busy_open()
            return
        dlg = SampleGeneratorDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.generated_path is not None:
            self.load_path(dlg.generated_path)
            if dlg.recommended_methods:
                self._apply_method_preset(dlg.recommended_methods)

    def _apply_method_preset(self, method_specs: list[str]) -> None:
        selected = set(method_specs)
        mapping = {
            "keypoint": self.chk_kp,
            "block": self.chk_bl,
            "ela": self.chk_ela,
            "noise": self.chk_noise,
            "noise_local": self.chk_nloc,
            "freq": self.chk_freq,
            "splicing": self.chk_spl,
            "deepfake": self.chk_df,
        }
        for spec, chk in mapping.items():
            chk.setChecked(spec in selected)

    def _emit_send_to_enhance(self) -> None:
        if self._path is not None and self._worker is None:
            self.send_to_enhance.emit(self._path)

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
        self.btn_camera.setEnabled(not running)
        self.btn_demo_sample.setEnabled(not running)
        self.btn_preset_default.setEnabled(not running)
        self.btn_preset_camera.setEnabled(not running)
        self.btn_preset_all.setEnabled(not running)
        self.btn_run.setEnabled(not running and self._path is not None)
        self.btn_send_enhance.setEnabled(not running and self._path is not None)
        self.btn_cancel.setEnabled(running)
        self._refresh_ref_status()
        if not running:
            self.progress.setFormat("%p%")
        for c in (self.chk_kp, self.chk_bl, self.chk_ela,
                  self.chk_noise, self.chk_nloc, self.chk_freq,
                  self.chk_spl, self.chk_df):
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
            self.btn_send_enhance.setEnabled(False)
            self._refresh_ref_status()
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
        self.btn_send_enhance.setEnabled(True)
        self._refresh_ref_status()

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
        if self.chk_df.isChecked():
            m.append("deepfake")
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
            path=self._path, methods=methods, cancel=self._cancel_event,
            reference_path=self._ref_store.get_reference())
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
        doc = self._doc
        if doc is None or getattr(out, "image_sha256", None) != doc.sha256:
            # File đã đổi kể từ preview — không ghép report của ảnh khác lên canvas cũ.
            self._clear_results()
            self._doc = None
            self._path = None
            self.canvas.clear_image()
            self.lbl_meta.setText("")
            self.lbl_err.setText(strings.ERR_SOURCE_CHANGED_REOPEN)
            self.btn_run.setEnabled(False)
            self.btn_send_enhance.setEnabled(False)
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
            if (
                m.mask.shape[:2] != shape
                or m.mask.ndim not in (2, 3)
                or (m.mask.ndim == 3 and m.mask.shape[2] != 1)
            ):
                self.canvas.set_overlay(None)
                self.lbl_err.setText(strings.FMT_OVERLAY_SKIP_DIMS.format(
                    kind="mask", dims=m.mask.shape, shape=shape))
                return
            try:
                blend = overlay_mask(
                    self._doc.array, m.mask, color=(0, 0, 255), alpha=0.4
                )
                pix = ndarray_to_qpixmap(blend)
            except ValueError:
                self.canvas.set_overlay(None)
                self.lbl_err.setText(strings.FMT_OVERLAY_SKIP_DIMS.format(
                    kind="mask", dims=m.mask.shape, shape=shape))
                return
            self.lbl_err.setText("")
            self.canvas.set_overlay(pix, alpha=1.0)
        elif kind == "heat" and m.heatmap is not None:
            if (
                m.heatmap.shape[:2] != shape
                or m.heatmap.ndim not in (2, 3)
                or (m.heatmap.ndim == 3 and m.heatmap.shape[2] <= 0)
            ):
                self.canvas.set_overlay(None)
                self.lbl_err.setText(strings.FMT_OVERLAY_SKIP_DIMS.format(
                    kind="heatmap", dims=m.heatmap.shape, shape=shape))
                return
            try:
                pix = heat_to_qpixmap(m.heatmap, shape)
            except ValueError:
                self.canvas.set_overlay(None)
                self.lbl_err.setText(strings.FMT_OVERLAY_SKIP_DIMS.format(
                    kind="heatmap", dims=m.heatmap.shape, shape=shape))
                return
            self.lbl_err.setText("")
            self.canvas.set_overlay(pix, alpha=0.55)
        elif kind == "pairs" and m.evidence.get("pairs_image") is not None:
            pi = m.evidence["pairs_image"]
            if (
                pi.shape[:2] != shape
                or pi.ndim not in (2, 3)
                or (pi.ndim == 3 and pi.shape[2] not in (1, 3, 4))
            ):
                self.canvas.set_overlay(None)
                self.lbl_err.setText(strings.FMT_OVERLAY_SKIP_DIMS.format(
                    kind="pairs", dims=pi.shape, shape=shape))
                return
            try:
                pix = ndarray_to_qpixmap(pi)
            except ValueError:
                self.canvas.set_overlay(None)
                self.lbl_err.setText(strings.FMT_OVERLAY_SKIP_DIMS.format(
                    kind="pairs", dims=pi.shape, shape=shape))
                return
            self.lbl_err.setText("")
            self.canvas.set_overlay(pix, alpha=1.0)
        else:
            self.lbl_err.setText("")
            self.canvas.set_overlay(None)
