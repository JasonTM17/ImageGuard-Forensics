"""Analyze screen — open ảnh → chọn methods → run/cancel → results inline.

Layout: trái = canvas preview + overlay toggle; phải = controls (file,
metadata, methods, progress, verdict banner + method cards khi xong).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QListWidget,
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
        self._path: Path | None = None
        self._report: ForensicReport | None = None
        self._doc = None
        self._build()

    def _build(self) -> None:
        split = QSplitter(self)
        lay = QVBoxLayout(self)
        lay.addWidget(split)

        self.canvas = ImageCanvas()
        split.addWidget(self.canvas)

        right = QWidget()
        rl = QVBoxLayout(right)
        self.btn_open = QPushButton(strings.BTN_OPEN)
        self.btn_open.clicked.connect(self._pick_file)
        rl.addWidget(self.btn_open)
        self.lbl_file = QLabel(strings.FMT_HINT)
        self.lbl_file.setWordWrap(True)
        rl.addWidget(self.lbl_file)
        self.lbl_meta = QLabel("")
        self.lbl_meta.setWordWrap(True)
        rl.addWidget(self.lbl_meta)

        rl.addWidget(QLabel("Phương pháp:"))
        self.chk_kp = QCheckBox(strings.METHOD_LABEL["copy_move_keypoint"])
        self.chk_kp.setChecked(True)
        self.chk_bl = QCheckBox(strings.METHOD_LABEL["copy_move_block"])
        self.chk_ela = QCheckBox(strings.METHOD_LABEL["ela"])
        self.chk_ela.setChecked(True)
        for c in (self.chk_kp, self.chk_bl, self.chk_ela):
            rl.addWidget(c)

        self.btn_run = QPushButton(strings.BTN_RUN)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton(strings.BTN_CANCEL)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        rl.addWidget(self.btn_run)
        rl.addWidget(self.btn_cancel)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        rl.addWidget(self.progress)
        self.lbl_err = QLabel("")
        self.lbl_err.setStyleSheet("color:#b91c1c;")
        self.lbl_err.setWordWrap(True)
        rl.addWidget(self.lbl_err)

        self.banner = VerdictBanner()
        self.banner.hide()
        rl.addWidget(self.banner)
        self.overlay_box = QComboBox()
        self.overlay_box.addItem("Không overlay", None)
        self.overlay_box.currentIndexChanged.connect(self._apply_overlay)
        self.overlay_box.hide()
        rl.addWidget(self.overlay_box)

        self.cards_area = QScrollArea()
        self.cards_area.setWidgetResizable(True)
        self.cards_inner = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_inner)
        self.cards_area.setWidget(self.cards_inner)
        rl.addWidget(self.cards_area, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        self.setAcceptDrops(True)

    # --- file ---
    def _pick_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, strings.BTN_OPEN, "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
        if f:
            self.load_path(Path(f))

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        urls = e.mimeData().urls()
        if urls:
            self.load_path(Path(urls[0].toLocalFile()))

    def load_path(self, path: Path) -> None:
        self._path = path
        self.lbl_err.setText("")
        self.lbl_file.setText(path.name)
        # Preview nhanh qua safe_open (header probe + decode đã chặn bomb).
        from imageguard.core.config import load_thresholds
        from imageguard.core.io import safe_open

        out = safe_open(path, load_thresholds())
        if isinstance(out, ErrorResult):
            self.lbl_err.setText(f"{out.code.value}: {out.message}")
            self.btn_run.setEnabled(False)
            return
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
        return m

    def _run(self) -> None:
        if not self._path or self._worker is not None:
            return
        self.lbl_err.setText("")
        req = AnalysisRequest(path=self._path, methods=self._methods())
        self._worker = AnalysisWorker(self._svc, req)
        self._worker.signals.progress.connect(self._on_progress)
        self._worker.signals.result.connect(self._on_result)
        self._worker.signals.failed.connect(self._on_failed)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._pool.start(self._worker)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.btn_cancel.setEnabled(False)

    def _on_progress(self, frac: float, stage: str) -> None:
        self.progress.setValue(int(frac * 100))

    def _on_failed(self, msg: str) -> None:
        self._worker = None
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_err.setText(msg)

    def _on_result(self, out) -> None:
        self._worker = None
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if isinstance(out, ErrorResult):
            self.lbl_err.setText(f"{out.code.value}: {out.message}")
            return
        self._report = out
        self.banner.set_verdict(out.overall_verdict,
                                [m.method.value for m in out.methods
                                 if m.verdict.value == "suspicious_indication"])
        self.banner.show()
        for i in reversed(range(self.cards_layout.count())):
            w = self.cards_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self.overlay_box.blockSignals(True)
        self.overlay_box.clear()
        self.overlay_box.addItem("Không overlay", None)
        for m in out.methods:
            self.cards_layout.addWidget(MethodCard(m))
            if m.mask is not None:
                self.overlay_box.addItem(f"Mask — {m.method.value}",
                                       ("mask", m))
            if m.heatmap is not None:
                self.overlay_box.addItem(f"Heatmap — {m.method.value}",
                                       ("heat", m))
        self.cards_layout.addStretch(1)
        self.overlay_box.blockSignals(False)
        self.overlay_box.show()
        self._apply_overlay()

    def _apply_overlay(self) -> None:
        data = self.overlay_box.currentData()
        if data is None or self._doc is None:
            self.canvas.set_overlay(None)
            return
        kind, m = data
        if kind == "mask" and m.mask is not None:
            bgr = self._doc.array.copy()
            bgr[m.mask > 0] = (0, 0, 255)
            blend = (self._doc.array * 0.6 + bgr * 0.4).astype(np.uint8)
            self.canvas.set_overlay(ndarray_to_qpixmap(blend), alpha=1.0)
        elif kind == "heat" and m.heatmap is not None:
            self.canvas.set_overlay(
                heat_to_qpixmap(m.heatmap, self._doc.array.shape[:2]), alpha=0.55)
