"""Enhance screen — xem ảnh gốc/ảnh tăng cường, KHÔNG phân tích.

Tab này chạy operator cổ điển hoặc NAFNet để xem thử. Không chạy dịch vụ phân
tích, không ghi database, không kết luận điều gì. Ảnh gốc và bytes của nó
không bao giờ bị ghi đè.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from imageguard.core.errors import ErrorResult
from imageguard.evaluation.enhance_study import RESTORER_PARAMS
from imageguard.ui import strings
from imageguard.ui.a11y import apply_a11y
from imageguard.ui.widgets.image_canvas import ImageCanvas, ndarray_to_qpixmap
from imageguard.ui.workers import EnhanceWorker, make_pool

_OPERATORS = (
    ("unsharp", "Unsharp mask"),
    ("clahe", "CLAHE"),
    ("nlm", "Non-local means"),
    ("nafnet", "NAFNet (AI)"),
)

_UNSHARP = RESTORER_PARAMS["unsharp"]
_CLAHE = RESTORER_PARAMS["clahe"]
_NLM = RESTORER_PARAMS["nlm"]

# (name, nhãn, loại, min, max, bước, mặc định) — mặc định lấy từ RESTORER_PARAMS.
_FIELD_SPECS = {
    "unsharp": (
        ("ksize", "Kích thước kernel", "int", 1, 99, 2, _UNSHARP["ksize"]),
        ("sigma", "Sigma", "float", 0.1, 20.0, 0.1, _UNSHARP["sigma"]),
        ("amount", "Độ mạnh", "float", 0.0, 5.0, 0.1, _UNSHARP["amount"]),
    ),
    "clahe": (
        ("clip_limit", "Clip limit", "float", 0.0, 10.0, 0.1,
         _CLAHE["clip_limit"]),
        ("tile_x", "Tile ngang", "int", 1, 64, 1, _CLAHE["tile_grid_size"][0]),
        ("tile_y", "Tile dọc", "int", 1, 64, 1, _CLAHE["tile_grid_size"][1]),
    ),
    "nlm": (
        ("h", "h", "float", 0.0, 20.0, 0.1, _NLM["h"]),
        ("hColor", "hColor", "float", 0.0, 20.0, 0.1, _NLM["hColor"]),
        ("templateWindowSize", "Cửa sổ template", "int", 1, 99, 2,
         _NLM["templateWindowSize"]),
        ("searchWindowSize", "Cửa sổ tìm kiếm", "int", 1, 99, 2,
         _NLM["searchWindowSize"]),
    ),
}


def _sanitize_exception_text(exc: BaseException) -> str:
    """Rút gọn thông báo lỗi — không lộ đường dẫn host, chỉ tên file nếu có."""
    raw = str(exc)

    def _to_basename(match: re.Match[str]) -> str:
        segment = match.group(0).replace("\\", "/")
        return Path(segment).name

    sanitized = re.sub(r"(?:[A-Za-z]:)?(?:[\\/])[^\s\"']+", _to_basename, raw)
    return f"{type(exc).__name__}: {sanitized}"


class EnhanceScreen(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pool = make_pool()
        self._worker: EnhanceWorker | None = None
        self._cancel_event = threading.Event()
        self._path: Path | None = None
        self._doc = None
        self._derived: np.ndarray | None = None
        self._derived_pixmap: QPixmap | None = None
        self._run_meta: dict | None = None
        self._run_kind: str | None = None
        self._run_params: dict | None = None
        self._fields: dict[tuple[str, str], tuple[QLabel, QWidget]] = {}
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

        # Banner cố định — luôn hiển thị, không ẩn được.
        self.banner = QFrame()
        self.banner.setFrameShape(QFrame.Shape.StyledPanel)
        self.banner.setStyleSheet("background:#fef3c7; color:#713f12; padding:4px;")
        banner_lay = QVBoxLayout(self.banner)
        banner_lay.setContentsMargins(8, 4, 8, 4)
        banner_lbl = QLabel(strings.ENHANCE_BANNER)
        banner_lbl.setWordWrap(True)
        banner_lay.addWidget(banner_lbl)
        rl.addWidget(self.banner)

        self.ai_banner = QFrame()
        self.ai_banner.setObjectName("enhance_ai_banner")
        self.ai_banner.setFrameShape(QFrame.Shape.StyledPanel)
        self.ai_banner.setStyleSheet(
            "background:#ede9fe; color:#4c1d95; padding:4px;"
        )
        ai_banner_lay = QVBoxLayout(self.ai_banner)
        ai_banner_lay.setContentsMargins(8, 4, 8, 4)
        ai_banner_lbl = QLabel(strings.ENHANCE_AI_BANNER)
        ai_banner_lbl.setWordWrap(True)
        ai_banner_lay.addWidget(ai_banner_lbl)
        self.ai_banner.hide()
        rl.addWidget(self.ai_banner)

        self.btn_open = QPushButton(strings.BTN_OPEN)
        apply_a11y(self.btn_open, strings.A11Y_ENHANCE_OPEN)
        self.btn_open.clicked.connect(self._pick_file)
        rl.addWidget(self.btn_open)
        self.lbl_file = QLabel(strings.FMT_HINT)
        self.lbl_file.setWordWrap(True)
        rl.addWidget(self.lbl_file)
        self.lbl_meta = QLabel("")
        self.lbl_meta.setWordWrap(True)
        rl.addWidget(self.lbl_meta)

        rl.addWidget(QLabel(strings.LBL_ENHANCE_METHODS))
        self.combo_op = QComboBox()
        apply_a11y(self.combo_op, strings.A11Y_ENHANCE_OPERATOR)
        for kind, label in _OPERATORS:
            self.combo_op.addItem(label, kind)
        self.combo_op.currentIndexChanged.connect(self._on_operator_changed)
        rl.addWidget(self.combo_op)

        box = QGroupBox(strings.LBL_PARAMS_GROUP)
        apply_a11y(box, strings.A11Y_ENHANCE_PARAMS)
        self.params_box = box
        form = QFormLayout(box)
        for kind, specs in _FIELD_SPECS.items():
            for name, text, wtype, mn, mx, step, default in specs:
                if wtype == "int":
                    w = QSpinBox()
                    w.setRange(int(mn), int(mx))
                    w.setSingleStep(int(step))
                    w.setValue(int(default))
                else:
                    w = QDoubleSpinBox()
                    w.setRange(float(mn), float(mx))
                    w.setSingleStep(float(step))
                    w.setValue(float(default))
                label = QLabel(text)
                form.addRow(label, w)
                self._fields[(kind, name)] = (label, w)
        rl.addWidget(box)

        self.btn_run = QPushButton(strings.BTN_ENHANCE_RUN)
        apply_a11y(self.btn_run, strings.A11Y_ENHANCE_RUN)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton(strings.BTN_CANCEL)
        apply_a11y(self.btn_cancel, strings.A11Y_ENHANCE_CANCEL)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        rl.addWidget(self.btn_run)
        rl.addWidget(self.btn_cancel)
        self.progress = QProgressBar()
        apply_a11y(self.progress, strings.A11Y_ENHANCE_PROGRESS)
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

        blend_row = QHBoxLayout()
        blend_row.addWidget(QLabel(strings.LBL_BLEND_ORIGIN))
        self.slider_blend = QSlider(Qt.Orientation.Horizontal)
        apply_a11y(self.slider_blend, strings.A11Y_ENHANCE_BLEND)
        self.slider_blend.setRange(0, 100)
        self.slider_blend.setValue(100)
        self.slider_blend.setEnabled(False)
        self.slider_blend.valueChanged.connect(self._on_blend_changed)
        blend_row.addWidget(self.slider_blend, 1)
        self.lbl_blend_value = QLabel(strings.FMT_BLEND_VALUE.format(value=100))
        apply_a11y(self.lbl_blend_value, strings.A11Y_ENHANCE_BLEND_VALUE)
        blend_row.addWidget(self.lbl_blend_value)
        blend_row.addWidget(QLabel(strings.LBL_BLEND_ENHANCED))
        rl.addLayout(blend_row)

        self.btn_export = QPushButton(strings.BTN_EXPORT_ENHANCED)
        apply_a11y(self.btn_export, strings.A11Y_ENHANCE_EXPORT)
        self.btn_export.clicked.connect(self._export)
        self.btn_export.hide()
        rl.addWidget(self.btn_export)

        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        self.setAcceptDrops(True)
        self._on_operator_changed()

    def _show_busy_open(self) -> None:
        self.lbl_status.setText(strings.MSG_BUSY_CANNOT_OPEN)

    def _set_blend_readout(self, value: int) -> None:
        self.lbl_blend_value.setText(strings.FMT_BLEND_VALUE.format(value=value))

    # --- file ---
    def _pick_file(self) -> None:
        if self._worker is not None:
            self._show_busy_open()
            return  # không mở file mới khi đang chạy
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
            return
        urls = e.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.is_file():
                e.acceptProposedAction()
                self.load_path(path)

    def load_path(self, path: Path) -> None:
        if self._worker is not None:  # đang chạy: bỏ qua, tránh lệch ảnh/kết quả
            self._show_busy_open()
            return
        self._clear_result()
        self.lbl_err.setText("")
        self.lbl_status.setText("")
        self.lbl_file.setText(path.name)
        self.lbl_meta.setText("")
        # safe_open: header probe + decode đã chặn bomb; ảnh hỏng trả ErrorResult.
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

    # --- params ---
    def _on_operator_changed(self) -> None:
        kind = self.combo_op.currentData()
        self.ai_banner.setVisible(kind == "nafnet")
        self.params_box.setVisible(kind != "nafnet")
        for (field_kind, _name), (label, widget) in self._fields.items():
            visible = field_kind == kind
            label.setVisible(visible)
            widget.setVisible(visible)

    def _field(self, kind: str, name: str) -> QWidget:
        return self._fields[(kind, name)][1]

    def _current_params(self) -> dict:
        kind = self.combo_op.currentData()
        if kind == "nafnet":
            return {}
        if kind == "unsharp":
            return {
                "ksize": self._field(kind, "ksize").value(),
                "sigma": self._field(kind, "sigma").value(),
                "amount": self._field(kind, "amount").value(),
            }
        if kind == "clahe":
            return {
                "clip_limit": self._field(kind, "clip_limit").value(),
                "tile_grid_size": [
                    self._field(kind, "tile_x").value(),
                    self._field(kind, "tile_y").value(),
                ],
            }
        return {
            "h": self._field(kind, "h").value(),
            "hColor": self._field(kind, "hColor").value(),
            "templateWindowSize": self._field(kind, "templateWindowSize").value(),
            "searchWindowSize": self._field(kind, "searchWindowSize").value(),
        }

    def _validate_params(self, kind: str, params: dict) -> str:
        if kind == "unsharp" and params["ksize"] % 2 == 0:
            return "Kích thước kernel phải là số lẻ."
        if kind == "nlm" and (
            params["templateWindowSize"] % 2 == 0
            or params["searchWindowSize"] % 2 == 0
        ):
            return "Cửa sổ template và tìm kiếm phải là số lẻ."
        return ""

    # --- run ---
    def _clear_result(self) -> None:
        self._derived = None
        self._derived_pixmap = None
        self._run_meta = None
        self._run_kind = None
        self._run_params = None
        self.canvas.set_overlay(None)
        self.slider_blend.blockSignals(True)
        self.slider_blend.setValue(100)
        self.slider_blend.blockSignals(False)
        self._set_blend_readout(100)
        self.slider_blend.setEnabled(False)
        self.btn_export.hide()

    def _set_running(self, running: bool) -> None:
        self.btn_open.setEnabled(not running)
        self.btn_run.setEnabled(not running and self._path is not None)
        self.btn_cancel.setEnabled(running)
        self.combo_op.setEnabled(not running)
        for _label, widget in self._fields.values():
            widget.setEnabled(not running)
        if running:
            self.progress.setRange(0, 0)  # busy — operator không báo progress
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)

    def _run(self) -> None:
        if self._path is None or self._doc is None or self._worker is not None:
            return  # chưa có ảnh hoặc đang chạy — click thứ hai không làm gì
        kind = self.combo_op.currentData()
        params = self._current_params()
        error = self._validate_params(kind, params)
        if error:
            self.lbl_err.setText(error)
            return
        self.lbl_err.setText("")
        self.lbl_status.setText("")
        self._clear_result()
        self._run_kind = kind
        self._run_params = params
        self._cancel_event = threading.Event()
        worker = EnhanceWorker(self._doc.array, kind, params, self._cancel_event)
        worker.signals.result.connect(self._on_result)
        worker.signals.failed.connect(self._on_failed)
        worker.signals.finished.connect(self._on_finished)
        self._worker = worker
        self._set_running(True)
        self._pool.start(worker)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.btn_cancel.setEnabled(False)

    def _on_finished(self) -> None:
        cancelled = self._cancel_event.is_set()
        self._worker = None
        self._set_running(False)
        if cancelled:
            self.lbl_err.setText("")
            self.lbl_status.setText(strings.STATUS_ENHANCE_CANCELLED)

    def _on_result(self, payload) -> None:
        if self._cancel_event.is_set():
            return  # đã hủy — bỏ qua kết quả
        image, meta = payload
        self._derived = image
        self._run_meta = meta
        self._derived_pixmap = ndarray_to_qpixmap(image)
        self.canvas.set_overlay(self._derived_pixmap, alpha=1.0)
        self.slider_blend.blockSignals(True)
        self.slider_blend.setValue(100)
        self.slider_blend.blockSignals(False)
        self._set_blend_readout(100)
        self.slider_blend.setEnabled(True)
        self.btn_export.show()

    def _on_failed(self, msg: str) -> None:
        if self._cancel_event.is_set():
            return
        self.lbl_err.setText(msg)

    def _on_blend_changed(self, value: int) -> None:
        self._set_blend_readout(value)
        # Chỉ đổi pixmap hiển thị — không ghi pixel, không chạy lại operator.
        if self._derived_pixmap is None:
            self.canvas.set_overlay(None)
            return
        self.canvas.set_overlay(self._derived_pixmap, alpha=value / 100.0)

    # --- export ---
    def _export(self) -> None:
        if self._derived is None or self._path is None or self._doc is None:
            return
        if self._run_kind is None or self._run_params is None:
            return
        from imageguard.enhancement.derive import write_sibling

        default_name = f"{self._path.stem}_enhanced.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Xuất ảnh tăng cường", default_name, "PNG (*.png)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not save_path:
            return
        dest = Path(save_path)
        sidecar = dest.with_suffix(".json")
        if (
            dest.resolve() == self._path.resolve()
            or sidecar.resolve() == self._path.resolve()
        ):
            QMessageBox.warning(
                self, "Không thể xuất",
                "Đích xuất trùng với ảnh gốc — không ghi đè.")
            return
        overwrite = False
        if dest.exists() or sidecar.exists():
            answer = QMessageBox.question(
                self, "Ghi đè file?",
                f"File đã tồn tại:\n{dest.name}\nGhi đè?")
            if answer != QMessageBox.StandardButton.Yes:
                return
            overwrite = True
        try:
            json_path = write_sibling(
                self._path, dest, self._derived,
                self._run_kind, self._run_params, self._doc.sha256,
                overwrite=overwrite,
                model_sha256=(self._run_meta or {}).get("model_sha256"))
        except Exception as e:  # noqa: BLE001 - ranh giới UI: báo lỗi, không crash
            QMessageBox.warning(
                self, "Xuất thất bại",
                f"Không xuất được ảnh:\n{_sanitize_exception_text(e)}")
            return
        QMessageBox.information(
            self, "Thành công",
            f"Đã xuất ảnh tăng cường:\n{dest.name}\nMetadata: {json_path.name}")
