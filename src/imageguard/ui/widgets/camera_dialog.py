"""CameraCaptureDialog — chọn nguồn Webcam, Camera USB ngoài, Dual Camera hoặc Camera IP/IoT."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
from collections.abc import Callable
from functools import partial
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from imageguard.core.io import _capture_layout
from imageguard.core.paths import data_root
from imageguard.ingest.camera import (
    capture_device_frame,
    capture_dual_device_frames,
    capture_iot_http_frame,
    probe_camera_indices,
)
from imageguard.ui import strings
from imageguard.ui.a11y import apply_a11y
from imageguard.ui.error_redaction import sanitize_exception_text
from imageguard.ui.widgets.camera_deepfake_preview import (
    format_burst_deepfake_preview,
    format_deepfake_preview,
)


def analyze_camera_frame_ipkit(
    bgr: np.ndarray, is_dual: bool = False, split_x: int | None = None
) -> str:
    """Chẩn đoán quang học thời gian thực bằng ipkit (Chương 2, 3, 4, 6)."""
    from imageguard.ipkit import edges, features_basic, histogram, threshold

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))
    otsu_t = int(threshold.otsu_threshold(gray))
    hist_counts = histogram.histogram(gray)
    total_px = max(int(np.sum(hist_counts)), 1)
    p_dark = 100.0 * float(np.sum(hist_counts[:30])) / total_px
    p_clip = 100.0 * float(np.sum(hist_counts[240:])) / total_px
    sharpness = float(np.var(edges.laplacian(gray)))

    if is_dual and w >= 64:
        mid = w // 2 if split_x is None else split_x
        if type(mid) is not int or not 0 < mid < w:
            raise ValueError("Vị trí ghép hai camera không hợp lệ.")
        left_g = gray[:, :mid]
        right_g = gray[:, mid:]
        dluma = abs(float(np.mean(left_g)) - float(np.mean(right_g)))
        sharp_l = float(np.var(edges.laplacian(left_g)))
        sharp_r = float(np.var(edges.laplacian(right_g)))
        left_s = cv2.resize(left_g, (96, 96), interpolation=cv2.INTER_AREA)
        right_s = cv2.resize(right_g, (96, 96), interpolation=cv2.INTER_AREA)
        h_l = features_basic.lbph(left_s, grid=(4, 4), uniform=True)
        h_r = features_basic.lbph(right_s, grid=(4, 4), uniform=True)
        chi2_raw = float(features_basic.chi_square_distance(h_l, h_r))
        matched_l = histogram.match_histogram(left_s, right_s)
        h_m = features_basic.lbph(matched_l, grid=(4, 4), uniform=True)
        chi2_matched = float(features_basic.chi_square_distance(h_m, h_r))
        return (
            f"Ảnh ghép 2 Camera ({w}×{h}) · Độ sáng TB: {mean_luma:.1f}/255 · "
            f"Độ nét Trái/Phải: {sharp_l:.0f}/{sharp_r:.0f} · Chênh sáng: {dluma:.1f} · "
            f"Sai biệt kết cấu χ²: {chi2_raw:.2f} → sau hiệu chuẩn: {chi2_matched:.2f}"
        )

    return (
        f"Chẩn đoán cảm biến: {w}×{h} · Độ sáng TB: {mean_luma:.1f}/255 · "
        f"Ngưỡng tách nền: {otsu_t} · Độ nét tiêu cự: {sharpness:.0f} · "
        f"Vùng tối/cháy sáng: {p_dark:.1f}%/{p_clip:.1f}%"
    )


def _append_deepfake_text(
    diagnostics: str, frame: np.ndarray | None, *,
    capture_layout: str = "single", capture_split_x: int | None = None,
) -> str:
    if frame is None:
        return diagnostics
    try:
        extra = format_deepfake_preview(
            frame, capture_layout=capture_layout, capture_split_x=capture_split_x
        )
    except Exception as exc:  # noqa: BLE001 - chỉ hiện tên loại lỗi, giữ chẩn đoán quang học
        extra = type(exc).__name__
    return f"{diagnostics}\n\n{extra}" if diagnostics else extra


def _pillar_text_from_file(path: Path) -> str:
    raw = path.read_bytes()
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không đọc được khung đã chụp.")
    with Image.open(BytesIO(raw)) as im:
        layout, split_x = _capture_layout(im.info, im.size, im.getexif().get(274, 1))
    manifest = path.with_name(f"{path.stem}.burst.json")
    if manifest.is_file():
        from imageguard.ingest.camera_burst import load_burst_frames

        return format_burst_deepfake_preview(
            load_burst_frames(
                manifest,
                expected_primary_sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    return format_deepfake_preview(frame, capture_layout=layout, capture_split_x=split_x)


def _preview_result(capture: Callable[[Path], Path], is_dual: bool) -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory() as tmp:
        preview_path = capture(Path(tmp))
        raw = preview_path.read_bytes()
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Không đọc được ảnh xem trước từ camera.")
        with Image.open(BytesIO(raw)) as im:
            layout, split_x = _capture_layout(im.info, im.size, im.getexif().get(274, 1))
        if is_dual and layout == "single":
            # Explicit UI choice supplies the layout for older capture adapters.
            layout, split_x = "dual_side_by_side", frame.shape[1] // 2
        diagnostics = analyze_camera_frame_ipkit(
            frame, is_dual=layout == "dual_side_by_side", split_x=split_x
        )
        if preview_path.with_name(f"{preview_path.stem}.burst.json").is_file():
            try:
                deepfake_text = _pillar_text_from_file(preview_path)
            except Exception as exc:  # noqa: BLE001 - preview phụ không được xoá chẩn đoán quang học
                deepfake_text = type(exc).__name__
            diagnostics = f"{diagnostics}\n\n{deepfake_text}" if diagnostics else deepfake_text
        else:
            diagnostics = _append_deepfake_text(
                diagnostics, frame, capture_layout=layout, capture_split_x=split_x
            )
    return raw, diagnostics


def _unlink_matching_file(path: Path, expected_sha256: str | None = None) -> None:
    """Xoá file artifact do dialog tạo; nếu có hash thì không xoá file đã bị đổi."""
    try:
        if not path.is_file():
            return
        if expected_sha256 is not None and (
            hashlib.sha256(path.read_bytes()).hexdigest().lower()
            != expected_sha256.lower()
        ):
            return
        path.unlink()
    except OSError:
        pass


def discard_capture_artifacts(path: Path, capture_dir: Path) -> None:
    """Bỏ capture chưa được accept cùng burst sidecar do chính dialog tạo.

    Chỉ đụng file ngay trong ``capture_dir``. Frame burst chỉ bị xoá khi hash
    hiện tại còn khớp manifest; nếu user đã chỉnh sửa file, giữ nguyên thay vì
    xoá dữ liệu đã thay đổi.
    """
    try:
        root = capture_dir.resolve()
        target = path.resolve()
    except OSError:
        return
    if target.parent != root:
        return
    manifest = target.with_name(f"{target.stem}.burst.json")
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_bytes().decode("utf-8"))
            sha_map = data.get("sha256") if isinstance(data, dict) else None
            names = data.get("frames") if isinstance(data, dict) else None
            if isinstance(sha_map, dict) and isinstance(names, list):
                for name in names:
                    if isinstance(name, str) and Path(name).name == name and ".." not in name:
                        expected = sha_map.get(name)
                        if isinstance(expected, str):
                            _unlink_matching_file(root / name, expected)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        manifest.unlink(missing_ok=True)
    try:
        target.unlink(missing_ok=True)
    except OSError:
        pass


def _capture_result(
    capture: Callable[[Path], Path], out_dir: Path
) -> tuple[Path, bytes, str, str]:
    path = capture(out_dir)
    raw = path.read_bytes()
    try:
        pillar_text = _pillar_text_from_file(path)
        pillar_error = ""
    except Exception as exc:  # noqa: BLE001 - giữ PNG gốc để phân tích dù bảng trụ lỗi
        pillar_text = ""
        pillar_error = type(exc).__name__
    return path, raw, pillar_text, pillar_error


class _CameraSignals(QObject):
    completed = Signal(object, str)


class _CameraWorker(QRunnable):
    """Native camera I/O can finish after cancellation. Always signal cleanup."""

    def __init__(self, task: Callable[[], object]) -> None:
        super().__init__()
        self.task = task
        self.cancelled = threading.Event()
        self.signals = _CameraSignals()

    def run(self) -> None:
        value, error = None, ""
        try:
            if not self.cancelled.is_set():
                value = self.task()
        except Exception as exc:  # noqa: BLE001 - surface device/decode failures through Qt signals
            error = sanitize_exception_text(exc)
        finally:
            self.signals.completed.emit(value, error)


class CameraCaptureDialog(QDialog):
    """Hộp thoại chụp một khung hình thô từ webcam máy tính hoặc camera ngoài."""

    def __init__(self, parent=None, *, out_dir: Path | None = None) -> None:
        super().__init__(parent)
        self._out_dir = Path(out_dir) if out_dir is not None else data_root() / "captures"
        self._worker: _CameraWorker | None = None
        self._operation: str | None = None
        self._closed = False
        self.captured_path: Path | None = None
        self.setWindowTitle(strings.CAMERA_DIALOG_TITLE)
        self.setMinimumWidth(480)
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)

        note_box = QFrame()
        note_box.setFrameShape(QFrame.Shape.StyledPanel)
        note_box.setStyleSheet("background:#eff6ff; color:#1e3a8a; padding:6px;")
        note_lay = QVBoxLayout(note_box)
        note_lay.setContentsMargins(8, 6, 8, 6)
        self.lbl_note = QLabel(strings.CAMERA_FORENSIC_RULE_NOTE)
        self.lbl_note.setWordWrap(True)
        note_lay.addWidget(self.lbl_note)
        lay.addWidget(note_box)

        self.rad_webcam = QRadioButton(strings.CAMERA_SOURCE_WEBCAM)
        self.rad_webcam.setChecked(True)
        lay.addWidget(self.rad_webcam)

        usb_row = QHBoxLayout()
        self.rad_usb = QRadioButton(strings.CAMERA_SOURCE_USB)
        self.spin_usb = QSpinBox()
        self.spin_usb.setRange(1, 5)
        self.spin_usb.setValue(1)
        self.spin_usb.setEnabled(False)
        usb_row.addWidget(self.rad_usb, 1)
        usb_row.addWidget(self.spin_usb)
        lay.addLayout(usb_row)

        self.rad_dual = QRadioButton(strings.CAMERA_SOURCE_DUAL)
        lay.addWidget(self.rad_dual)

        self.rad_iot = QRadioButton(strings.CAMERA_SOURCE_IOT)
        lay.addWidget(self.rad_iot)
        self.edit_url = QLineEdit()
        self.edit_url.setPlaceholderText("http://192.168.1.100:8080/shot.jpg")
        self.edit_url.setEnabled(False)
        lay.addWidget(self.edit_url)

        self.chk_burst = QCheckBox(strings.CAMERA_BURST_CHECK)
        self.chk_burst.setToolTip(strings.CAMERA_BURST_HINT)
        self._burst_enabled = True
        self.chk_burst.blockSignals(True)
        self.chk_burst.setChecked(True)
        self.chk_burst.blockSignals(False)
        self.chk_burst.toggled.connect(self._on_burst_toggled)
        lay.addWidget(self.chk_burst)

        for rad in (self.rad_webcam, self.rad_usb, self.rad_dual, self.rad_iot):
            rad.toggled.connect(self._sync_inputs)

        self.lbl_preview = QLabel("Chưa có ảnh xem trước")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setMinimumHeight(140)
        self.lbl_preview.setStyleSheet(
            "border:1px solid #d1d5db; background:#f9fafb; color:#6b7280;"
        )
        lay.addWidget(self.lbl_preview)

        self.lbl_err = QLabel("")
        self.lbl_err.setWordWrap(True)
        self.lbl_err.setStyleSheet("color:#b91c1c;")
        lay.addWidget(self.lbl_err)

        self.lbl_probe_info = QLabel("")
        self.lbl_probe_info.setWordWrap(True)
        self.lbl_probe_info.setStyleSheet("color:#1e40af; font-size:12px;")
        lay.addWidget(self.lbl_probe_info)

        btn_row = QHBoxLayout()
        self.btn_probe = QPushButton(strings.CAMERA_BTN_PROBE)
        self.btn_probe.clicked.connect(self.probe_ports)
        self.btn_preview = QPushButton("Xem thử thiết bị")
        self.btn_preview.clicked.connect(self.preview_now)
        self.btn_capture = QPushButton("Chụp khung hình")
        apply_a11y(self.btn_capture, strings.A11Y_CAMERA_OPEN)
        self.btn_capture.clicked.connect(self.capture_now)
        self.btn_cancel = QPushButton(strings.BTN_CANCEL)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_probe)
        btn_row.addWidget(self.btn_preview)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_capture)
        btn_row.addWidget(self.btn_cancel)
        lay.addLayout(btn_row)

        post_row = QHBoxLayout()
        self.btn_accept_frame = QPushButton(strings.CAMERA_BTN_ACCEPT_FRAME)
        self.btn_accept_frame.clicked.connect(self.accept)
        self.btn_accept_frame.setVisible(False)
        self.btn_retake = QPushButton(strings.CAMERA_BTN_RETAKE)
        self.btn_retake.clicked.connect(self._retake_capture)
        self.btn_retake.setVisible(False)
        post_row.addStretch(1)
        post_row.addWidget(self.btn_retake)
        post_row.addWidget(self.btn_accept_frame)
        lay.addLayout(post_row)

    def probe_ports(self) -> None:
        """Start a port probe and update the suggestion when it completes."""
        self._start_job("probe", partial(probe_camera_indices, max_index=4))

    def _show_ports(self, found: list[int]) -> None:
        if not found:
            self.lbl_probe_info.setText("Không tìm thấy thiết bị camera cục bộ (index 0..3).")
            return
        ext = [idx for idx in found if idx >= 1]
        if ext:
            self.spin_usb.setValue(int(ext[0]))
        lines = [
            f"Thiết bị phát hiện: {found}"
            + (f" · Cổng camera ngoài gợi ý: {ext[0]}" if ext else " · Chỉ thấy Webcam 0")
        ]
        if 0 in found and ext:
            lines.append(strings.CAMERA_DUAL_HINT)
        elif len(found) == 1:
            lines.append(strings.CAMERA_SINGLE_LIMIT)
        self.lbl_probe_info.setText("\n".join(lines))

    def _on_burst_toggled(self, checked: bool) -> None:
        self._burst_enabled = bool(checked)
        hint = strings.CAMERA_BURST_HINT
        text = self.lbl_probe_info.text()
        if checked and hint not in text:
            self.lbl_probe_info.setText((text + "\n" + hint).strip())
        elif not checked and hint in text:
            self.lbl_probe_info.setText(text.replace(hint, "").strip())

    def _burst_selected(self) -> bool:
        return self._burst_enabled and (
            self.rad_webcam.isChecked() or self.rad_usb.isChecked()
        )

    def _capture_burst(self, device_index: int, out_dir: Path) -> Path:
        from imageguard.ingest.camera_burst import capture_device_burst

        primary, _manifest = capture_device_burst(
            device_index, out_dir, frame_count=5
        )
        return primary

    def _sync_inputs(self) -> None:
        idle = self._worker is None
        single = self.rad_webcam.isChecked() or self.rad_usb.isChecked()
        self.spin_usb.setEnabled(idle and (self.rad_usb.isChecked() or self.rad_dual.isChecked()))
        self.edit_url.setEnabled(idle and self.rad_iot.isChecked())
        self.chk_burst.setEnabled(idle and single)
        if not single:
            self._burst_enabled = False
            if self.chk_burst.isChecked():
                self.chk_burst.setChecked(False)

    def _capture_request(self) -> Callable[[Path], Path]:
        """Snapshot Qt inputs on the GUI thread before dispatching native I/O."""
        if self._burst_selected():
            idx = 0 if self.rad_webcam.isChecked() else int(self.spin_usb.value())
            return partial(self._capture_burst, idx)
        if self.rad_webcam.isChecked():
            return partial(capture_device_frame, 0)
        if self.rad_usb.isChecked():
            return partial(capture_device_frame, int(self.spin_usb.value()))
        if self.rad_dual.isChecked():
            return partial(capture_dual_device_frames, 0, int(self.spin_usb.value()))
        return partial(capture_iot_http_frame, self.edit_url.text())

    def _set_busy(self, busy: bool) -> None:
        for control in (
            self.btn_probe,
            self.btn_preview,
            self.btn_capture,
            self.rad_webcam,
            self.rad_usb,
            self.rad_dual,
            self.rad_iot,
            self.chk_burst,
        ):
            control.setEnabled(not busy)
        if busy:
            self.btn_accept_frame.setEnabled(False)
            self.btn_retake.setEnabled(False)
        else:
            pending = self.captured_path is not None
            self.btn_accept_frame.setEnabled(pending and self.btn_accept_frame.isVisible())
            self.btn_retake.setEnabled(pending and self.btn_retake.isVisible())
        self._sync_inputs()

    def _show_frame_preview(self, raw: bytes) -> bool:
        pix = QPixmap()
        if not pix.loadFromData(raw):
            return False
        self.lbl_preview.setPixmap(
            pix.scaled(
                420,
                180,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        return True

    def _retake_capture(self) -> None:
        if self.captured_path is not None:
            discard_capture_artifacts(self.captured_path, self._out_dir)
        self.captured_path = None
        self.lbl_preview.clear()
        self.lbl_preview.setText("Chưa có ảnh xem trước")
        self.lbl_probe_info.setText("")
        self.btn_accept_frame.setVisible(False)
        self.btn_retake.setVisible(False)
        self._set_busy(False)

    def _start_job(self, operation: str, task: Callable[[], object]) -> None:
        if self._worker is not None or self._closed:
            return
        self.lbl_err.setText("")
        worker = _CameraWorker(task)
        self._worker, self._operation = worker, operation
        worker.signals.completed.connect(self._on_completed)
        self._set_busy(True)
        # The application pool outlives a cancelled/deleted modal dialog.
        QThreadPool.globalInstance().start(worker)

    @Slot(object, str)
    def _on_completed(self, value: object, error: str) -> None:
        worker, operation = self._worker, self._operation
        self._worker, self._operation = None, None
        if self._closed:
            if operation == "capture" and isinstance(value, tuple) and value:
                discard_capture_artifacts(Path(value[0]), self._out_dir)
            return
        self._set_busy(False)
        if worker is None or worker.cancelled.is_set():
            return
        if error:
            self.lbl_err.setText(error)
            return
        if operation == "probe":
            self._show_ports(value)
        elif operation == "preview":
            raw, diagnostics = value
            if not self._show_frame_preview(raw):
                self.lbl_err.setText("Không hiển thị được ảnh xem trước từ camera.")
                return
            self.lbl_probe_info.setText(diagnostics)
        elif operation == "capture":
            path, raw, pillar_text, pillar_error = value
            self.captured_path = path
            if not self._show_frame_preview(raw):
                self.lbl_err.setText("Không hiển thị được khung đã chụp.")
                self.captured_path = None
                return
            self.lbl_probe_info.setText(pillar_text)
            if pillar_error:
                self.lbl_err.setText(pillar_error)
            self.btn_accept_frame.setVisible(True)
            self.btn_retake.setVisible(True)
            self._set_busy(False)

    def preview_now(self) -> None:
        self._start_job("preview", partial(_preview_result, self._capture_request(),
                                           self.rad_dual.isChecked()))

    def capture_now(self) -> None:
        if self._worker is not None or self._closed:
            return
        self.captured_path = None
        self._start_job("capture", partial(_capture_result, self._capture_request(), self._out_dir))

    def done(self, result: int) -> None:
        self._closed = True
        if result != QDialog.DialogCode.Accepted and self.captured_path is not None:
            discard_capture_artifacts(self.captured_path, self._out_dir)
            self.captured_path = None
        if self._worker is not None:
            self._worker.cancelled.set()
        super().done(result)

    def showEvent(self, event) -> None:
        self._closed = False
        self.captured_path = None
        self.btn_accept_frame.setVisible(False)
        self.btn_retake.setVisible(False)
        self._set_busy(self._worker is not None)
        super().showEvent(event)
