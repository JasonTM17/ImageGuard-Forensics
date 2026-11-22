"""SampleGeneratorDialog — tạo nhanh ảnh mẫu tổng hợp phục vụ demo/kiểm thử đồ án."""

from __future__ import annotations

import secrets
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from imageguard.core.paths import data_root
from imageguard.ui import strings

DEFAULT_DEMO_DIR = data_root() / "demo_samples"

_DEMO_SPECS: dict[str, tuple[str, list[str]]] = {
    "copymove": (
        "Ảnh kết cấu đa tần số 256×256 có vùng 68×68 được sao chép và dán dịch vị trí.",
        ["keypoint", "block", "ela"],
    ),
    "splicing": (
        "Ảnh ghép vùng 256×256 tạo chênh lệch nhiễu cảm biến rõ rệt giữa vùng ghép và nền.",
        ["ela", "noise", "noise_local", "splicing"],
    ),
    "deepfake": (
        "Khung hình 320×320 mô phỏng khuôn mặt da (YCrCb/HSV) bị làm mịn mất lỗ chân lông kèm dấu vết tuần hoàn FFT.",
        ["ela", "noise_local", "freq", "deepfake"],
    ),
    "clean": (
        "Ảnh đối chứng sạch 256×256 có nhiễu cảm biến đồng nhất tự nhiên — không có can thiệp.",
        ["keypoint", "block", "ela", "deepfake"],
    ),
}


def generate_demo_sample(
    kind: str,
    out_dir: Path | None = None,
) -> tuple[Path, list[str]]:
    """Sinh ảnh mẫu tổng hợp theo ``kind`` và trả về ``(đường dẫn PNG, methods gợi ý)``."""
    if kind not in _DEMO_SPECS:
        raise ValueError(f"Loại mẫu không hợp lệ: {kind!r}")

    target_dir = Path(out_dir) if out_dir is not None else data_root() / "demo_samples"
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261113)

    if kind == "copymove":
        yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
        base = np.zeros((256, 256, 3), dtype=np.float32)
        base[:, :, 0] = 115 + 45 * np.sin(xx / 7.5) * np.cos(yy / 9.0) + 22 * np.sin((xx + yy) / 5.0)
        base[:, :, 1] = 128 + 42 * np.cos(xx / 6.0) * np.sin(yy / 8.0)
        base[:, :, 2] = 120 + 38 * np.sin((xx - yy) / 6.5)
        base += rng.normal(0.0, 6.0, base.shape).astype(np.float32)
        img = np.clip(base, 0, 255).astype(np.uint8)
        # Vẽ thêm chi tiết đa góc cạnh trong vùng nguồn để keypoint & DCT bắt chuẩn xác.
        cv2.circle(img, (68, 68), 22, (230, 195, 65), -1)
        cv2.rectangle(img, (44, 48), (88, 90), (45, 75, 220), 2)
        cv2.line(img, (40, 96), (98, 42), (245, 245, 245), 2)
        patch = img[36:104, 36:104].copy()
        img[142:210, 142:210] = patch

    elif kind == "splicing":
        yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
        bg = np.clip(
            122 + 25 * np.sin(xx / 18.0)[:, :, None] + rng.normal(0.0, 15.0, (256, 256, 3)),
            0,
            255,
        ).astype(np.uint8)
        patch = np.clip(
            136 + 18 * np.cos(yy[64:192, 64:192] / 14.0)[:, :, None] + rng.normal(0.0, 1.2, (128, 128, 3)),
            0,
            255,
        ).astype(np.uint8)
        bg[64:192, 64:192] = patch
        img = bg

    elif kind == "deepfake":
        img = np.clip(96 + rng.normal(0.0, 14.0, (320, 320, 3)), 0, 255).astype(np.uint8)
        img[:, :, 0] = np.clip(img[:, :, 0].astype(np.int16) + 32, 0, 255).astype(np.uint8)
        face_mask = np.zeros((320, 320), dtype=np.uint8)
        cv2.ellipse(face_mask, (175, 136), (58, 74), 0, 0, 360, 255, -1)
        face_patch = np.zeros((320, 320, 3), dtype=np.float32)
        face_patch[:, :] = np.array([135.0, 168.0, 215.0], dtype=np.float32)
        cv2.circle(face_patch, (152, 118), 11, (55.0, 75.0, 105.0), -1)
        cv2.circle(face_patch, (198, 118), 11, (55.0, 75.0, 105.0), -1)
        cv2.ellipse(face_patch, (175, 166), (20, 8), 0, 0, 180, (95.0, 105.0, 155.0), 2)
        face_patch = cv2.GaussianBlur(face_patch, (15, 15), 4.5)
        yy, xx = np.mgrid[0:320, 0:320].astype(np.float32)
        grid = 10.0 * np.cos(2.0 * np.pi * xx / 3.0) * np.cos(2.0 * np.pi * yy / 3.0)
        face_patch += grid[:, :, None]
        m = face_mask > 0
        img[m] = np.clip(face_patch[m], 0, 255).astype(np.uint8)

    else:  # clean
        yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
        base = np.zeros((256, 256, 3), dtype=np.float32)
        base[:, :, 0] = 105 + 32 * np.sin(xx / 24.0) * np.cos(yy / 20.0)
        base[:, :, 1] = 128 + 26 * np.cos(xx / 19.0)
        base[:, :, 2] = 135 + 24 * np.sin(yy / 22.0)
        base += rng.normal(0.0, 11.0, base.shape).astype(np.float32)
        img = np.clip(base, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Không tạo được tệp PNG ảnh mẫu.")
    out_path = target_dir / f"demo_{kind}_{secrets.token_hex(4)}.png"
    out_path.write_bytes(buf.tobytes())
    return out_path, list(_DEMO_SPECS[kind][1])


class SampleGeneratorDialog(QDialog):
    """Hộp thoại chọn và sinh nhanh ảnh mẫu phục vụ trình diễn đồ án."""

    def __init__(self, parent=None, *, out_dir: Path | None = None) -> None:
        super().__init__(parent)
        self._out_dir = Path(out_dir) if out_dir is not None else data_root() / "demo_samples"
        self.generated_path: Path | None = None
        self.recommended_methods: list[str] = []
        self.setWindowTitle(strings.DEMO_DIALOG_TITLE)
        self.setMinimumWidth(460)
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)

        intro = QLabel(
            "Chọn một kịch bản tổng hợp để sinh ảnh kiểm thử ngay và tự động gợi ý các phương pháp tương ứng:"
        )
        intro.setWordWrap(True)
        lay.addWidget(intro)

        self.rad_copymove = QRadioButton("1. Mẫu Copy-Move (sao chép vùng SIFT / Block DCT)")
        self.rad_splicing = QRadioButton("2. Mẫu Ghép vùng Splicing (lệch nhiễu cảm biến cục bộ)")
        self.rad_deepfake = QRadioButton("3. Mẫu Tín hiệu Deepfake / AI (làm mịn khuôn mặt & lưới FFT)")
        self.rad_clean = QRadioButton("4. Ảnh sạch đối chứng (nhiễu đồng nhất tự nhiên)")
        self.rad_copymove.setChecked(True)

        self._radios = (
            ("copymove", self.rad_copymove),
            ("splicing", self.rad_splicing),
            ("deepfake", self.rad_deepfake),
            ("clean", self.rad_clean),
        )
        for _k, rad in self._radios:
            rad.toggled.connect(self._update_desc)
            lay.addWidget(rad)

        self.lbl_desc = QLabel("")
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setStyleSheet("color:#374151; padding:6px; background:#f3f4f6; border-radius:4px;")
        lay.addWidget(self.lbl_desc)
        self._update_desc()

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_create = QPushButton("Tạo & Mở vào Phân tích")
        self.btn_create.clicked.connect(self.create_now)
        self.btn_cancel = QPushButton(strings.BTN_CANCEL)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_create)
        btn_row.addWidget(self.btn_cancel)
        lay.addLayout(btn_row)

    def _selected_kind(self) -> str:
        for key, rad in self._radios:
            if rad.isChecked():
                return key
        return "copymove"

    def _update_desc(self) -> None:
        key = self._selected_kind()
        desc, methods = _DEMO_SPECS[key]
        self.lbl_desc.setText(f"{desc}\nPhương pháp gợi ý: {', '.join(methods)}")

    def create_now(self) -> None:
        path, methods = generate_demo_sample(self._selected_kind(), self._out_dir)
        self.generated_path = path
        self.recommended_methods = methods
        self.accept()
