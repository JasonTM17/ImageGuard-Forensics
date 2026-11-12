"""ImageCanvas — QGraphicsView zoom/pan + overlay layer.

numpy→QImage PHẢI .copy() — QImage giữ pointer tới buffer ndarray, ndarray
bị GC = dangling pointer crash. Đây là bug kinh điển của bridge này.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QImage,
    QKeyEvent,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from imageguard.ui import strings
from imageguard.ui.a11y import apply_a11y


def ndarray_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(img)


def heat_to_qpixmap(heat: np.ndarray, base_shape: tuple[int, int]) -> QPixmap:
    from imageguard.forensics.localize.heatmap import render_heatmap

    hm = render_heatmap(heat)
    if hm.shape[:2] != base_shape:
        hm = cv2.resize(hm, (base_shape[1], base_shape[0]))
    return ndarray_to_qpixmap(hm)


class ImageCanvas(QGraphicsView):
    image_dropped = Signal(Path)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._base: QGraphicsPixmapItem | None = None
        self._overlay: QGraphicsPixmapItem | None = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        apply_a11y(self, strings.A11Y_CANVAS, strings.A11Y_CANVAS_DESC)
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self._scene.clear()
        self._base = None
        self._overlay = None
        text_item = self._scene.addText(strings.CANVAS_PLACEHOLDER)
        text_item.setDefaultTextColor(Qt.GlobalColor.darkGray)
        font = text_item.font()
        font.setPointSize(11)
        text_item.setFont(font)
        text_item.setPos(
            -text_item.boundingRect().width() / 2,
            -text_item.boundingRect().height() / 2,
        )
        self.setSceneRect(-250, -150, 500, 300)

    def clear_image(self) -> None:
        self._show_placeholder()

    def set_image(self, bgr: np.ndarray) -> None:
        self._scene.clear()
        self._base = self._scene.addPixmap(ndarray_to_qpixmap(bgr))
        self._overlay = None
        self.setSceneRect(self._base.boundingRect())
        self.fitInView(self._base, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # fitInView lúc set_image có thể chạy TRƯỚC khi viewport có kích thước
        # cuối → ảnh zoom sai. Re-fit khi widget thay đổi kích thước.
        if self._base is not None:
            self.fitInView(self._base, Qt.AspectRatioMode.KeepAspectRatio)

    def set_overlay(self, pixmap: QPixmap | None, alpha: float = 0.45) -> None:
        if self._overlay is not None:
            self._scene.removeItem(self._overlay)
            self._overlay = None
        if pixmap is not None:
            self._overlay = self._scene.addPixmap(pixmap)
            self._overlay.setOpacity(alpha)

    def _zoom_step(self, zoom_in: bool) -> None:
        if self._base is None:
            return
        factor = 1.25 if zoom_in else 0.8
        self.scale(factor, factor)

    def _pan_step(self, dx: int, dy: int) -> None:
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - dx)
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - dy)

    def _reset_view(self) -> None:
        if self._base is not None:
            self.resetTransform()
            self.fitInView(self._base, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._base is not None:
            self._zoom_step(event.angleDelta().y() > 0)
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._zoom_step(True)
            event.accept()
            return
        if key == Qt.Key.Key_Minus:
            self._zoom_step(False)
            event.accept()
            return
        step = 48
        if key == Qt.Key.Key_Left:
            self._pan_step(step, 0)
            event.accept()
            return
        if key == Qt.Key.Key_Right:
            self._pan_step(-step, 0)
            event.accept()
            return
        if key == Qt.Key.Key_Up:
            self._pan_step(0, step)
            event.accept()
            return
        if key == Qt.Key.Key_Down:
            self._pan_step(0, -step)
            event.accept()
            return
        if key == Qt.Key.Key_Home:
            self._reset_view()
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                path = Path(urls[0].toLocalFile())
                if path.is_file():
                    event.acceptProposedAction()
                    self.image_dropped.emit(path)
                    return
        super().dropEvent(event)
