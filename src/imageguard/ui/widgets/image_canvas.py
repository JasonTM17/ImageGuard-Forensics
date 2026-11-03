"""ImageCanvas — QGraphicsView zoom/pan + overlay layer.

numpy→QImage PHẢI .copy() — QImage giữ pointer tới buffer ndarray, ndarray
bị GC = dangling pointer crash. Đây là bug kinh điển của bridge này.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


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
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._base: QGraphicsPixmapItem | None = None
        self._overlay: QGraphicsPixmapItem | None = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def set_image(self, bgr: np.ndarray) -> None:
        self._scene.clear()
        self._base = self._scene.addPixmap(ndarray_to_qpixmap(bgr))
        self._overlay = None
        self.fitInView(self._base, Qt.AspectRatioMode.KeepAspectRatio)

    def set_overlay(self, pixmap: QPixmap | None, alpha: float = 0.45) -> None:
        if self._overlay is not None:
            self._scene.removeItem(self._overlay)
            self._overlay = None
        if pixmap is not None:
            self._overlay = self._scene.addPixmap(pixmap)
            self._overlay.setOpacity(alpha)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
