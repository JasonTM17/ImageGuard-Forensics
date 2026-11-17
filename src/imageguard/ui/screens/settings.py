"""Settings screen — thresholds viewer (nguồn mỗi giá trị hiển thị rõ)."""

from __future__ import annotations

import platform

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from imageguard import __version__
from imageguard.core.config import ThresholdSet
from imageguard.ui import strings
from imageguard.ui.a11y import apply_a11y


class SettingsScreen(QWidget):
    def __init__(self, cfg: ThresholdSet, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        intro_lbl = QLabel(strings.SETTINGS_THRESHOLDS_INTRO)
        intro_lbl.setWordWrap(True)
        lay.addWidget(intro_lbl)
        self.lbl_env = QLabel(
            f"ImageGuard v{__version__} · Thresholds hash: `{cfg.snapshot_hash()}` · "
            f"Python {platform.python_version()} · OpenCV {cv2.__version__} · "
            f"NumPy {np.__version__}"
        )
        self.lbl_env.setWordWrap(True)
        self.lbl_env.setStyleSheet("color:#4b5563; font-size:12px;")
        lay.addWidget(self.lbl_env)
        self.tree = QTreeWidget()
        apply_a11y(self.tree, strings.A11Y_SETTINGS_TREE)
        self.tree.setHeaderLabels([
            strings.SETTINGS_TREE_HDR_KEY,
            strings.SETTINGS_TREE_HDR_VALUE,
            strings.SETTINGS_TREE_HDR_SOURCE,
        ])
        lay.addWidget(self.tree)
        for section, entries in cfg.raw.items():
            if not isinstance(entries, dict):
                continue
            snode = QTreeWidgetItem([section])
            self.tree.addTopLevelItem(snode)
            for k, v in entries.items():
                if isinstance(v, dict) and "value" in v:
                    snode.addChild(QTreeWidgetItem(
                        [k, str(v["value"]), str(v.get("source", "—"))]))
        self.tree.expandAll()
