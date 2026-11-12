"""Settings screen — thresholds viewer (nguồn mỗi giá trị hiển thị rõ)."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

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
