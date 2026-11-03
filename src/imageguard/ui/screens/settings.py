"""Settings screen — thresholds viewer (nguồn mỗi giá trị hiển thị rõ)."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QLabel,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from imageguard.core.config import ThresholdSet


class SettingsScreen(QWidget):
    def __init__(self, cfg: ThresholdSet, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "Ngưỡng thuật toán (configs/thresholds.yaml) — mỗi giá trị kèm "
            "nguồn; chỉnh sửa bằng cách sửa file yaml rồi khởi động lại."))
        tree = QTreeWidget()
        tree.setHeaderLabels(["Key", "Value", "Nguồn"])
        lay.addWidget(tree)
        for section, entries in cfg.raw.items():
            if not isinstance(entries, dict):
                continue
            snode = QTreeWidgetItem([section])
            tree.addTopLevelItem(snode)
            for k, v in entries.items():
                if isinstance(v, dict) and "value" in v:
                    snode.addChild(QTreeWidgetItem(
                        [k, str(v["value"]), str(v.get("source", "—"))]))
        tree.expandAll()
