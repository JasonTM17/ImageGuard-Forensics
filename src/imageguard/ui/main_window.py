"""MainWindow — tabs Analyze / History / Settings + disclaimer cố định."""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from imageguard.services.analysis_service import AnalysisService
from imageguard.services.history_service import HistoryService
from imageguard.ui import strings
from imageguard.ui.screens.analyze import AnalyzeScreen
from imageguard.ui.screens.history import HistoryScreen
from imageguard.ui.screens.settings import SettingsScreen
from imageguard.ui.widgets.verdict_panel import DisclaimerBar


class MainWindow(QMainWindow):
    def __init__(self, analysis: AnalysisService, history: HistoryService) -> None:
        super().__init__()
        self.setWindowTitle(strings.APP_TITLE)
        self.resize(1200, 800)
        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.analyze = AnalyzeScreen(analysis)
        self.history = HistoryScreen(history)
        self.settings = SettingsScreen(analysis.cfg)
        self.tabs.addTab(self.analyze, strings.TAB_ANALYZE)
        self.tabs.addTab(self.history, strings.TAB_HISTORY)
        self.tabs.addTab(self.settings, strings.TAB_SETTINGS)
        self.tabs.currentChanged.connect(
            lambda _: self.history.refresh() if self.tabs.currentWidget() is self.history else None)
        lay.addWidget(self.tabs, 1)
        lay.addWidget(DisclaimerBar())  # cố định dưới cùng — không bao giờ ẩn
