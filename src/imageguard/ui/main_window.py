"""MainWindow — tabs Analyze / Enhance / History / Settings."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from imageguard.services.analysis_service import AnalysisService
from imageguard.services.history_service import HistoryService
from imageguard.ui import strings
from imageguard.ui.a11y import apply_a11y
from imageguard.ui.screens.analyze import AnalyzeScreen
from imageguard.ui.screens.enhance import EnhanceScreen
from imageguard.ui.screens.history import HistoryScreen
from imageguard.ui.screens.settings import SettingsScreen
from imageguard.ui.screens.stats import StatsScreen


class MainWindow(QMainWindow):
    def __init__(self, analysis: AnalysisService, history: HistoryService) -> None:
        super().__init__()
        self.setWindowTitle(strings.APP_TITLE)
        self.resize(1200, 800)
        self.setAcceptDrops(True)
        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.analyze = AnalyzeScreen(analysis)
        self.enhance = EnhanceScreen()
        self.history = HistoryScreen(history)
        self.stats = StatsScreen(history)
        self.settings = SettingsScreen(analysis.cfg)
        self.tabs.addTab(self.analyze, strings.TAB_ANALYZE)
        self.tabs.addTab(self.enhance, strings.TAB_ENHANCE)
        self.tabs.addTab(self.history, strings.TAB_HISTORY)
        self.tabs.addTab(self.stats, strings.TAB_STATS)
        self.tabs.addTab(self.settings, strings.TAB_SETTINGS)
        apply_a11y(self.tabs, strings.A11Y_MAIN_TABS)

        def _refresh(i):
            cur = self.tabs.currentWidget()
            if cur is self.history:
                self.history.refresh()
            elif cur is self.stats:
                self.stats.refresh()

        self.tabs.currentChanged.connect(_refresh)
        lay.addWidget(self.tabs, 1)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                path = Path(urls[0].toLocalFile())
                if path.is_file():
                    event.acceptProposedAction()
                    self.tabs.setCurrentWidget(self.analyze)
                    self.analyze.load_path(path)
                    return
        super().dropEvent(event)
