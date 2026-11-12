"""Main window must not show the removed fixed disclaimer bar (offscreen)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from imageguard.ui import strings

_REMOVED_MAIN_DISCLAIMER = (
    "Công cụ phát hiện dấu hiệu nghi ngờ — không kết luận ảnh bị can thiệp; "
    "ELA/EXIF/noise chỉ là bằng chứng khám phá."
)
_REMOVED_SNIPPET = _REMOVED_MAIN_DISCLAIMER[:24]


def _widget_texts(root) -> list[str]:
    from PySide6.QtWidgets import QFrame, QLabel

    texts: list[str] = []
    for w in (*root.findChildren(QLabel), *root.findChildren(QFrame)):
        if isinstance(w, QLabel):
            texts.append(w.text())
        elif isinstance(w, QFrame) and w.styleSheet():
            for child in w.findChildren(QLabel):
                texts.append(child.text())
    return texts


@pytest.fixture()
def services(tmp_path):
    from imageguard.ui.app import build_services

    return build_services(db_path=tmp_path / "t.db", artifacts_root=tmp_path / "art")


def test_main_window_builds_without_disclaimer_bar(qtbot, services):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)

    lay = w.centralWidget().layout()
    assert lay is not None
    assert lay.count() == 1
    assert w.tabs is lay.itemAt(0).widget()


def test_no_removed_disclaimer_text_anywhere_in_main_window(qtbot, services):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)

    blob = "\n".join(_widget_texts(w))
    assert _REMOVED_SNIPPET not in blob
    assert _REMOVED_MAIN_DISCLAIMER not in blob


def test_analyze_verdict_banner_still_uses_integrity_wording(qtbot, services):
    from imageguard.core.enums import Verdict
    from imageguard.ui.main_window import MainWindow
    from imageguard.ui.widgets.verdict_panel import VerdictBanner

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    a = w.analyze

    assert isinstance(a.banner, VerdictBanner)
    a.banner.set_verdict(Verdict.NO_INDICATION, [])
    a.banner.show()

    verdict_text = a.banner._label.text()
    assert verdict_text == strings.VERDICT_LABEL["no_indication"]
    assert verdict_text in strings.VERDICT_LABEL.values()
    assert "kết luận" not in verdict_text.lower()
