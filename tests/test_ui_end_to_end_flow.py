"""Offscreen end-to-end: Analyze → export → History/Stats → Enhance (classical).

Drives MainWindow with a seeded synthforge copy-move PNG under tmp_path,
real AnalysisWorker/EnhanceWorker, bounded qtbot.waitUntil (no sleep).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog, QLabel, QMessageBox

from imageguard.evaluation.synthforge import generate_set
from imageguard.ui import strings
from imageguard.ui.widgets.verdict_panel import MethodCard

_FORBIDDEN_EN = (
    "forged",
    "fake",
    "authentic",
    "percent confidence",
    "% confidence",
)
_VERDICT_TEXTS = frozenset(strings.VERDICT_LABEL.values())
_RE_WIN_ABS_PATH = re.compile(
    r"[A-Za-z]:(?:(?:\\[^\"'\n\r]+)+|(?:/[^/\"'\n\r]+)+)"
)
_RE_POSIX_ABS_PATH = re.compile(r"/(?:home|Users|tmp|var|etc)(?:/[^/\"'\n\r]+)+")


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    """Isolated DB + artifacts — never writes repo ``data/``."""
    root = tmp_path / "session_data"
    root.mkdir()
    return root


@pytest.fixture()
def services(data_root: Path):
    from imageguard.ui.app import build_services

    return build_services(
        db_path=data_root / "imageguard.db",
        artifacts_root=data_root / "artifacts",
    )


@pytest.fixture()
def synth_copy_move(tmp_path: Path) -> Path:
    """Small forged PNG (seed 7, 256² medium) for fast keypoint/block."""
    out = tmp_path / "forge_out"
    generate_set(n=1, seed=7, out_dir=out, size=(256, 256), tier="medium")
    forged = next(out.glob("*_F.png"))
    assert forged.is_file()
    return forged


@pytest.fixture()
def window(qtbot, services):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    w.show()
    return w


def _select_kp_block_ela(analyze) -> None:
    analyze.chk_kp.setChecked(True)
    analyze.chk_bl.setChecked(True)
    analyze.chk_ela.setChecked(True)
    for name in ("chk_noise", "chk_nloc", "chk_freq", "chk_spl"):
        getattr(analyze, name).setChecked(False)
    assert analyze.chk_kp.isChecked()
    assert analyze.chk_bl.isChecked()
    assert analyze.chk_ela.isChecked()


def _assert_no_forbidden(text: str) -> None:
    low = text.lower()
    for token in _FORBIDDEN_EN:
        assert token not in low, f"forbidden wording: {token!r}"


def _assert_no_host_paths(text: str, tmp_path: Path) -> None:
    assert str(tmp_path.resolve()) not in text
    assert _RE_WIN_ABS_PATH.search(text) is None
    assert _RE_POSIX_ABS_PATH.search(text) is None


def _card_texts(analyze) -> str:
    parts: list[str] = []
    for card in analyze.cards_inner.findChildren(MethodCard):
        for lbl in card.findChildren(QLabel):
            parts.append(lbl.text())
    return "\n".join(parts)


def test_analyze_export_history_stats_enhance_flow(
    qtbot,
    window,
    synth_copy_move: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    a = window.analyze
    _select_kp_block_ela(a)
    a.load_path(synth_copy_move)
    assert a.btn_run.isEnabled()

    a.btn_run.click()
    qtbot.waitUntil(
        lambda: a._worker is None
        and a.banner.isVisible()
        and a.btn_export.isVisible(),
        timeout=60_000,
    )

    report = a._report
    assert report is not None
    expected_banner = strings.VERDICT_LABEL[report.overall_verdict.value]
    assert a.banner._label.text() == expected_banner
    _assert_no_forbidden(a.banner._label.text())

    cards = a.cards_inner.findChildren(MethodCard)
    method_values = {m.method.value for m in report.methods}
    assert {"copy_move_keypoint", "copy_move_block", "ela"}.issubset(method_values)
    assert len(cards) >= 3
    card_blob = _card_texts(a)
    _assert_no_forbidden(card_blob)
    for vtxt in _VERDICT_TEXTS:
        if vtxt in card_blob:
            break
    else:
        pytest.fail("no closed-enum verdict label on method cards")
    for method in report.methods:
        vlabel = strings.VERDICT_LABEL[method.verdict.value]
        assert vlabel in card_blob

    overlay_n = a.overlay_box.count()
    assert overlay_n >= 2
    for idx in range(overlay_n):
        a.overlay_box.setCurrentIndex(idx)
        qtbot.wait(10)

    verdict_md = strings.VERDICT_LABEL[report.overall_verdict.value]
    md_path = tmp_path / "e2e_report.md"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(md_path), ""),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args, **kwargs: QMessageBox.StandardButton.Ok,
    )
    a.btn_export.click()
    assert md_path.is_file()
    md_text = md_path.read_text(encoding="utf-8")
    assert verdict_md in md_text
    _assert_no_forbidden(md_text)
    _assert_no_host_paths(md_text, tmp_path)
    assert report.report_id in md_text

    window.tabs.setCurrentIndex(2)
    qtbot.waitUntil(lambda: window.history.table.rowCount() == 1, timeout=10_000)
    hist = window.history
    assert hist.table.item(0, 1).text() == synth_copy_move.name
    hist_verdict = hist.table.item(0, 2).text()
    assert hist_verdict == expected_banner
    assert hist_verdict in _VERDICT_TEXTS

    window.tabs.setCurrentIndex(3)
    qtbot.waitUntil(
        lambda: "1" in window.stats.lbl_head.text()
        or window.stats.table.rowCount() >= 1,
        timeout=10_000,
    )
    stats_blob = window.stats.lbl_head.text() + "\n"
    for r in range(window.stats.table.rowCount()):
        vitem = window.stats.table.item(r, 0)
        if vitem:
            stats_blob += vitem.text() + "\n"
    assert expected_banner in stats_blob or window.stats.table.rowCount() >= 1

    window.tabs.setCurrentIndex(1)
    e = window.enhance
    assert e.combo_op.currentData() == "unsharp"
    e.load_path(synth_copy_move)
    assert e.btn_run.isEnabled()
    e.btn_run.click()
    qtbot.waitUntil(
        lambda: e._worker is None and e.slider_blend.isEnabled(),
        timeout=30_000,
    )
    assert e._derived is not None
    assert e.canvas._overlay is not None
