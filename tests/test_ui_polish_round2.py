"""UI polish round 2 — overlay nhãn VN, trạng thái hủy tăng cường."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ForensicReport
from imageguard.ui import strings
from imageguard.ui import workers as workers_mod


def _combo_texts(combo) -> list[str]:
    return [combo.itemText(i) for i in range(combo.count())]


@pytest.fixture()
def services(tmp_path: Path):
    from imageguard.ui.app import build_services

    return build_services(db_path=tmp_path / "t.db", artifacts_root=tmp_path / "art")


def test_overlay_combo_no_raw_method_ids_or_english_mask(qtbot, services, tmp_path, ts):
    from imageguard.services.analysis_service import AnalysisService
    from imageguard.ui.screens.analyze import AnalyzeScreen

    svc = AnalysisService(cfg=ts)
    screen = AnalyzeScreen(svc)
    qtbot.addWidget(screen)

    img = np.random.default_rng(3).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    p = tmp_path / "overlay.png"
    p.write_bytes(cv2.imencode(".png", img)[1].tobytes())
    screen.load_path(p)

    pairs = np.zeros((64, 64, 3), dtype=np.uint8)
    report = ForensicReport(
        report_id="r",
        image_sha256="x" * 64,
        image_name="overlay.png",
        image_width=64,
        image_height=64,
        created_at="t",
        tool_version="0.1.0",
        methods=[
            AlgorithmResult(
                method=MethodName.COPY_MOVE_KEYPOINT,
                verdict=Verdict.NO_INDICATION,
                mask=np.zeros((64, 64), np.uint8),
                heatmap=np.zeros((64, 64), dtype=np.float32),
                evidence={"pairs_image": pairs},
            )
        ],
    )
    screen._on_result(report)

    texts = _combo_texts(screen.overlay_box)
    assert texts[0] == strings.OVERLAY_NONE
    joined = " | ".join(texts)
    assert "copy_move_keypoint" not in joined
    assert "Mask" not in joined
    assert "Heatmap" not in joined
    friendly = strings.METHOD_LABEL["copy_move_keypoint"]
    assert any(friendly in t for t in texts[1:])


def test_enhance_cancel_sets_status_and_idle_controls(
    qtbot, services, valid_png, monkeypatch,
):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    e = w.enhance

    real_derive = workers_mod.derive

    def slow_derive(image, kind, params):
        time.sleep(0.6)
        return real_derive(image, kind, params)

    monkeypatch.setattr(workers_mod, "derive", slow_derive)

    e.load_path(valid_png)
    assert e.btn_run.isEnabled()
    e.btn_run.click()
    qtbot.waitUntil(lambda: e._worker is not None, timeout=3000)
    e.btn_cancel.click()
    qtbot.waitUntil(
        lambda: e._worker is None and not e.btn_cancel.isEnabled(),
        timeout=30000,
    )

    assert e.lbl_status.text() == strings.STATUS_ENHANCE_CANCELLED
    assert e.btn_run.isEnabled()
    assert not e.btn_cancel.isEnabled()
    assert e.combo_op.isEnabled()
    assert e.progress.minimum() == 0 and e.progress.maximum() == 100
    assert e.progress.value() == 0
