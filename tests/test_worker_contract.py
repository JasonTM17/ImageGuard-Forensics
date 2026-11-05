"""Worker contract — signals marshal đúng, cancel cooperative, serialized pool."""

from __future__ import annotations

import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from imageguard.core.models import ForensicReport
from imageguard.services.analysis_service import AnalysisRequest
from imageguard.ui.workers import AnalysisWorker, make_pool


@pytest.fixture()
def services(tmp_path):
    from imageguard.ui.app import build_services

    return build_services(db_path=tmp_path / "t.db",
                          artifacts_root=tmp_path / "art")


@pytest.fixture()
def forged_png(tmp_path):
    """Ảnh nhỏ có copy-move để keypoint chạy nhanh."""
    from imageguard.evaluation.synthforge import generate_set

    out = tmp_path / "s"
    generate_set(1, seed=7, out_dir=out, size=(256, 256), tier="medium")
    return next(out.glob("*_F.png"))


def _collect(worker):
    out = {"result": None, "failed": None, "prog": []}
    worker.signals.result.connect(lambda r: out.update(result=r))
    worker.signals.failed.connect(lambda m: out.update(failed=m))
    worker.signals.progress.connect(lambda f, s: out["prog"].append((f, s)))
    return out


def test_worker_emits_result(qtbot, services, forged_png):
    analysis, _ = services
    req = AnalysisRequest(path=forged_png, methods=["keypoint"])
    w = AnalysisWorker(analysis, req)
    out = _collect(w)
    pool = make_pool()
    pool.start(w)
    assert pool.waitForDone(30000)
    qtbot.wait(100)  # cho queued signal deliver
    assert out["failed"] is None
    assert isinstance(out["result"], ForensicReport)
    assert out["prog"], "worker phải emit progress"


def test_worker_cancel_is_cooperative(qtbot, services, forged_png):
    analysis, _ = services
    cancel = threading.Event()
    req = AnalysisRequest(path=forged_png, methods=["keypoint", "block"],
                          cancel=cancel)
    w = AnalysisWorker(analysis, req)
    out = _collect(w)
    pool = make_pool()
    pool.start(w)
    cancel.set()  # cancel ngay — worker phải thoát sớm, không treo
    assert pool.waitForDone(30000)
    qtbot.wait(100)
    assert out["failed"] is None  # cancel = graceful, không phải exception


def test_pool_serializes(qtbot, services, forged_png):
    """maxThreadCount=1 — hai worker không chạy song song."""
    analysis, _ = services
    pool = make_pool()
    assert pool.maxThreadCount() == 1
    order = []
    for i in range(2):
        req = AnalysisRequest(path=forged_png, methods=["ela"])
        w = AnalysisWorker(analysis, req)
        w.signals.result.connect(lambda r, i=i: order.append(i))
        pool.start(w)
    assert pool.waitForDone(60000)
    qtbot.wait(100)
    assert order == [0, 1]


def test_worker_never_crashes_on_garbage(qtbot, services, tmp_path):
    analysis, _ = services
    bad = tmp_path / "x.png"
    bad.write_bytes(b"\xff\xd8\xff corrupt jpeg bytes")
    req = AnalysisRequest(path=bad, methods=["keypoint"])
    w = AnalysisWorker(analysis, req)
    out = _collect(w)
    pool = make_pool()
    pool.start(w)
    assert pool.waitForDone(15000)
    qtbot.wait(100)
    # safe_open trả ErrorResult qua result signal — KHÔNG failed
    assert out["failed"] is None
    assert out["result"] is not None
