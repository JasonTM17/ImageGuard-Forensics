"""Phase 8: report export + stats + robustness battery."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.core.enums import Verdict
from imageguard.evaluation.robustness import run_battery
from imageguard.forensics.registry import load_method
from imageguard.services.analysis_service import AnalysisRequest


@pytest.fixture()
def services(tmp_path):
    from imageguard.ui.app import build_services

    return build_services(db_path=tmp_path / "t.db",
                          artifacts_root=tmp_path / "art")


@pytest.fixture()
def forged(tmp_path):
    from imageguard.evaluation.synthforge import generate_set

    out = tmp_path / "s"
    generate_set(2, seed=11, out_dir=out, size=(256, 256), tier="medium")
    return sorted(out.glob("*_F.png"))


def test_report_markdown_export(services, forged, tmp_path):
    analysis, _ = services
    rep = analysis.analyze(AnalysisRequest(path=forged[0], methods=["ela"]))
    aid = rep.report_id
    from imageguard.services.report_service import ReportService

    store = analysis.store
    rs = ReportService(store)
    out = rs.export_markdown(aid, tmp_path / "r.md")
    assert out is not None and out.exists()
    md = out.read_text(encoding="utf-8")
    assert "Báo cáo phân tích" in md
    assert "không khẳng định" in md  # disclaimer bắt buộc
    assert "score" in md or "ela" in md
    # Không chứa xác suất tự chế
    assert "%" not in md.split("Tổng quan")[0]


def test_stats_reads_db(services, forged):
    analysis, history = services
    for p in forged[:2]:
        analysis.analyze(AnalysisRequest(path=p, methods=["ela"]))
    s = history.stats()
    assert s["n_analyses"] == 2
    assert "thống kê phiên" in s["label"]
    assert sum(s["by_verdict"].values()) == 2


def test_robustness_battery_shape(services, forged, ts):
    m = load_method("keypoint")
    res = run_battery(forged[:1], [m], ts,
                      distortions={"none": lambda x: x,
                                   "jpeg_70": lambda i: __import__(
                                       "imageguard.evaluation.robustness",
                                       fromlist=["_jpeg"])._jpeg(i, 70)})
    assert m.name in res
    assert set(res[m.name]) == {"none", "jpeg_70"}
    for d in res[m.name].values():
        assert d["n"] == 1
        assert d["n_suspicious"] + d["n_inconclusive"] <= 1


def test_fusion_aux_corroboration():
    from imageguard.core.enums import MethodName, Verdict
    from imageguard.core.models import AlgorithmResult
    from imageguard.forensics.fusion import combine

    def r(method, verdict):
        return AlgorithmResult(method=method, verdict=verdict)

    # ≥2 aux INCONCLUSIVE → corroboration signal, verdict vẫn INCONCLUSIVE
    v, sig, _ = combine([
        r(MethodName.COPY_MOVE_KEYPOINT, Verdict.NO_INDICATION),
        r(MethodName.ELA, Verdict.INCONCLUSIVE),
        r(MethodName.NOISE_INCONSISTENCY, Verdict.INCONCLUSIVE),
    ])
    assert v == Verdict.INCONCLUSIVE
    assert any("aux_corroboration" in s for s in sig)
    # Aux không bao giờ kéo lên SUSPICIOUS
    v2, _, _ = combine([
        r(MethodName.ELA, Verdict.INCONCLUSIVE),
        r(MethodName.FREQUENCY_ANALYSIS, Verdict.INCONCLUSIVE),
        r(MethodName.SPLICING_CUES, Verdict.INCONCLUSIVE),
    ])
    assert v2 == Verdict.INCONCLUSIVE


def test_keypoint_pairs_artifact(services, forged):
    analysis, history = services
    rep = analysis.analyze(AnalysisRequest(path=forged[0], methods=["keypoint"]))
    kp_res = next((m for m in rep.methods if m.method.value == "copy_move_keypoint"), None)
    assert kp_res is not None
    if kp_res.score and kp_res.score > 0:
        assert "pairs_image" in kp_res.evidence
        arts = history.get(rep.report_id)["artifacts"]
        assert any(a["kind"] == "pairs" for a in arts)
