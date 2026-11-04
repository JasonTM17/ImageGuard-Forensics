"""Regression tests cho các bug được sửa trong audit 2026-11-04.

Mỗi test gắn với finding ID (F-A1, F-B2, ...) — test phải FAIL trên code cũ
và PASS trên code đã sửa.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.storage.artifacts import ArtifactStore


# ---------- F-A1: EXIF orientation swap ----------

def _exif_jpeg(path: Path, orientation: int, w: int = 128, h: int = 64) -> Path:
    from PIL import Image

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif[0x0112] = orientation  # Orientation tag
    Image.fromarray(img).save(path, exif=exif, quality=90)
    return path


def test_exif_orientation_dims_match_array(tmp_path: Path) -> None:
    """F-A1: doc.width/height phải khớp doc.array sau exif_transpose."""
    from imageguard.core.io import safe_open

    p = _exif_jpeg(tmp_path / "rot6.jpg", 6)
    doc = safe_open(p)
    assert not hasattr(doc, "code")
    assert doc.width == doc.array.shape[1]
    assert doc.height == doc.array.shape[0]
    assert (doc.width, doc.height) == (64, 128)  # 128x64 xoay 90°


# ---------- F-A2: integral image off-by-one ----------

def test_noise_map_sigma_matches_direct_blocks(tmp_path: Path, ts) -> None:
    """F-A2: σ per block từ integral image phải bằng std trực tiếp."""
    from imageguard.forensics.auxiliary.noise_map import NoiseMap

    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    doc = ImageDocument(array=img, width=256, height=256, channels=3,
                        image_format="png", sha256="x", file_bytes=1,
                        file_name="a.png")
    r = NoiseMap().analyze(doc, ts)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rr = gray - cv2.medianBlur(gray, int(ts.value("noise", "median_k"))).astype(np.float32)
    bs = int(ts.value("noise", "block"))
    direct = np.array([[rr[y:y + bs, x:x + bs].std()
                        for x in range(0, 256, bs)] for y in range(0, 256, bs)])
    assert abs(r.evidence["sigma_median"] - float(np.median(direct))) < 0.02


# ---------- F-A3/F-C15: NOISE_LOCAL là aux signal ----------

def test_noise_local_counts_as_aux_corroboration(ts) -> None:
    from imageguard.forensics.fusion import AUX_METHODS, combine

    assert MethodName.NOISE_LOCAL in AUX_METHODS
    results = [
        AlgorithmResult(method=MethodName.NOISE_LOCAL,
                        verdict=Verdict.INCONCLUSIVE),
        AlgorithmResult(method=MethodName.ELA, verdict=Verdict.INCONCLUSIVE),
    ]
    overall, signals, _ = combine(results)
    assert overall == Verdict.INCONCLUSIVE
    assert any("aux_corroboration" in s and "noise_local" in s for s in signals)


# ---------- F-B1/F-B2/F-B16: artifact store guards ----------

def test_artifact_store_traversal_guards(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "art")

    # F-B2: startswith prefix check bị bypass bằng sibling dir
    with pytest.raises(ValueError):
        store.abs_path("../art-evil/secret.txt")
    with pytest.raises(ValueError):
        store.read("../art-evil/secret.txt")

    # F-B1: delete_analysis không nhận '..' / absolute
    with pytest.raises(ValueError):
        store.delete_analysis("..")
    with pytest.raises(ValueError):
        store.delete_analysis("D:/evil")

    # F-B16: filename validation
    with pytest.raises(ValueError):
        store.put_bytes("ok", "a.b.png", b"x")
    with pytest.raises(ValueError):
        store.put_bytes("ok", "../x.png", b"x")


def test_delete_analysis_only_removes_own_dir(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "art")
    store.put_bytes("id-1", "report.json", b"{}")
    store.put_bytes("id-2", "report.json", b"{}")
    store.delete_analysis("id-1")
    assert not (tmp_path / "art" / "id-1").exists()
    assert (tmp_path / "art" / "id-2").exists()  # không đụng id khác


# ---------- F-B3: spec→MethodName mapping khi method crash/cancel ----------

@pytest.mark.parametrize("spec,expected", [
    ("keypoint", MethodName.COPY_MOVE_KEYPOINT),
    ("block", MethodName.COPY_MOVE_BLOCK),
    ("ela", MethodName.ELA),
    ("noise", MethodName.NOISE_INCONSISTENCY),
    ("noise_local", MethodName.NOISE_LOCAL),
    ("freq", MethodName.FREQUENCY_ANALYSIS),
    ("splicing", MethodName.SPLICING_CUES),
])
def test_error_result_preserves_method(spec: str, expected: MethodName) -> None:
    from imageguard.services.analysis_service import _cancelled_result, _error_result

    assert _error_result(spec, RuntimeError("boom")).method == expected
    assert _cancelled_result(spec).method == expected


# ---------- F-B5/F-B7/F-B17: runner robustness ----------

class _FakeAdapter:
    name = "fake"

    def __init__(self, samples) -> None:
        self._s = samples

    def version(self) -> str:
        return "fake-1"

    def samples(self, split: str = "dev"):
        return iter(self._s)


class _NaMethod:
    name = "na_method"

    def analyze(self, doc, cfg, cancel=None):
        return AlgorithmResult(method=MethodName.ELA,
                               verdict=Verdict.NOT_APPLICABLE)


class _OkMethod:
    name = "ok_method"

    def analyze(self, doc, cfg, cancel=None):
        return AlgorithmResult(method=MethodName.COPY_MOVE_KEYPOINT,
                               verdict=Verdict.NO_INDICATION)


def _doc(path: Path, w: int = 64, h: int = 64) -> ImageDocument:
    img = np.zeros((h, w, 3), np.uint8)
    path.write_bytes(cv2.imencode(".png", img)[1].tobytes())
    return ImageDocument(array=img, width=w, height=h, channels=3,
                         image_format="png", sha256="x", file_bytes=1,
                         file_name=path.name)


def test_runner_excludes_wrong_size_mask(tmp_path: Path, ts) -> None:
    """F-B5: mask sai dims → exclude + đếm, KHÔNG giết cả eval."""
    from imageguard.evaluation.datasets import Sample
    from imageguard.evaluation.runner import run_eval

    _doc(tmp_path / "a.png", 64, 64)
    bad_mask = tmp_path / "bad_mask.png"
    bad_mask.write_bytes(cv2.imencode(".png", np.zeros((32, 32), np.uint8))[1].tobytes())
    adapter = _FakeAdapter([Sample(tmp_path / "a.png", bad_mask, "forged", "c", "a")])
    r = run_eval(adapter, [_OkMethod()], ts, "dev")
    d = r.per_method["ok_method"]
    assert d["n_excluded"] == 1
    assert d["n"] == 0


def test_runner_excludes_not_applicable_from_metrics(tmp_path: Path, ts) -> None:
    """F-B17: NOT_APPLICABLE/ERROR KHÔNG được tính thành authentic."""
    from imageguard.evaluation.datasets import Sample
    from imageguard.evaluation.runner import run_eval

    _doc(tmp_path / "a.png", 64, 64)
    adapter = _FakeAdapter([Sample(tmp_path / "a.png", None, "authentic", "c", "a")])
    r = run_eval(adapter, [_NaMethod()], ts, "dev")
    d = r.per_method["na_method"]
    assert d["n_not_applicable"] == 1
    assert d["n"] == 0
    assert d["tp"] == 0 and d["fp"] == 0


def test_runner_experiment_ids_unique(tmp_path: Path, ts) -> None:
    """F-B7: experiment_id không được trùng giữa 2 run."""
    from imageguard.evaluation.datasets import Sample
    from imageguard.evaluation.runner import run_eval

    _doc(tmp_path / "a.png", 64, 64)
    adapter = _FakeAdapter([Sample(tmp_path / "a.png", None, "authentic", "c", "a")])
    r1 = run_eval(adapter, [_OkMethod()], ts, "dev")
    r2 = run_eval(adapter, [_OkMethod()], ts, "dev")
    assert r1.experiment_id != r2.experiment_id


# ---------- F-B8: FK pragma per connection ----------

def test_foreign_keys_on_all_connections(tmp_path: Path) -> None:
    from sqlalchemy import text

    from imageguard.storage.db import make_session_factory

    sf = make_session_factory(tmp_path / "fk.db")
    for _ in range(2):  # mỗi session = connection khác từ pool
        with sf() as s:
            assert s.execute(text("PRAGMA foreign_keys")).scalar() == 1


# ---------- F-B6: comofod mask preference ----------

def test_comofod_prefers_binary_mask(tmp_path: Path) -> None:
    from imageguard.evaluation.comofod import CoMoFoDAdapter

    for name in ("001_F.png", "001_B.png", "001_C.png", "001_M.png", "001_O.png"):
        (tmp_path / name).write_bytes(
            cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))[1].tobytes())
    ad = CoMoFoDAdapter(tmp_path)
    samples = ad._discover()
    forged = [s for s in samples if s.label == "forged"][0]
    assert forged.gt_mask_path.name == "001_B.png"


# ---------- F-B4: modelbench render None guard ----------

def test_modelbench_render_none_sigma() -> None:
    from imageguard.evaluation.modelbench import render_markdown

    res = {"set": "fake", "n": 1, "samples": [{
        "id": "s1", "kind": "erase",
        "noise_sigma": {"in_median": None, "out_median": None,
                        "mwu_p": float("nan")}}],
        "aggregate": {"erase": {"m": {"n": 1, "suspicious": 0,
                                      "inconclusive": 0, "detection_rate": 0.0,
                                      "mean_iou": None, "mean_bf1": None}}}}
    md = render_markdown(res)
    assert "—" in md and "None" not in md


# ---------- F-B10: robustness empty distortions ----------

def test_robustness_empty_distortions_means_none(tmp_path: Path, ts) -> None:
    from imageguard.evaluation.robustness import run_battery

    out = run_battery([], [_OkMethod()], ts, distortions={})
    assert out == {"ok_method": {}}


# ---------- F-A7: DCT feature int32 ----------

def test_dct_features_int32_no_overflow() -> None:
    from imageguard.forensics.block.dct_blocks import extract_features

    rng = np.random.default_rng(2)
    # DC ~255*256 = 65280 > int16 max, nhưng std block > flat_std_eps để giữ lại
    img = np.clip(255 - rng.normal(0, 10, (256, 256)), 0, 255).astype(np.uint8)
    feats, _ = extract_features(img, B=256, n_coeffs=4)
    assert feats.dtype == np.int32
    assert feats[:, 0].max() > 32767


# ---------- F-B11: persist ghi wall-clock duration ----------

def test_persist_stores_wall_clock_duration(tmp_path: Path) -> None:
    from imageguard.services.analysis_service import AnalysisService
    from imageguard.storage.db import make_session_factory
    from imageguard.storage.repository import AnalysisRepository, ImageRepository
    from imageguard.storage.artifacts import ArtifactStore

    sf = make_session_factory(tmp_path / "t.db")
    svc = AnalysisService(
        images=ImageRepository(sf), analyses=AnalysisRepository(sf),
        store=ArtifactStore(tmp_path / "art"))
    doc = _doc(tmp_path / "d.png", 64, 64)
    from imageguard.core.models import ForensicReport

    report = ForensicReport(report_id="r1", image_sha256="x",
                            image_name="d.png", image_width=64, image_height=64,
                            created_at="2026-01-01T00:00:00+00:00",
                            tool_version="t")
    svc._persist(report, doc, [], wall_ms=1234)
    row = svc.analyses.get("r1")
    assert row is not None
    # duration wall-clock lưu vào DB
    from sqlalchemy import text

    with sf() as s:
        v = s.execute(text("SELECT duration_ms FROM analyses WHERE id='r1'")).scalar()
    assert v == 1234


# ---------- F-B15: --version trên subcommand ----------

def test_cli_version_on_subcommand() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "imageguard", "eval", "--version"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert out.returncode == 0
    assert "ImageGuard Forensics v" in out.stdout


# ---------- F-B13: modelforge clear error cho ảnh nhỏ ----------

def test_modelforge_rand_region_small_image() -> None:
    from imageguard.evaluation.modelforge import _rand_region

    with pytest.raises(ValueError, match="quá nhỏ"):
        _rand_region(np.random.default_rng(0), (8, 8))


# ---------- F-B12: SynthAdapter tương thích modelforge manifest ----------

def test_synth_adapter_accepts_modelforge_manifest(tmp_path: Path) -> None:
    from imageguard.evaluation.datasets import SynthAdapter

    meta = [{"id": "m1", "seed": 2026, "kind": "erase", "backend": "lama",
             "spec": {"kind": "erase", "shape": "ellipse", "rect": [0, 0, 8, 8]},
             "format": "png"}]
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for name in ("m1_F.png", "m1_M.png"):
        (tmp_path / name).write_bytes(
            cv2.imencode(".png", np.zeros((16, 16, 3), np.uint8))[1].tobytes())
    ad = SynthAdapter(tmp_path)
    found = list(ad.samples("dev")) + list(ad.samples("test"))
    forged = [s for s in found if s.label == "forged"]
    assert len(forged) == 1
    assert forged[0].category.startswith("ellipse/")


# ---------- F-A6: local_k range validation ----------

def test_local_k_out_of_range_rejected() -> None:
    raw = {"noise": {"local_k": {"value": 7, "source": "test", "range": [3, 5]}}}
    ts = ThresholdSet(raw=raw, config_hash="")
    with pytest.raises(ValueError, match="ngoài range"):
        ts.value("noise", "local_k")


# ---------- F-C1/F-C2/F-C8: UI stale-result guards ----------

@pytest.mark.skipif(not __import__("importlib").util.find_spec("PySide6"),
                    reason="PySide6 không cài")
def test_analyze_screen_clears_results_on_new_load(qtbot, tmp_path, ts):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from imageguard.ui.screens.analyze import AnalyzeScreen
    from imageguard.services.analysis_service import AnalysisService

    svc = AnalysisService(cfg=ts)
    screen = AnalyzeScreen(svc)
    qtbot.addWidget(screen)

    img = np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    p = tmp_path / "a.png"
    p.write_bytes(cv2.imencode(".png", img)[1].tobytes())

    # Giả lập kết quả cũ
    screen._report = object()
    screen.banner.show()
    screen.btn_export.show()
    from imageguard.ui.widgets.verdict_panel import MethodCard

    screen.cards_layout.addWidget(MethodCard(AlgorithmResult(
        method=MethodName.ELA, verdict=Verdict.NO_INDICATION)))
    screen.cards_layout.addStretch(1)

    screen.load_path(p)
    assert screen._report is None
    assert not screen.banner.isVisible()
    assert not screen.btn_export.isVisible()
    assert screen.cards_layout.count() == 0
    assert screen._path == p


@pytest.mark.skipif(not __import__("importlib").util.find_spec("PySide6"),
                    reason="PySide6 không cài")
def test_overlay_shape_mismatch_does_not_crash(qtbot, tmp_path, ts):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from imageguard.ui.screens.analyze import AnalyzeScreen
    from imageguard.services.analysis_service import AnalysisService
    from imageguard.core.models import ForensicReport

    svc = AnalysisService(cfg=ts)
    screen = AnalyzeScreen(svc)
    qtbot.addWidget(screen)

    img = np.random.default_rng(1).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    p = tmp_path / "b.png"
    p.write_bytes(cv2.imencode(".png", img)[1].tobytes())
    screen.load_path(p)

    report = ForensicReport(
        report_id="r", image_sha256="x", image_name="b.png",
        image_width=128, image_height=128, created_at="t", tool_version="t",
        methods=[AlgorithmResult(
            method=MethodName.COPY_MOVE_KEYPOINT, verdict=Verdict.NO_INDICATION,
            mask=np.zeros((32, 32), np.uint8))])
    screen._on_result(report)
    # chọn overlay mask 32x32 ≠ ảnh 128x128 → không raise, hiện cảnh báo
    screen.overlay_box.setCurrentIndex(1)
    assert "Overlay bỏ qua" in screen.lbl_err.text()
