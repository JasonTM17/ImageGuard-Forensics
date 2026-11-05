"""Tests cho modelbench metrics — boundary-F1, MWU, aggregate."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from imageguard.core.config import load_thresholds
from imageguard.core.enums import Verdict
from imageguard.evaluation.modelbench import (
    _git_snapshot,
    aggregate,
    aggregate_original_controls,
    analyze_sample,
    boundary_f1,
    iou,
    mann_whitney_u,
    noise_sigma_map,
    render_markdown,
    run_bench,
)


def _mask(h, w, rect):
    m = np.zeros((h, w), np.uint8)
    x, y, rw, rh = rect
    cv2.rectangle(m, (x, y), (x + rw, y + rh), 255, -1)
    return m


def test_iou_perfect_and_zero():
    gt = _mask(100, 100, (10, 10, 30, 30))
    assert iou(gt, gt) == 1.0
    far = _mask(100, 100, (60, 60, 30, 30))
    assert iou(far, gt) == 0.0


def test_boundary_f1_tolerates_shift():
    """Mask lệch 4px: IoU thuần thấp nhưng bF1@8 phải cao — đúng mục đích
    của dung sai mép."""
    gt = _mask(100, 100, (20, 20, 40, 40))
    shifted = _mask(100, 100, (24, 24, 40, 40))
    assert iou(shifted, gt) < 0.8
    assert boundary_f1(shifted, gt, tol=8) > 0.85


def test_boundary_f1_empty_pred_is_zero():
    gt = _mask(100, 100, (20, 20, 40, 40))
    assert boundary_f1(np.zeros_like(gt), gt) == 0.0


def test_mwu_separated_vs_same():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 1, 50)
    b = rng.normal(14, 1, 50)
    assert 0 < mann_whitney_u(a, b) < 0.001
    a2 = rng.normal(10, 1, 50)
    b2 = rng.normal(10, 1, 50)
    assert mann_whitney_u(a2, b2) > 0.05


def test_noise_sigma_map_catches_smooth_patch():
    rng = np.random.default_rng(1)
    img = rng.normal(128, 20, (128, 128)).astype(np.float32)
    img[40:88, 40:88] = cv2.GaussianBlur(img[40:88, 40:88], (11, 11), 5)
    sig = noise_sigma_map(img)
    assert sig.shape == (8, 8)
    # block (2..5,2..5) mịn → σ thấp hơn rõ ngoài
    inner = sig[2:6, 2:6].mean()
    outer = np.delete(sig, slice(2, 6), axis=0)[2:6].mean()
    assert inner < outer * 0.6


def test_aggregate_counts():
    samples = [{
        "kind": "erase",
        "methods": {
            "keypoint": {"verdict": "inconclusive", "score": 4,
                          "iou": 0.1, "bf1": 0.3},
            "noise": {"verdict": "no_indication", "score": 0.01,
                       "iou": 0.0, "bf1": 0.0},
        },
        "noise_sigma": {"in_median": 1, "out_median": 2, "mwu_p": 0.5},
    }] * 3
    agg = aggregate(samples)
    cell = agg["erase"]["keypoint"]
    assert cell["n"] == 3 and cell["inconclusive"] == 3
    assert cell["mean_iou"] == pytest.approx(0.1)
    assert cell["detection_rate"] == 0.0


def test_missing_mask_counts_as_zero_in_localization_mean():
    """A missed mask must remain in the localization denominator."""
    gt = _mask(128, 128, (24, 24, 40, 40))
    doc = SimpleNamespace(
        array=np.random.default_rng(5).integers(0, 256, (128, 128, 3),
                                                dtype=np.uint8))

    class FixedMask:
        name = "test"

        def __init__(self, mask):
            self.mask = mask

        def analyze(self, doc, cfg):
            return SimpleNamespace(
                verdict=Verdict.INCONCLUSIVE, score=None, mask=self.mask,
                params_hash=cfg.snapshot_hash())

    cfg = load_thresholds()
    full = analyze_sample(doc, gt, [FixedMask(gt)], cfg)["methods"]["test"]
    missed = analyze_sample(doc, gt, [FixedMask(None)], cfg)["methods"]["test"]
    cell = aggregate([
        {"kind": "erase", "methods": {"test": full}},
        {"kind": "erase", "methods": {"test": missed}},
    ])["erase"]["test"]

    assert missed["iou"] == 0.0
    assert missed["bf1"] == 0.0
    assert cell["mean_iou"] == pytest.approx(0.5)
    assert cell["mean_bf1"] == pytest.approx(0.5)
    assert cell["n_iou_scored"] == 2
    assert cell["n_mask"] == 1


def test_run_bench_records_provenance_and_clean_controls(tmp_path):
    class NoMaskMethod:
        name = "test"

        def analyze(self, doc, cfg):
            return SimpleNamespace(
                verdict=Verdict.INCONCLUSIVE, score=0.01, mask=None,
                params_hash=cfg.snapshot_hash())

    rng = np.random.default_rng(9)
    image = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    gt = _mask(128, 128, (24, 24, 32, 32))
    assert cv2.imwrite(str(tmp_path / "erase_0000_O.png"), image)
    assert cv2.imwrite(str(tmp_path / "erase_0000_F.png"), image)
    assert cv2.imwrite(str(tmp_path / "erase_0000_M.png"), gt)
    (tmp_path / "meta.json").write_text(json.dumps([{
        "id": "erase_0000", "kind": "erase", "format": "png",
        "backend": "classic", "seed": 9, "model_sha256": None,
    }]), encoding="utf-8")

    result = run_bench(tmp_path, [NoMaskMethod()], load_thresholds())
    edited = result["samples"][0]["methods"]["test"]
    control = result["original_controls"]["aggregate"]["test"]

    assert result["experiment_id"].startswith("modelbench-")
    assert len(result["provenance"]["dataset_sha256"]) == 64
    assert len(result["provenance"]["dataset_manifest_sha256"]) == 64
    assert len(result["provenance"]["source_tree_sha256"]) == 64
    assert result["provenance"]["generator_backends"] == ["classic"]
    assert edited["params_hash"] == load_thresholds().snapshot_hash()
    assert edited["iou"] == 0.0
    assert result["aggregate"]["erase"]["test"]["mean_iou"] == 0.0
    assert result["aggregate"]["erase"]["test"]["n_iou_scored"] == 1
    assert control["inconclusive"] == 1
    assert control["n_nonempty_mask"] == 0
    assert control["nonempty_mask_rate"] == 0.0
    assert control["mean_mask_area_ratio"] == 0.0


def test_original_control_report_includes_error_and_not_applicable():
    controls = aggregate_original_controls([
        {"methods": {"test": {
            "verdict": Verdict.ERROR.value,
            "mask_present": False, "mask_status": "missing",
            "mask_area_ratio": 0.0,
        }}},
        {"methods": {"test": {
            "verdict": Verdict.NOT_APPLICABLE.value,
            "mask_present": False, "mask_status": "missing",
            "mask_area_ratio": 0.0,
        }}},
        {"methods": {"test": {
            "verdict": Verdict.INCONCLUSIVE.value,
            "mask_present": True, "mask_status": "shape_mismatch",
            "mask_area_ratio": None,
        }}},
    ])
    report = render_markdown({
        "set": "test", "n": 0, "aggregate": {}, "samples": [],
        "original_controls": {"n": 3, "n_skipped": 0,
                              "aggregate": controls},
    })

    assert controls["test"]["error"] == 1
    assert controls["test"]["not_applicable"] == 1
    assert controls["test"]["n_mask_invalid"] == 1
    assert controls["test"]["n_area_scored"] == 2
    assert "| method | NO_INDICATION | INCONCLUSIVE | SUSPICIOUS | ERROR | NOT_APPLICABLE | non-empty masks | invalid masks | area scored |" in report
    assert "| test | 0 | 1 | 0 | 1 | 1 | 0/3 | 1 | 2/3 |" in report
    lines = report.splitlines()
    header_index = next(index for index, line in enumerate(lines)
                        if line.startswith("| method |"))
    header, separator = lines[header_index:header_index + 2]
    assert header.count("|") == separator.count("|")


def test_mwu_tie_correction_matches_hand_computed():
    """Ties phải hiệu chỉnh phương sai; giá trị tính tay: var=10, p=0.34278."""
    p = mann_whitney_u([1, 1, 2, 2], [1, 2, 2, 3])
    assert p == pytest.approx(0.34278171114791145)


def test_mwu_all_tied_is_unavailable():
    """Một lớp giá trị duy nhất → không có thống kê → N/A (R5)."""
    assert math.isnan(mann_whitney_u([1, 1, 1, 1], [1, 1, 1, 1]))


def test_git_snapshot_uses_repo_root_not_cwd(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    expected = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
    monkeypatch.chdir(tmp_path)
    sha, clean = _git_snapshot()
    assert sha == expected
    assert clean in (True, False)


def test_git_snapshot_sets_timeout_and_utf8(monkeypatch):
    calls = []

    def fake(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return "abc123\n"

    monkeypatch.setattr(subprocess, "check_output", fake)
    sha, _ = _git_snapshot()
    assert sha == "abc123"
    assert len(calls) == 2
    root = str(Path(__file__).resolve().parents[1])
    for _cmd, kwargs in calls:
        assert str(kwargs.get("cwd")) == root
        assert kwargs.get("timeout") is not None
        assert kwargs.get("encoding") == "utf-8"


def test_run_bench_counts_unreadable_edited_images(tmp_path):
    class NoMaskMethod:
        name = "test"

        def analyze(self, doc, cfg):
            return SimpleNamespace(
                verdict=Verdict.INCONCLUSIVE, score=0.01, mask=None,
                params_hash=cfg.snapshot_hash())

    rng = np.random.default_rng(9)
    image = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    gt = _mask(128, 128, (24, 24, 32, 32))
    assert cv2.imwrite(str(tmp_path / "erase_0000_O.png"), image)
    assert cv2.imwrite(str(tmp_path / "erase_0000_M.png"), gt)
    # Cố ý không ghi _F.png → sample không đọc được.
    (tmp_path / "meta.json").write_text(json.dumps([{
        "id": "erase_0000", "kind": "erase", "format": "png",
        "backend": "classic", "seed": 9, "model_sha256": None,
    }]), encoding="utf-8")

    result = run_bench(tmp_path, [NoMaskMethod()], load_thresholds())
    assert result["n"] == 0
    assert result["n_skipped_unreadable"] == 1
    assert "skipped 1 unreadable" in render_markdown(result)


def test_run_bench_json_report_has_no_nan(tmp_path):
    class NoMaskMethod:
        name = "test"

        def analyze(self, doc, cfg):
            return SimpleNamespace(
                verdict=Verdict.INCONCLUSIVE, score=0.01, mask=None,
                params_hash=cfg.snapshot_hash())

    rng = np.random.default_rng(9)
    image = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    # GT 8×8 px → 1 block σ → n_in < 4 → MWU N/A.
    gt = np.zeros((128, 128), np.uint8)
    gt[24:32, 24:32] = 255
    assert cv2.imwrite(str(tmp_path / "erase_0000_O.png"), image)
    assert cv2.imwrite(str(tmp_path / "erase_0000_F.png"), image)
    assert cv2.imwrite(str(tmp_path / "erase_0000_M.png"), gt)
    (tmp_path / "meta.json").write_text(json.dumps([{
        "id": "erase_0000", "kind": "erase", "format": "png",
        "backend": "classic", "seed": 9, "model_sha256": None,
    }]), encoding="utf-8")

    result = run_bench(tmp_path, [NoMaskMethod()], load_thresholds())
    assert result["samples"][0]["noise_sigma"]["mwu_p"] is None
    json.dumps(result, allow_nan=False)  # NaN → None, JSON hợp lệ


def test_analyze_sample_tiny_image_reports_unavailable_noise_stats():
    """Ảnh nhỏ hơn block 16px → không có block nào → noise stats N/A, vẫn
    chấm method và JSON hợp lệ (không crash cv2.resize)."""
    class NoMaskMethod:
        name = "test"

        def analyze(self, doc, cfg):
            return SimpleNamespace(
                verdict=Verdict.INCONCLUSIVE, score=0.01, mask=None,
                params_hash=cfg.snapshot_hash())

    image = np.zeros((14, 14, 3), dtype=np.uint8)
    gt = np.zeros((14, 14), np.uint8)
    gt[2:6, 2:6] = 255
    doc = SimpleNamespace(array=image)
    row = analyze_sample(doc, gt, [NoMaskMethod()], load_thresholds())
    ns = row["noise_sigma"]
    assert ns["in_median"] is None
    assert ns["out_median"] is None
    assert ns["mwu_p"] is None
    json.dumps(row, allow_nan=False)


def test_run_bench_reads_non_ascii_set_dir(tmp_path):
    from imageguard.evaluation.modelforge import generate_set

    class NoMaskMethod:
        name = "test"

        def analyze(self, doc, cfg):
            return SimpleNamespace(
                verdict=Verdict.INCONCLUSIVE, score=0.01, mask=None,
                params_hash=cfg.snapshot_hash())

    out = tmp_path / "bộ_ảnh"
    generate_set(2, seed=1, out_dir=out, size=(128, 128), backend="classic",
                 kinds=("erase", "splice_keep"))
    result = run_bench(out, [NoMaskMethod()], load_thresholds())
    assert result["n"] == 2
    assert result["n_skipped_missing_gt"] == 0
    assert result["n_skipped_unreadable"] == 0


def test_render_markdown_model_hash_list_is_not_python_repr():
    report = render_markdown({
        "set": "test", "n": 0, "aggregate": {}, "samples": [],
        "provenance": {"generator_model_sha256": ["abc123"],
                       "config_hash": "x", "git_sha": "y",
                       "working_tree_clean": True, "dataset_sha256": "z",
                       "dataset_manifest_sha256": "w",
                       "source_tree_sha256": "v", "generator_backends": [],
                       "seeds": [], "python": "3", "opencv": "5",
                       "numpy": "2"},
    })
    assert "`abc123`" in report
    assert "['abc123']" not in report
