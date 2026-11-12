"""Smoke test cho scripts/bench_methods — cấu trúc JSON, không assert thời gian."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "scripts" / "bench_methods.py"


def _load_bench_module():
    spec = importlib.util.spec_from_file_location("bench_methods", BENCH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench_mod():
    return _load_bench_module()


def test_quick_benchmark_json_shape(bench_mod, tmp_path: Path) -> None:
    data = bench_mod.run_benchmark(
        seed=7,
        sizes=(64, 96),
        repetitions=1,
        work_dir=tmp_path / "assets",
    )
    assert data["meta"]["seed"] == 7
    assert data["meta"]["repetitions"] == 1
    assert data["meta"]["sizes"] == [64, 96]
    assert "per_method" in data
    assert "analysis_service_no_persist" in data
    for spec in data["meta"]["method_specs"]:
        assert spec in data["per_method"]
        for side in ("64", "96"):
            cell = data["per_method"][spec][side]
            assert "samples_s" in cell and len(cell["samples_s"]) == 1
            assert "min_s" in cell and "median_s" in cell
    for side in ("64", "96"):
        svc = data["analysis_service_no_persist"][side]
        assert svc["methods"] == ["keypoint", "block", "ela"]
        assert len(svc["samples_s"]) == 1


def test_cli_quick_main(bench_mod, tmp_path: Path) -> None:
    out = tmp_path / "bench_quick.json"
    rc = bench_mod.main(["--quick", "--out", str(out), "--work-dir", str(tmp_path / "w")])
    assert rc == 0
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert '"per_method"' in text
    assert '"1024"' not in text  # quick không chạy size lớn


def test_invalid_repetitions_raises(bench_mod, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repetitions"):
        bench_mod.run_benchmark(
            repetitions=0, sizes=(64,), work_dir=tmp_path / "x"
        )
