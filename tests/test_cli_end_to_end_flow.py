"""Subprocess end-to-end: analyze, ingest once, eval synth (real `python -m imageguard`)."""

from __future__ import annotations

import copy
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from imageguard.core.enums import Verdict
from imageguard.evaluation.datasets import split_of
from imageguard.evaluation.synthforge import ForgeSpec, _base_image, forge, generate_set

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
_PYTHON = Path(r"D:\ImageGuard Forensics\.venv\Scripts\python.exe")
_FORBIDDEN_OUT = ("forged", "fake", "authentic")
_VOLATILE_TOP = frozenset({"report_id", "created_at"})


def _env() -> dict[str, str]:
    env = os.environ.copy()
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(_REPO_SRC) if not prior else os.pathsep.join((str(_REPO_SRC), prior))
    )
    return env


def _run(
    tmp_path: Path,
    args: list[str],
    *,
    timeout: float = 45.0,
) -> subprocess.CompletedProcess[str]:
    exe = _PYTHON if _PYTHON.is_file() else sys.executable
    return subprocess.run(
        [str(exe), "-m", "imageguard", *args],
        cwd=tmp_path,
        env=_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _jpeg_copy_move(tmp_path: Path, seed: int = 31) -> Path:
    rng = np.random.default_rng(seed)
    base = _base_image(rng, (128, 128))
    spec = ForgeSpec(
        size=(128, 128),
        shape="rect",
        src_polygon=[[24, 24], [24, 56], [56, 56], [56, 24]],
        dst_offset=(40, 36),
        rotate_deg=0.0,
        scale=1.0,
        postproc="jpeg80",
        tier="easy",
    )
    forged, _mask = forge(base, spec)
    out = tmp_path / "copy_move.jpg"
    ok, buf = cv2.imencode(".jpg", forged, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    out.write_bytes(buf.tobytes())
    return out


def _clean_noise_png(tmp_path: Path, seed: int = 17) -> Path:
    noise = np.random.default_rng(seed).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    out = tmp_path / "noise_clean.png"
    ok, buf = cv2.imencode(".png", noise)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    out.write_bytes(buf.tobytes())
    return out


def _strip_volatile(payload: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    for key in _VOLATILE_TOP:
        out.pop(key, None)
    for method in out.get("methods", []):
        if isinstance(method, dict):
            method.pop("duration_ms", None)
    return out


def _assert_report_contract(data: dict[str, Any], raw_stdout: str) -> None:
    assert data["overall_verdict"] in (v.value for v in Verdict)
    low = raw_stdout.lower()
    for word in _FORBIDDEN_OUT:
        assert word not in low, f"forbidden word {word!r} in CLI output"
    assert "Traceback" not in raw_stdout


def _analyze_twice(tmp_path: Path, image: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    args = ["analyze", str(image), "--methods", "ela", "--no-persist"]
    r1 = _run(tmp_path, args)
    r2 = _run(tmp_path, args)
    assert r1.returncode == 0, r1.stderr
    assert r2.returncode == 0, r2.stderr
    d1 = json.loads(r1.stdout)
    d2 = json.loads(r2.stdout)
    _assert_report_contract(d1, r1.stdout)
    _assert_report_contract(d2, r2.stdout)
    return d1, d2


@pytest.mark.parametrize("fixture", ("copy_move_jpeg", "clean_noise_png"))
def test_analyze_no_persist_json_contract_and_deterministic(
    tmp_path: Path, fixture: str
) -> None:
    image = (
        _jpeg_copy_move(tmp_path) if fixture == "copy_move_jpeg" else _clean_noise_png(tmp_path)
    )
    d1, d2 = _analyze_twice(tmp_path, image)
    assert _strip_volatile(d1) == _strip_volatile(d2)


def test_analyze_persist_stays_under_tmp_path_data(tmp_path: Path) -> None:
    image = _clean_noise_png(tmp_path, seed=91)
    repo_db = _REPO_SRC.parent / "data" / "imageguard.db"
    had_repo_db = repo_db.is_file()

    res = _run(tmp_path, ["analyze", str(image), "--methods", "ela"])
    assert res.returncode == 0, res.stderr
    assert "Traceback" not in res.stderr

    data_root = tmp_path / "data"
    db = data_root / "imageguard.db"
    assert db.is_file()
    art_root = data_root / "artifacts"
    assert art_root.is_dir()
    assert list(art_root.rglob("*.json"))

    with sqlite3.connect(db) as conn:
        n = conn.execute("SELECT count(*) FROM analyses").fetchone()[0]
    assert n == 1

    if not had_repo_db:
        assert not repo_db.is_file()


def test_ingest_once_counts_custody_and_second_run_is_quiet(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    body = _jpeg_copy_move(tmp_path, seed=5).read_bytes()
    (incoming / "primary.jpg").write_bytes(body)
    (incoming / "duplicate.jpg").write_bytes(body)
    (incoming / "other.png").write_bytes(_clean_noise_png(tmp_path, seed=6).read_bytes())
    (incoming / "corrupt.jpg").write_bytes(b"\xff\xd8\xffbroken")

    cfg = tmp_path / "ingest.yaml"
    cfg.write_text(
        "camera_label: e2e bench\nmax_pending: 4\nmethods: [ela]\ndata_dir: data\n",
        encoding="utf-8",
    )

    first = _run(
        tmp_path,
        ["ingest", "once", str(incoming), "--config", str(cfg)],
    )
    assert first.returncode == 0, first.stderr
    assert "analyzed=2" in first.stdout
    assert "errors=1" in first.stdout
    assert "skipped_duplicate=1" in first.stdout

    custody = tmp_path / "data" / "ingest" / "custody.jsonl"
    assert custody.is_file()
    lines = custody.read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        record = json.loads(line)
        assert "basename" in record
        assert re.search(r"[\\/]", record["basename"]) is None
        blob = json.dumps(record, ensure_ascii=False)
        assert str(tmp_path) not in blob

    with sqlite3.connect(tmp_path / "data" / "imageguard.db") as conn:
        count_after_first = conn.execute(
            "SELECT count(*) FROM analyses"
        ).fetchone()[0]

    second = _run(
        tmp_path,
        ["ingest", "once", str(incoming), "--config", str(cfg)],
    )
    assert second.returncode == 0, second.stderr
    assert "analyzed=0" in second.stdout
    assert "skipped_duplicate=0" in second.stdout
    # corrupt.jpg is retried each scan (error is not a completed custody event)
    assert "errors=" in second.stdout

    with sqlite3.connect(tmp_path / "data" / "imageguard.db") as conn:
        count_after_second = conn.execute(
            "SELECT count(*) FROM analyses"
        ).fetchone()[0]
    assert count_after_second == count_after_first


def test_eval_synth_keypoint_block_writes_reports(tmp_path: Path) -> None:
    ds = tmp_path / "synthroot"
    generate_set(2, seed=9, out_dir=ds, size=(128, 128), tier="easy")
    meta = json.loads((ds / "meta.json").read_text(encoding="utf-8"))
    split = "dev"
    if not any(split_of(e["id"]) == "dev" for e in meta):
        split = "test"

    reports = tmp_path / "reports"
    res = _run(
        tmp_path,
        [
            "eval",
            "--dataset",
            "synth",
            "--root",
            str(ds),
            "--split",
            split,
            "--methods",
            "keypoint,block",
            "--seed",
            "9",
            "--out",
            str(reports),
        ],
        timeout=55.0,
    )
    assert res.returncode == 0, res.stderr + res.stdout
    assert "Traceback" not in res.stderr
    assert "experiment_id:" in res.stdout
    assert "copy_move_keypoint:" in res.stdout or "copy_move_block:" in res.stdout

    written = list(reports.glob("eval-*.json")) + list(reports.glob("eval-*.csv"))
    assert written, "expected eval report files under tmp reports dir"
