"""Tests for `python -m imageguard analyze`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.errors import ErrorCode

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"


def _run_analyze(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    merged = {"timeout": 60, "check": False, "capture_output": True,
              "text": True, "encoding": "utf-8", "errors": "replace"}
    merged.update(kwargs)
    return subprocess.run(
        [sys.executable, "-m", "imageguard", *args],
        **merged,
    )


def test_analyze_valid_jpeg_no_persist_prints_report(tmp_path: Path) -> None:
    img = np.random.default_rng(42).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    jpg_path = tmp_path / "tiny.jpg"
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    jpg_path.write_bytes(buf.tobytes())

    env = {**os.environ, "PYTHONPATH": str(_REPO_SRC)}
    res = _run_analyze(
        ["analyze", str(jpg_path), "--methods", "ela", "--no-persist"],
        cwd=tmp_path,
        env=env,
    )
    assert res.returncode == 0
    assert "Traceback" not in res.stderr
    assert "Traceback" not in res.stdout
    data = json.loads(res.stdout)
    assert data["overall_verdict"] in (v.value for v in Verdict)


def test_analyze_truncated_jpeg_error_json_no_traceback(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.jpg"
    bad_path.write_bytes(b"\xff\xd8\xff\xd9" + b"not-a-jpeg")

    env = {**os.environ, "PYTHONPATH": str(_REPO_SRC)}
    res = _run_analyze(
        ["analyze", str(bad_path), "--methods", "ela", "--no-persist"],
        cwd=tmp_path,
        env=env,
    )
    assert res.returncode != 0
    assert "Traceback" not in res.stderr
    assert "Traceback" not in res.stdout
    err = json.loads(res.stdout)
    assert err["code"] in (c.value for c in ErrorCode)


def test_analyze_subprocess_e2e_matches_flags(tmp_path: Path) -> None:
    img = np.random.default_rng(7).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    p = tmp_path / "e2e.jpg"
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    p.write_bytes(buf.tobytes())

    env = {**os.environ, "PYTHONPATH": str(_REPO_SRC)}
    cmd = [
        sys.executable,
        "-m",
        "imageguard",
        "analyze",
        str(p),
        "--methods",
        "ela",
        "--no-persist",
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
        cwd=tmp_path,
        env=env,
    )
    assert res.returncode == 0
    assert "Traceback" not in res.stderr
    data = json.loads(res.stdout)
    assert "overall_verdict" in data


def test_analyze_persist_creates_db_in_cwd(tmp_path: Path) -> None:
    img = np.random.default_rng(99).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    p = tmp_path / "persist.jpg"
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    assert ok
    p.write_bytes(buf.tobytes())

    forensics_db = Path("D:/ImageGuard Forensics/data/imageguard.db")
    had_forensics_db = forensics_db.is_file()
    env = {**os.environ, "PYTHONPATH": str(_REPO_SRC)}
    res = _run_analyze(
        ["analyze", str(p), "--methods", "ela"],
        cwd=tmp_path,
        env=env,
    )
    assert res.returncode == 0
    db = tmp_path / "data" / "imageguard.db"
    assert db.is_file()
    if not had_forensics_db:
        assert not forensics_db.is_file()


def test_analyze_help_forbidden_words() -> None:
    env = {**os.environ, "PYTHONPATH": str(_REPO_SRC)}
    res = _run_analyze(["analyze", "--help"], env=env)
    assert res.returncode == 0
    out = (res.stdout + res.stderr).lower()
    assert "forged" not in out
    assert "fake" not in out
    assert "authentic" not in out


def test_cli_version_unchanged() -> None:
    res = subprocess.run(
        [sys.executable, "-m", "imageguard", "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
        env={**os.environ, "PYTHONPATH": str(_REPO_SRC)},
    )
    assert res.returncode == 0
    assert "ImageGuard Forensics v" in res.stdout
