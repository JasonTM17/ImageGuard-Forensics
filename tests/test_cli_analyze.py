"""Tests for `python -m imageguard analyze`."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.enums import Verdict
from imageguard.core.errors import ErrorCode
from imageguard.forensics.registry import METHOD_SPECS

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"


def _run_analyze(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    merged = {"timeout": 60, "capture_output": True, "text": True,
              "encoding": "utf-8", "errors": "replace"}
    merged.update(kwargs)
    return subprocess.run(
        [sys.executable, "-m", "imageguard", *args],
        check=False,
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

    forensics_db = _REPO_SRC.parent / "data" / "imageguard.db"
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
    # The option --deepfake-model names a method, not an authenticity verdict.
    assert re.search(r"\bfake\b", out) is None
    assert "authentic" not in out
    assert "deepfake_ensemble" in out
    assert "keypoint" in out


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

def _tiny_jpeg(tmp_path: Path) -> Path:
    img = np.random.default_rng(42).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    jpg_path = tmp_path / "tiny.jpg"
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    jpg_path.write_bytes(buf.tobytes())
    return jpg_path


def test_analyze_unknown_method_exits_2_one_line_stderr_lists_valid_methods(tmp_path: Path) -> None:
    jpg = _tiny_jpeg(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(_REPO_SRC)}
    res = _run_analyze(
        ["analyze", str(jpg), "--methods", "bogus", "--no-persist"],
        cwd=tmp_path,
        env=env,
    )
    assert res.returncode == 2
    assert res.stdout == ""
    assert "Traceback" not in res.stderr
    assert "error: unknown analyze method 'bogus'" in res.stderr
    assert res.stderr.count("\n") <= 2
    for name in METHOD_SPECS:
        assert name in res.stderr
    assert not (tmp_path / "data").exists()


def test_analyze_mixed_valid_and_unknown_method_exits_2_without_report(tmp_path: Path) -> None:
    jpg = _tiny_jpeg(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(_REPO_SRC)}
    res = _run_analyze(
        ["analyze", str(jpg), "--methods", "keypoint,nope", "--no-persist"],
        cwd=tmp_path,
        env=env,
    )
    assert res.returncode == 2
    assert res.stdout == ""
    assert "unknown analyze method 'nope'" in res.stderr


def test_analyze_case_variant_is_unknown_not_duplicate(tmp_path: Path) -> None:
    jpg = _tiny_jpeg(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(_REPO_SRC)}
    res = _run_analyze(
        ["analyze", str(jpg), "--methods", "KEYPOINT", "--no-persist"],
        cwd=tmp_path,
        env=env,
    )
    assert res.returncode == 2
    assert "unknown analyze method 'KEYPOINT'" in res.stderr
    assert "duplicate" not in res.stderr


@pytest.mark.parametrize(
    "methods_arg, expected",
    [
        ("ela,ela", "ela"),
        (" ela , ela ", "ela"),
        ("keypoint,block,keypoint", "keypoint"),
        ("block,keypoint,block", "block"),
    ],
)
def test_analyze_duplicate_method_exits_2_one_line_stderr(
    tmp_path: Path, methods_arg: str, expected: str,
) -> None:
    jpg = _tiny_jpeg(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(_REPO_SRC)}
    res = _run_analyze(
        ["analyze", str(jpg), "--methods", methods_arg, "--no-persist"],
        cwd=tmp_path,
        env=env,
    )
    assert res.returncode == 2
    assert res.stdout == ""
    assert "Traceback" not in res.stderr
    assert "duplicate analyze method" in res.stderr
    assert repr(expected) in res.stderr


def test_analyze_distinct_valid_methods_still_exit_0(tmp_path: Path) -> None:
    jpg = _tiny_jpeg(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(_REPO_SRC)}
    res = _run_analyze(
        ["analyze", str(jpg), "--methods", "ela,noise", "--no-persist"],
        cwd=tmp_path,
        env=env,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["overall_verdict"] in (v.value for v in Verdict)


def test_analyze_empty_methods_still_uses_defaults(tmp_path: Path) -> None:
    jpg = _tiny_jpeg(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(_REPO_SRC)}
    res = _run_analyze(
        ["analyze", str(jpg), "--methods", ",", "--no-persist"],
        cwd=tmp_path,
        env=env,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert "overall_verdict" in data
