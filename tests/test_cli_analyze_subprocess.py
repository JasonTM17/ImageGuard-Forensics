"""CLI `analyze --no-persist` subprocess JSON contract, repeatability & error codes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from imageguard import cli as cli_mod
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.errors import ErrorCode

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"

_TOP_KEYS = {
    "report_id",
    "image_sha256",
    "image_name",
    "image_width",
    "image_height",
    "created_at",
    "tool_version",
    "env",
    "thresholds_hash",
    "overall_verdict",
    "limitations",
    "disclaimer",
    "methods",
}
_ENV_KEYS = {"python", "opencv", "numpy"}
_METHOD_KEYS = {
    "method",
    "verdict",
    "score",
    "score_definition",
    "evidence",
    "limitations",
    "params_hash",
    "duration_ms",
    "has_mask",
}
_ERROR_KEYS = {"code", "message", "detail"}


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(_REPO_SRC)}
    return subprocess.run(
        [sys.executable, "-m", "imageguard", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
        cwd=cwd,
        env=env,
    )


def _assert_valid_report_payload(payload: dict, expected_name: str, w: int, h: int) -> None:
    assert set(payload.keys()) == _TOP_KEYS
    assert len(payload["image_sha256"]) == 64
    assert payload["image_name"] == expected_name
    assert payload["image_width"] == w
    assert payload["image_height"] == h
    assert isinstance(payload["tool_version"], str) and payload["tool_version"]
    assert set(payload["env"].keys()) == _ENV_KEYS
    assert all(isinstance(v, str) and v for v in payload["env"].values())
    assert payload["overall_verdict"] in {v.value for v in Verdict}
    assert isinstance(payload["limitations"], list)
    assert isinstance(payload["disclaimer"], str) and payload["disclaimer"]

    valid_methods = {m.value for m in MethodName}
    valid_verdicts = {v.value for v in Verdict}
    for m in payload["methods"]:
        assert set(m.keys()) == _METHOD_KEYS
        assert m["method"] in valid_methods
        assert m["verdict"] in valid_verdicts
        assert isinstance(m["has_mask"], bool)
        assert isinstance(m["duration_ms"], int) and m["duration_ms"] >= 0
        assert isinstance(m["evidence"], dict)
        assert isinstance(m["limitations"], list)
        if m["score"] is not None:
            assert isinstance(m["score"], (int, float))
            assert isinstance(m["score_definition"], str) and m["score_definition"]


def test_cli_analyze_no_persist_all_methods_json_schema_and_repeatability(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, (128, 160, 3), dtype=np.uint8)
    jpg = tmp_path / "probe.jpg"
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    jpg.write_bytes(buf.tobytes())

    all_specs = "keypoint,block,ela,noise,noise_local,freq,splicing,deepfake"
    res1 = _run_cli(["analyze", str(jpg), "--methods", all_specs, "--no-persist"], cwd=tmp_path)
    assert res1.returncode == 0
    assert "Traceback" not in res1.stderr
    assert not (tmp_path / "data").exists()

    p1 = json.loads(res1.stdout)
    _assert_valid_report_payload(p1, "probe.jpg", 160, 128)
    returned_methods = {m["method"] for m in p1["methods"]}
    assert returned_methods == {m.value for m in MethodName}

    # Second run on same bytes must only vary report_id, created_at, and duration_ms.
    res2 = _run_cli(["analyze", str(jpg), "--methods", "ela", "--no-persist"], cwd=tmp_path)
    res3 = _run_cli(["analyze", str(jpg), "--methods", "ela", "--no-persist"], cwd=tmp_path)
    assert res2.returncode == 0 and res3.returncode == 0
    p2, p3 = json.loads(res2.stdout), json.loads(res3.stdout)
    assert p2["report_id"] != p3["report_id"]
    for d in (p2, p3):
        d.pop("report_id")
        d.pop("created_at")
        for m in d["methods"]:
            m.pop("duration_ms")
    assert p2 == p3


def test_cli_analyze_error_paths_exit_1_and_match_error_contract(tmp_path: Path) -> None:
    missing = tmp_path / "nope.png"

    bad_magic = tmp_path / "text.png"
    bad_magic.write_text("not an image", encoding="utf-8")

    too_small = tmp_path / "tiny.png"
    Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(too_small, format="PNG")

    corrupt_png = tmp_path / "corrupt.png"
    corrupt_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)

    cases = [
        (missing, ErrorCode.PATH_INVALID.value),
        (bad_magic, ErrorCode.UNSUPPORTED_FORMAT.value),
        (too_small, ErrorCode.IMAGE_TOO_SMALL.value),
        (corrupt_png, ErrorCode.DECODE_FAILED.value),
    ]
    for path, expected_code in cases:
        res = _run_cli(["analyze", str(path), "--methods", "ela", "--no-persist"], cwd=tmp_path)
        assert res.returncode == 1
        assert "Traceback" not in res.stderr
        err = json.loads(res.stdout)
        assert set(err.keys()) == _ERROR_KEYS
        assert err["code"] == expected_code
        assert isinstance(err["message"], str) and err["message"]
        assert isinstance(err["detail"], dict)


def test_cli_run_analyze_command_in_process_and_fallback_internal(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    png = tmp_path / "inproc.png"
    Image.fromarray(img).save(png, format="PNG")

    with pytest.raises(SystemExit) as exc0:
        cli_mod._run_analyze_command(
            Namespace(file=png, methods="ela", no_persist=True)
        )
    assert exc0.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    _assert_valid_report_payload(payload, "inproc.png", 64, 64)
    # PNG -> ELA is NOT_APPLICABLE
    ela_entry = next(m for m in payload["methods"] if m["method"] == "ela")
    assert ela_entry["verdict"] == Verdict.NOT_APPLICABLE.value
    assert ela_entry["has_mask"] is False

    # Unexpected return type from AnalysisService triggers INTERNAL JSON exit 1.
    from imageguard.services import analysis_service as asvc

    monkeypatch.setattr(asvc.AnalysisService, "analyze", lambda *_a, **_kw: object())
    with pytest.raises(SystemExit) as exc_int:
        cli_mod._run_analyze_command(
            Namespace(file=png, methods="", no_persist=True)
        )
    assert exc_int.value.code == 1
    err = json.loads(capsys.readouterr().out)
    assert err["code"] == "INTERNAL"
