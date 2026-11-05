"""E2E tests for CLI and process execution."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from imageguard.evaluation.datasets import split_of
from imageguard.evaluation.synthforge import generate_set


def test_cli_version() -> None:
    res = subprocess.run(
        [sys.executable, "-m", "imageguard", "--version"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert res.returncode == 0
    assert "ImageGuard Forensics v" in res.stdout
    assert "Traceback" not in res.stderr
    assert "Traceback" not in res.stdout


def test_cli_eval_synth_ela(tmp_path: Path) -> None:
    ds_dir = tmp_path / "dataset"
    out_dir = tmp_path / "out"
    manifest_p = generate_set(n=2, seed=5, out_dir=ds_dir, size=(128, 128), tier="easy")

    meta = json.loads(manifest_p.read_text(encoding="utf-8"))
    for entry in meta:
        entry["format"] = "jpeg"
        stem = entry["id"]
        for suffix in ("_O", "_F"):
            png_p = ds_dir / f"{stem}{suffix}.png"
            jpg_p = ds_dir / f"{stem}{suffix}.jpg"
            if png_p.is_file():
                img = cv2.imread(str(png_p))
                cv2.imwrite(str(jpg_p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    manifest_p.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    base_ids = [f"{e['id']}-seed{e.get('seed', 5)}" for e in meta]
    chosen_split = "dev" if any(split_of(b) == "dev" for b in base_ids) else "test"

    cmd = [
        sys.executable,
        "-m",
        "imageguard",
        "eval",
        "--dataset",
        "synth",
        "--root",
        str(ds_dir),
        "--methods",
        "ela",
        "--split",
        chosen_split,
        "--seed",
        "5",
        "--out",
        str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0
    assert "Traceback" not in res.stderr
    assert "Traceback" not in res.stdout
    assert "experiment_id:" in res.stdout
    assert "ela:" in res.stdout

    m = re.search(r"ela:.*?n=(\d+)", res.stdout)
    assert m is not None, f"Expected 'n=' in ela output: {res.stdout}"
    assert int(m.group(1)) > 0, f"Expected n > 0 in ela line: {res.stdout}"

    report_files = list(out_dir.glob("eval-*.json")) + list(out_dir.glob("eval-*.csv"))
    assert len(report_files) >= 1, f"No report files written to {out_dir}"


def test_cli_eval_bad_invocation() -> None:
    res = subprocess.run(
        [sys.executable, "-m", "imageguard", "eval"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert res.returncode != 0
    assert "Traceback" not in res.stderr
    assert "Traceback" not in res.stdout
    combined = res.stdout + res.stderr
    assert ("the following arguments are required" in combined
            or "error:" in combined)


def test_cli_gui_launch_smoke(tmp_path: Path) -> None:
    img_p = tmp_path / "smoke.png"
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    img_p.write_bytes(buf.tobytes())

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        [sys.executable, "-m", "imageguard", str(img_p)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out = ""
    err = ""
    try:
        # Observe for a short polling interval within 15s to verify it stays running
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            ret = proc.poll()
            if ret is not None:
                out, err = proc.communicate()
                assert ret is None, (
                    f"GUI process exited prematurely with code {ret}.\n"
                    f"stdout:\n{out}\nstderr:\n{err}"
                )
            time.sleep(0.1)

        assert proc.poll() is None, "Process should still be running in event loop"
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                out, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate(timeout=5)
        elif not out and not err:
            out, err = proc.communicate()

    assert proc.poll() is not None
    assert "Traceback" not in out
    assert "Traceback" not in err
