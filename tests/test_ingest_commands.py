"""Unit tests for `run_ingest` exit codes and once-mode side effects.

Covers the CLI boundary: invalid config → 2, missing directory → 2, setup
failure → 1, and a successful `once` scan → 0 with custody + summary written.
Runs in-process (no subprocess) with `monkeypatch.chdir` so `data_dir: data`
resolves inside `tmp_path`.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from imageguard.ingest import commands


def _jpeg_bytes(seed: int = 1) -> bytes:
    pixels = np.random.default_rng(seed).integers(
        0, 255, (128, 128, 3), dtype=np.uint8
    )
    ok, encoded = cv2.imencode(".jpg", pixels)
    assert ok
    return encoded.tobytes()


def test_invalid_config_returns_2(tmp_path: Path, capsys) -> None:
    cfg = tmp_path / "ingest.yaml"
    cfg.write_text("max_pending: 0\n", encoding="utf-8")  # invalid: must be >= 1

    rc = commands.run_ingest("once", tmp_path, cfg)

    assert rc == 2
    assert "invalid ingest configuration" in capsys.readouterr().err


def test_missing_directory_returns_2(tmp_path: Path, capsys) -> None:
    cfg = tmp_path / "ingest.yaml"
    cfg.write_text("methods: [ela]\ndata_dir: data\n", encoding="utf-8")

    rc = commands.run_ingest("once", tmp_path / "does-not-exist", cfg)

    assert rc == 2
    assert "does not exist" in capsys.readouterr().err


def test_setup_failure_returns_1(tmp_path: Path, monkeypatch, capsys) -> None:
    watch = tmp_path / "incoming"
    watch.mkdir()
    cfg = tmp_path / "ingest.yaml"
    cfg.write_text("methods: [ela]\ndata_dir: data\n", encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(commands, "make_session_factory", boom)

    rc = commands.run_ingest("once", watch, cfg)

    assert rc == 1
    assert "setup failed" in capsys.readouterr().err


def test_once_mode_writes_custody_and_summary(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    watch = tmp_path / "incoming"
    watch.mkdir()
    (watch / "sample.jpg").write_bytes(_jpeg_bytes())
    cfg = tmp_path / "ingest.yaml"
    cfg.write_text(
        "camera_label: test camera\nmethods: [ela]\ndata_dir: data\n",
        encoding="utf-8",
    )

    rc = commands.run_ingest("once", str(watch), cfg)

    assert rc == 0
    out = capsys.readouterr().out
    assert "analyzed=1" in out

    custody = tmp_path / "data" / "ingest" / "custody.jsonl"
    assert custody.is_file()
    events = [json.loads(line) for line in custody.read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events] == ["analyzed"]
