"""CLI `main()` routing — unit tests qua monkeypatch, không chạy subprocess."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from imageguard import cli


def test_version_flag_exits_zero(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["imageguard", "--version"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert "ImageGuard Forensics v" in capsys.readouterr().out


def test_unknown_positional_routes_to_gui(monkeypatch) -> None:
    fake_app = types.ModuleType("imageguard.ui.app")
    captured: dict = {}

    def fake_gui_main(image=None):
        captured["image"] = image
        return 0

    fake_app.main = fake_gui_main
    monkeypatch.setitem(sys.modules, "imageguard.ui.app", fake_app)
    exits: list[int] = []
    monkeypatch.setattr(sys, "exit", lambda code: exits.append(code))
    monkeypatch.setattr(sys, "argv", ["imageguard", "photo.png"])

    cli.main()

    assert captured["image"] == Path("photo.png")
    assert exits == [0]


def test_analyze_no_persist_delegates(monkeypatch) -> None:
    captured: dict = {}

    def fake_run_analyze(args):
        captured["args"] = args

    monkeypatch.setattr(cli, "_run_analyze_command", fake_run_analyze)
    monkeypatch.setattr(
        sys, "argv", ["imageguard", "analyze", "img.png", "--no-persist"]
    )

    cli.main()

    args = captured["args"]
    assert args.no_persist is True
    assert args.file == Path("img.png")


def test_ingest_delegates_to_run_ingest(monkeypatch) -> None:
    from imageguard.ingest import commands as ingest_commands

    captured: dict = {}

    def fake_run_ingest(mode, directory, config):
        captured.update(mode=mode, directory=directory, config=config)
        return 5

    monkeypatch.setattr(ingest_commands, "run_ingest", fake_run_ingest)
    monkeypatch.setattr(
        sys,
        "argv",
        ["imageguard", "ingest", "once", "watch", "--config", "cfg.yaml"],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 5
    assert captured["mode"] == "once"
    assert captured["directory"] == Path("watch")
    assert captured["config"] == Path("cfg.yaml")
