"""Regression: eval rejects unknown --methods without a traceback."""

from __future__ import annotations

import sys

import pytest

from imageguard import cli
from imageguard.forensics.registry import METHOD_SPECS


def test_eval_unknown_method_exits_2_one_line_stderr(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "imageguard",
            "eval",
            "--dataset",
            "synth",
            "--root",
            str(tmp_path),
            "--methods",
            "bogus",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "bogus" in captured.err
    for name in METHOD_SPECS:
        assert name in captured.err


def test_eval_unknown_method_skips_run_eval(monkeypatch, tmp_path) -> None:
    import imageguard.evaluation.runner as runner_mod

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "imageguard",
            "eval",
            "--dataset",
            "synth",
            "--root",
            str(tmp_path),
            "--methods",
            "nope",
        ],
    )

    def fail_run_eval(*_args, **_kwargs):
        raise AssertionError("run_eval must not run for unknown methods")

    monkeypatch.setattr(runner_mod, "run_eval", fail_run_eval)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
