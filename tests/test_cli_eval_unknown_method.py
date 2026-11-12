"""Regression: eval rejects bad --methods without a traceback."""

from __future__ import annotations

import sys
from typing import ClassVar

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


@pytest.mark.parametrize("methods_arg", ["", " ", ",", " , "])
def test_eval_empty_methods_exits_2_one_line_stderr(
    monkeypatch, capsys, tmp_path, methods_arg: str,
) -> None:
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
            methods_arg,
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "must name at least one method" in captured.err
    for name in METHOD_SPECS:
        assert name in captured.err


@pytest.mark.parametrize("methods_arg", ["", " ", ",", " , "])
def test_eval_empty_methods_skips_run_eval(
    monkeypatch, tmp_path, methods_arg: str,
) -> None:
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
            methods_arg,
        ],
    )

    def fail_run_eval(*_args, **_kwargs):
        raise AssertionError("run_eval must not run when --methods is empty")

    monkeypatch.setattr(runner_mod, "run_eval", fail_run_eval)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2


def _eval_argv(tmp_path, methods_arg: str) -> list[str]:
    return [
        "imageguard",
        "eval",
        "--dataset",
        "synth",
        "--root",
        str(tmp_path),
        "--methods",
        methods_arg,
    ]


@pytest.mark.parametrize(
    ("methods_arg", "duplicate"),
    [
        ("keypoint,keypoint", "keypoint"),
        (" keypoint , keypoint ", "keypoint"),
        ("keypoint,block,keypoint", "keypoint"),
        ("block,keypoint,block", "block"),
    ],
)
def test_eval_duplicate_method_exits_2_one_line_stderr(
    monkeypatch, capsys, tmp_path, methods_arg: str, duplicate: str,
) -> None:
    """Repeated names would double-count samples and metrics in run_eval."""
    monkeypatch.setattr(sys, "argv", _eval_argv(tmp_path, methods_arg))

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "duplicate eval method" in captured.err
    assert repr(duplicate) in captured.err
    assert len(captured.err.strip().splitlines()) == 1


def test_eval_duplicate_method_skips_run_eval(monkeypatch, tmp_path) -> None:
    import imageguard.core.config as config_mod
    import imageguard.evaluation.runner as runner_mod

    monkeypatch.setattr(sys, "argv", _eval_argv(tmp_path, "keypoint,keypoint"))

    def fail(*_args, **_kwargs):
        raise AssertionError("must not run past duplicate validation")

    monkeypatch.setattr(runner_mod, "run_eval", fail)
    monkeypatch.setattr(config_mod, "load_thresholds", fail)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2


def test_eval_case_variant_is_unknown_not_duplicate(
    monkeypatch, capsys, tmp_path,
) -> None:
    """The parser strips whitespace only; names stay case-sensitive."""
    monkeypatch.setattr(sys, "argv", _eval_argv(tmp_path, "keypoint,KEYPOINT"))

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "unknown eval method 'KEYPOINT'" in captured.err
    assert "duplicate" not in captured.err


def test_eval_distinct_methods_still_run(monkeypatch, tmp_path) -> None:
    import imageguard.core.config as config_mod
    import imageguard.evaluation.runner as runner_mod

    calls: dict[str, object] = {}

    class _Report:
        experiment_id = "test"
        per_method: ClassVar[dict[str, dict]] = {}

        def save(self, out):
            return out / "report.json", out / "report.csv"

    def fake_run_eval(adapter, methods, cfg, split, seed):
        calls["methods"] = [m.name for m in methods]
        calls["split"] = split
        calls["seed"] = seed
        return _Report()

    monkeypatch.setattr(sys, "argv", _eval_argv(tmp_path, "keypoint, block"))
    monkeypatch.setattr(config_mod, "load_thresholds", lambda: object())
    monkeypatch.setattr(runner_mod, "run_eval", fake_run_eval)

    cli.main()

    assert calls["methods"] == ["copy_move_keypoint", "copy_move_block"]
    assert calls["split"] == "dev"
    assert calls["seed"] == 42
