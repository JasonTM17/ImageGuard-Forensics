"""Lock CLI --help text and argparse / analyze error exit contract (subprocess).

Observed exit codes (do not change behavior in this module):
- argparse usage / invalid choice: 2
- ``analyze`` ingest failures (``ErrorResult`` on stdout JSON): 1
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_SRC = _REPO_ROOT / "src"

# Documented from live CLI runs (``python -m imageguard …``).
EXIT_ARGPARSE_ERROR = 2
EXIT_ANALYZE_ERROR_RESULT = 1

_HELP_CASES: list[tuple[Sequence[str], tuple[str, ...]]] = [
    ([], ("eval", "gui", "analyze", "ingest", "--help")),
    (["eval"], ("--dataset", "--root", "--split", "--methods", "--seed", "--out")),
    (["gui"], ("image",)),
    (
        ["analyze"],
        ("file", "--methods", "--no-persist", "keypoint,block,ela"),
    ),
    (["ingest"], ("watch", "once")),
    (["ingest", "once"], ("directory", "--config")),
    (["ingest", "watch"], ("directory", "--config")),
]

_MISSING_REQUIRED_CASES: list[tuple[Sequence[str], str]] = [
    (["eval"], "--dataset"),
    (["analyze"], "file"),
    (["ingest"], "ingest_mode"),
    (["ingest", "once"], "directory"),
]


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(_REPO_SRC)
        if not prior
        else os.pathsep.join((str(_REPO_SRC), prior))
    )
    return env


def _run_cli(
    tmp_path: Path,
    *args: str,
    timeout: float = 60,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "imageguard", *args],
        cwd=tmp_path,
        env=_cli_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _combined_output(res: subprocess.CompletedProcess[str]) -> str:
    return res.stdout + res.stderr


def _assert_no_traceback(res: subprocess.CompletedProcess[str]) -> None:
    combined = _combined_output(res)
    assert "Traceback" not in combined
    assert "Traceback (most recent call last)" not in combined


def _stderr_error_lines(stderr: str) -> list[str]:
    return [ln for ln in stderr.splitlines() if "error:" in ln.lower()]


@pytest.mark.parametrize(
    "argv,needles",
    _HELP_CASES,
    ids=[
        "root",
        "eval",
        "gui",
        "analyze",
        "ingest",
        "ingest-once",
        "ingest-watch",
    ],
)
def test_subcommand_help_exits_zero_and_lists_options(
    tmp_path: Path,
    argv: Sequence[str],
    needles: tuple[str, ...],
) -> None:
    res = _run_cli(tmp_path, *argv, "--help")
    assert res.returncode == 0, res.stderr
    _assert_no_traceback(res)
    out = _combined_output(res).lower()
    for needle in needles:
        assert needle.lower() in out, f"missing {needle!r} in help for {argv!r}"


@pytest.mark.parametrize("argv,required_token", _MISSING_REQUIRED_CASES)
def test_missing_required_args_exit_2_stderr_error_line(
    tmp_path: Path,
    argv: Sequence[str],
    required_token: str,
) -> None:
    res = _run_cli(tmp_path, *argv)
    assert res.returncode == EXIT_ARGPARSE_ERROR
    _assert_no_traceback(res)
    err_lines = _stderr_error_lines(res.stderr)
    assert len(err_lines) == 1, err_lines
    assert required_token in err_lines[0].lower()


def test_ingest_unknown_mode_exit_2_one_line_stderr(tmp_path: Path) -> None:
    res = _run_cli(tmp_path, "ingest", "bogus", str(tmp_path))
    assert res.returncode == EXIT_ARGPARSE_ERROR
    _assert_no_traceback(res)
    err_lines = _stderr_error_lines(res.stderr)
    assert len(err_lines) == 1, err_lines
    assert "invalid choice" in err_lines[0].lower()
    assert "bogus" in err_lines[0]


def test_eval_unknown_method_exit_2_one_line_stderr(tmp_path: Path) -> None:
    res = _run_cli(
        tmp_path,
        "eval",
        "--dataset",
        "synth",
        "--root",
        str(tmp_path),
        "--methods",
        "nosuch",
    )
    assert res.returncode == EXIT_ARGPARSE_ERROR
    _assert_no_traceback(res)
    non_empty = [ln for ln in res.stderr.splitlines() if ln.strip()]
    assert len(non_empty) == 1
    assert non_empty[0].startswith("error: unknown eval method")
    assert "nosuch" in non_empty[0]


@pytest.mark.parametrize(
    ("fixture_name", "write_bytes", "expected_code"),
    [
        ("missing.jpg", None, "PATH_INVALID"),
        ("empty.jpg", b"", "UNSUPPORTED_FORMAT"),
        ("notes.txt", b"hello", "UNSUPPORTED_FORMAT"),
    ],
)
def test_analyze_bad_input_json_error_exit_1_no_traceback(
    tmp_path: Path,
    fixture_name: str,
    write_bytes: bytes | None,
    expected_code: str,
) -> None:
    target = tmp_path / fixture_name
    if write_bytes is None:
        path_arg = str(tmp_path / "nope-never-created.jpg")
    else:
        target.write_bytes(write_bytes)
        path_arg = str(target)

    res = _run_cli(
        tmp_path,
        "analyze",
        path_arg,
        "--methods",
        "ela",
        "--no-persist",
    )
    assert res.returncode == EXIT_ANALYZE_ERROR_RESULT
    _assert_no_traceback(res)
    assert res.stdout.strip(), "expected JSON ErrorResult on stdout"
    payload = json.loads(res.stdout)
    assert payload["code"] == expected_code
    assert "message" in payload


def test_analyze_help_matches_readme_flags(tmp_path: Path) -> None:
    """README / user-guide example: ``--methods keypoint,block,ela`` and ``--no-persist``."""
    res = _run_cli(tmp_path, "analyze", "--help")
    assert res.returncode == 0
    help_text = _combined_output(res)
    assert re.search(r"--methods", help_text)
    assert re.search(r"--no-persist", help_text)
    assert "keypoint,block,ela" in help_text.replace(" ", "")
