"""The build wrapper must not announce success after PyInstaller fails."""

import json
import os
import shutil
import subprocess
import venv
from pathlib import Path

import pytest


def _powershell_or_skip():
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is required for the build wrapper")
    return powershell


def _stage_build_script(tmp_path):
    script = tmp_path / "scripts" / "build_app.ps1"
    script.parent.mkdir()
    source = Path(__file__).resolve().parents[1] / "scripts" / "build_app.ps1"
    shutil.copyfile(source, script)
    return script


def _ps_quote(path):
    return str(path).replace("'", "''")


def _run_powershell(powershell, command, extra_env=None):
    env = os.environ.copy()
    env.update(extra_env or {})
    return subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def _make_venv(path):
    """A real interpreter the wrapper must resolve through PATH lookup."""
    venv.create(path, with_pip=False)
    python_exe = path / "Scripts" / "python.exe"
    if not python_exe.is_file():
        pytest.skip("stdlib venv did not produce Scripts\\python.exe")
    return python_exe


_FAKE_PYINSTALLER_MAIN = """\
import json
import os
import sys

record = {
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "argv": sys.argv[1:],
    "path": os.environ.get("PATH", ""),
}
with open(os.environ["IG_BUILD_LOG"], "a", encoding="ascii") as handle:
    handle.write(json.dumps(record) + "\\n")
sys.exit(int(os.environ.get("IG_BUILDER_EXIT", "0")))
"""


def _fake_pyinstaller(tmp_path):
    """Fake builder importable through a test-only PYTHONPATH."""
    package = tmp_path / "fakepyi" / "PyInstaller"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="ascii")
    (package / "__main__.py").write_text(_FAKE_PYINSTALLER_MAIN, encoding="ascii")
    env = {
        "PYTHONPATH": str(package.parent),
        "IG_BUILD_LOG": str(tmp_path / "builder-log.jsonl"),
        "IG_BUILDER_EXIT": "0",
    }
    return env, Path(env["IG_BUILD_LOG"])


def _read_build_log(log_path):
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]


def _norm(path):
    return os.path.normcase(os.path.normpath(str(path)))


@pytest.mark.parametrize("builder_exit", [0, 23])
def test_build_script_propagates_builder_failure(tmp_path, builder_exit):
    powershell = _powershell_or_skip()
    script = _stage_build_script(tmp_path)
    venv_python = _make_venv(tmp_path / "path-venv")
    extra_env, _ = _fake_pyinstaller(tmp_path)
    extra_env["IG_BUILDER_EXIT"] = str(builder_exit)

    # No .venv in the isolated fixture: the wrapper resolves the fallback
    # interpreter through the real PATH, then runs the fake builder.
    command = (
        f"$env:Path = '{_ps_quote(venv_python.parent)};' + $env:Path; "
        f"& '{_ps_quote(script)}'"
    )
    result = _run_powershell(powershell, command, extra_env)
    assert (result.returncode == 0) == (builder_exit == 0)
    assert ("Done ->" in result.stdout) == (builder_exit == 0)


@pytest.mark.parametrize("builder_exit", [0, 23])
def test_build_script_restricts_child_path_and_restores_env(tmp_path, builder_exit):
    powershell = _powershell_or_skip()
    script = _stage_build_script(tmp_path)
    # A third-party bin dir standing in for the host tool dir that shipped a
    # broken icuuc.dll ahead of System32 on the build PATH.
    contaminated = tmp_path / "contaminated"
    contaminated.mkdir()
    (contaminated / "icuuc.dll").write_bytes(b"")
    venv_dir = tmp_path / "path-venv"
    venv_python = _make_venv(venv_dir)
    extra_env, log_path = _fake_pyinstaller(tmp_path)
    extra_env["IG_BUILDER_EXIT"] = str(builder_exit)

    command = (
        "$originalPath = $env:Path; "
        "$originalLocation = (Get-Location).Path; "
        f"$env:Path = '{_ps_quote(contaminated)};"
        f"{_ps_quote(venv_python.parent)};' + $env:Path; "
        "$scriptFailed = $false; "
        f"try {{ & '{_ps_quote(script)}' }} catch {{ $scriptFailed = $true }}; "
        f"if ($env:Path -eq ('{_ps_quote(contaminated)};"
        f"{_ps_quote(venv_python.parent)};' + $originalPath)) "
        "{ 'PATH-RESTORED' } else { 'PATH-NOT-RESTORED' }; "
        "if ((Get-Location).Path -eq $originalLocation) "
        "{ 'CWD-RESTORED' } else { 'CWD-NOT-RESTORED' }; "
        "exit ([int]$scriptFailed)"
    )
    result = _run_powershell(powershell, command, extra_env)

    assert (result.returncode == 0) == (builder_exit == 0)
    assert ("Done ->" in result.stdout) == (builder_exit == 0)
    assert "PATH-RESTORED" in result.stdout
    assert "CWD-RESTORED" in result.stdout

    records = _read_build_log(log_path)
    assert len(records) == 1
    record = records[0]
    assert _norm(record["prefix"]) == _norm(venv_dir)
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    expected = {
        _norm(os.path.join(sysroot, "System32")),
        _norm(sysroot),
        _norm(record["base_prefix"]),
        _norm(venv_python.parent),
    }
    observed = {_norm(entry) for entry in record["path"].split(";") if entry.strip()}
    assert observed == expected


def test_build_script_prefers_venv_interpreter(tmp_path):
    powershell = _powershell_or_skip()
    script = _stage_build_script(tmp_path)
    repo_venv = tmp_path / ".venv"
    _make_venv(repo_venv)
    path_venv = tmp_path / "path-venv"
    path_python = _make_venv(path_venv)
    extra_env, log_path = _fake_pyinstaller(tmp_path)

    command = (
        f"$env:Path = '{_ps_quote(path_python.parent)};' + $env:Path; "
        f"& '{_ps_quote(script)}'"
    )
    result = _run_powershell(powershell, command, extra_env)

    assert result.returncode == 0
    assert "Done ->" in result.stdout
    records = _read_build_log(log_path)
    assert len(records) == 1
    assert _norm(records[0]["prefix"]) == _norm(repo_venv)


def test_build_script_fallback_keeps_resolved_venv_interpreter(tmp_path):
    """No .venv: probe and build must share the PATH-resolved interpreter.

    Regression: a bare 'python' fallback re-resolved after the PATH allowlist
    ran, landing on the base install (no PyInstaller, wrong site-packages)
    instead of the venv python the probe measured.
    """
    powershell = _powershell_or_skip()
    script = _stage_build_script(tmp_path)
    venv_dir = tmp_path / "path-venv"
    venv_python = _make_venv(venv_dir)
    extra_env, log_path = _fake_pyinstaller(tmp_path)
    # A malicious dir later on PATH: resolution must take the first real
    # match, and the restricted PATH must drop the sentinel entirely.
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    (sentinel / "icuuc.dll").write_bytes(b"")
    (sentinel / "python.cmd").write_text("@exit /b 1\r\n", encoding="ascii")

    command = (
        f"$env:Path = '{_ps_quote(venv_python.parent)};"
        f"{_ps_quote(sentinel)};' + $env:Path; "
        f"& '{_ps_quote(script)}'"
    )
    result = _run_powershell(powershell, command, extra_env)

    assert result.returncode == 0
    assert "Done ->" in result.stdout
    records = _read_build_log(log_path)
    assert len(records) == 1
    record = records[0]
    assert _norm(record["executable"]) == _norm(venv_python)
    assert _norm(record["prefix"]) == _norm(venv_dir)
    child_path = {_norm(entry) for entry in record["path"].split(";") if entry.strip()}
    assert _norm(sentinel) not in child_path


@pytest.mark.parametrize("mode", ["nonzero-exit", "bogus-path"])
def test_build_script_fails_when_base_probe_fails(tmp_path, mode):
    powershell = _powershell_or_skip()
    script = _stage_build_script(tmp_path)
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    if mode == "nonzero-exit":
        body = "@exit /b 1\r\n"
    else:
        body = f"@echo {tmp_path / 'missing-base'}\r\n"
    (shim_dir / "python.cmd").write_text(body, encoding="ascii")
    extra_env, log_path = _fake_pyinstaller(tmp_path)

    command = (
        "$originalPath = $env:Path; "
        f"$env:Path = '{_ps_quote(shim_dir)};' + $env:Path; "
        f"try {{ & '{_ps_quote(script)}' ; 'REACHED-END' }} "
        "catch { 'FAILED-CLEAN' }; "
        f"if ($env:Path -eq ('{_ps_quote(shim_dir)};' + $originalPath)) "
        "{ 'PATH-RESTORED' } else { 'PATH-NOT-RESTORED' }; "
        "exit 0"
    )
    result = _run_powershell(powershell, command, extra_env)

    assert result.returncode == 0
    assert "FAILED-CLEAN" in result.stdout
    assert "PATH-RESTORED" in result.stdout
    assert "REACHED-END" not in result.stdout
    assert "Done ->" not in result.stdout
    assert not log_path.exists()


def test_build_script_fails_without_any_python(tmp_path):
    powershell = _powershell_or_skip()
    script = _stage_build_script(tmp_path)
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")

    command = (
        f"$env:Path = '{_ps_quote(os.path.join(sysroot, 'System32'))}'; "
        f"try {{ & '{_ps_quote(script)}' ; 'REACHED-END' }} "
        "catch { 'FAILED-CLEAN' }; "
        "exit 0"
    )
    result = _run_powershell(powershell, command)

    assert result.returncode == 0
    assert "FAILED-CLEAN" in result.stdout
    assert "REACHED-END" not in result.stdout
    assert "Done ->" not in result.stdout


def test_build_script_forwards_dist_and_work_paths(tmp_path):
    powershell = _powershell_or_skip()
    script = _stage_build_script(tmp_path)
    venv_python = _make_venv(tmp_path / "path-venv")
    extra_env, log_path = _fake_pyinstaller(tmp_path)
    dist_path = tmp_path / "isolated-dist"
    work_path = tmp_path / "isolated-work"

    command = (
        f"$env:Path = '{_ps_quote(venv_python.parent)};' + $env:Path; "
        f"& '{_ps_quote(script)}' -DistPath '{_ps_quote(dist_path)}' "
        f"-WorkPath '{_ps_quote(work_path)}'"
    )
    result = _run_powershell(powershell, command, extra_env)

    assert result.returncode == 0
    records = _read_build_log(log_path)
    assert len(records) == 1
    argv = records[0]["argv"]
    assert argv[:2] == ["--noconfirm", "--clean"]
    assert "--distpath" in argv and str(dist_path) in argv
    assert "--workpath" in argv and str(work_path) in argv
    assert argv[-1] == "ImageGuard.spec"
    assert f"{dist_path}\\ImageGuard\\ImageGuard.exe" in result.stdout
