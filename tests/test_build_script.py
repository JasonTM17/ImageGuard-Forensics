"""The build wrapper must not announce success after PyInstaller fails."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("builder_exit", [0, 23])
def test_build_script_propagates_builder_failure(tmp_path, builder_exit):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is required for the build wrapper")
    script = tmp_path / "scripts" / "build_app.ps1"
    script.parent.mkdir()
    source = Path(__file__).resolve().parents[1] / "scripts" / "build_app.ps1"
    shutil.copyfile(source, script)
    # No venv in the isolated fixture: intercept the fallback builder without
    # running PyInstaller or creating a release artifact.
    quoted = str(script).replace("'", "''")
    command = (
        f"function python {{ $global:LASTEXITCODE = {builder_exit} }}; "
        f"& '{quoted}'"
    )
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
        check=False, capture_output=True, text=True, timeout=30,
    )
    assert (result.returncode == 0) == (builder_exit == 0)
    assert ("Done ->" in result.stdout) == (builder_exit == 0)
