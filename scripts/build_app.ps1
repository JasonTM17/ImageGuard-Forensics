# Build ImageGuard.exe — onedir (nhanh hơn onefile lúc startup với Qt).
# Nguồn duy nhất: ImageGuard.spec (cùng nội dung CLI cũ: collect-submodules
# imageguard + --add-data configs). Chạy:
#   powershell -ExecutionPolicy Bypass -File scripts/build_app.ps1
param(
    [string]$DistPath,
    [string]$WorkPath
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$originalPath = $env:Path
$originalLocation = (Get-Location).Path

try {
    Set-Location $root

    $py = Join-Path $root ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $py)) {
        # Pin the first real python on PATH by absolute path: a bare "python"
        # would re-resolve after the PATH restriction below and could land on
        # a different interpreter than the one the probe measured.
        $py = Get-Command -Name python -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty Source
        if (-not $py) {
            throw "No .venv interpreter and no python application on PATH."
        }
    }

    # Resolve the chosen interpreter's real base install before touching PATH,
    # then build with an allowlisted PATH so PyInstaller's DLL search cannot
    # pick up third-party dirs (e.g. Poppler's icuuc.dll) ahead of System32.
    $probe = & $py -c "import sys; print(sys.base_prefix)"
    if ($LASTEXITCODE -ne 0) {
        throw "Python base-prefix probe failed (exit code $LASTEXITCODE)."
    }
    $pyBase = ([string]($probe | Select-Object -Last 1)).Trim()
    if (-not $pyBase -or -not (Test-Path -LiteralPath $pyBase -PathType Container)) {
        throw "Python base-prefix probe returned an invalid path: $pyBase"
    }

    $allowedDirs = @("$env:SystemRoot\System32", "$env:SystemRoot", $pyBase)
    $pyDir = Split-Path -Parent $py
    if ($pyDir) { $allowedDirs += $pyDir }
    $allowedDirs = @(
        $allowedDirs |
            Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) } |
            Select-Object -Unique
    )
    $env:Path = $allowedDirs -join ";"

    $arguments = @("-m", "PyInstaller", "--noconfirm", "--clean")
    if ($DistPath) { $arguments += @("--distpath", $DistPath) }
    if ($WorkPath) { $arguments += @("--workpath", $WorkPath) }
    $arguments += "ImageGuard.spec"

    & $py @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed (exit code $LASTEXITCODE)."
    }

    $dist = if ($DistPath) { $DistPath } else { "dist" }
    Write-Host "Done -> $dist\ImageGuard\ImageGuard.exe"
} finally {
    $env:Path = $originalPath
    Set-Location $originalLocation
}
