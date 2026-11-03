# Build ImageGuard.exe — onedir (nhanh hơn onefile lúc startup với Qt).
# Chạy: powershell -ExecutionPolicy Bypass -File scripts/build_app.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py -m PyInstaller --noconfirm --clean --onedir --windowed `
    --name ImageGuard `
    --collect-submodules imageguard `
    --add-data "configs;configs" `
    imageguard_gui.py

Write-Host "Done -> dist\ImageGuard\ImageGuard.exe"
