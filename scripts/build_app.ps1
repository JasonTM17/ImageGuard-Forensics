# Build ImageGuard.exe — onedir (nhanh hơn onefile lúc startup với Qt).
# Nguồn duy nhất: ImageGuard.spec (cùng nội dung CLI cũ: collect-submodules
# imageguard + --add-data configs). Chạy:
#   powershell -ExecutionPolicy Bypass -File scripts/build_app.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py -m PyInstaller --noconfirm --clean ImageGuard.spec

Write-Host "Done -> dist\ImageGuard\ImageGuard.exe"
