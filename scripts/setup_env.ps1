# Setup môi trường dev cho ImageGuard Forensics (Windows PowerShell).
$ErrorActionPreference = "Stop"

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\verify_env.py
