@echo off
title ImageGuard Forensics Launcher
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================================
echo         ImageGuard Forensics - Khoi dong ung dung
echo ========================================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo Dang su dung moi truong ao .venv...
    ".venv\Scripts\python.exe" -m imageguard
) else (
    echo Khong tim thay .venv, dang thu voi python he thong...
    python -m imageguard
)

if errorlevel 1 (
    echo.
    echo [THONG BAO] Ung dung da dong hoac gap su co khoi dong.
    echo Neu gap loi thu vien, hay kiem tra moi truong qua:
    echo   .venv\Scripts\python.exe scripts\verify_env.py
    pause
)
