"""Launcher cho PyInstaller — entry point của .exe build.

Giữ ở repo root: PyInstaller cần file script thật (không phải -m)."""

from imageguard.ui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
