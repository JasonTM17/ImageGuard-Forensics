"""Path resolution — chạy được cả dev lẫn PyInstaller frozen.

Frozen (sys.frozen=True): PyInstaller bung vào sys._MEIPASS (thư mục
`_internal`), exe nằm ở `sys.executable`. Config bundle ở
`_MEIPASS/configs/`; data (db + artifacts) neo vào THƯ MỤC CHỨA EXE —
không theo cwd, tránh rải data/ lung tung khi user double-click từ nơi khác.
"""

from __future__ import annotations

import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))


def config_path() -> Path:
    if FROZEN:
        return Path(sys._MEIPASS) / "configs" / "thresholds.yaml"
    return Path(__file__).resolve().parents[3] / "configs" / "thresholds.yaml"


def data_root() -> Path:
    if FROZEN:
        return Path(sys.executable).resolve().parent / "data"
    return Path("data")
