"""Path resolution for source, PyInstaller, and Nuitka standalone builds.

PyInstaller uses ``sys._MEIPASS`` for bundled config and the executable's
directory for writable data. Nuitka standalone keeps this module under
``<dist>/imageguard/core``; its config, data, and optional models live in
``<dist>``. Source mode keeps config/model paths at the repository root and
data relative to the current working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))


def _is_nuitka_standalone() -> bool:
    return not FROZEN and globals().get("__compiled__") is not None


def _nuitka_dist_root() -> Path:
    # Nuitka's containing_dir can name the build directory, not the .dist root.
    # The compiled module itself is laid out at <dist>/imageguard/core/paths.py.
    return Path(__file__).resolve().parents[2]


def config_path() -> Path:
    if FROZEN:
        return Path(sys._MEIPASS) / "configs" / "thresholds.yaml"
    if _is_nuitka_standalone():
        return _nuitka_dist_root() / "configs" / "thresholds.yaml"
    return Path(__file__).resolve().parents[3] / "configs" / "thresholds.yaml"


def data_root() -> Path:
    if FROZEN:
        return Path(sys.executable).resolve().parent / "data"
    if _is_nuitka_standalone():
        return _nuitka_dist_root() / "data"
    return Path("data")


def model_root() -> Path:
    if _is_nuitka_standalone():
        return _nuitka_dist_root()
    return Path(__file__).resolve().parents[3]
