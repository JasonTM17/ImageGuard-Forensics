"""scripts/make_report_figures.write_png_limited — size cap is never exceeded."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_report_figures.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_report_figures_under_test", _SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {_SCRIPT.name}")
    module = importlib.util.module_from_spec(spec)
    # dataclass resolution needs the module registered while it executes.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


@pytest.fixture(scope="module")
def figures_script() -> ModuleType:
    return _load_script()


@pytest.fixture()
def noisy_bgr() -> np.ndarray:
    rng = np.random.default_rng(20261112)
    return rng.integers(0, 256, (400, 400, 3), dtype=np.uint8)


def test_png_under_cap_is_written_unchanged(
    figures_script: ModuleType, noisy_bgr: np.ndarray, tmp_path: Path
) -> None:
    target = tmp_path / "nested" / "fig.png"
    size = figures_script.write_png_limited(noisy_bgr, target, max_bytes=10**7)
    assert size == target.stat().st_size
    decoded = cv2.imdecode(
        np.frombuffer(target.read_bytes(), np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None
    assert decoded.shape == noisy_bgr.shape


def test_oversized_png_is_downscaled_until_under_cap(
    figures_script: ModuleType, noisy_bgr: np.ndarray, tmp_path: Path
) -> None:
    cap = 4000
    target = tmp_path / "small.png"
    size = figures_script.write_png_limited(noisy_bgr, target, max_bytes=cap)
    assert size <= cap
    assert target.stat().st_size == size
    decoded = cv2.imdecode(
        np.frombuffer(target.read_bytes(), np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None
    assert decoded.shape[0] < noisy_bgr.shape[0]


def test_unreachable_cap_raises_and_writes_nothing(
    figures_script: ModuleType, noisy_bgr: np.ndarray, tmp_path: Path
) -> None:
    target = tmp_path / "impossible.png"
    with pytest.raises(ValueError, match="cannot encode PNG within"):
        figures_script.write_png_limited(noisy_bgr, target, max_bytes=20)
    assert not target.exists()


def test_failed_cap_keeps_existing_file_untouched(
    figures_script: ModuleType, noisy_bgr: np.ndarray, tmp_path: Path
) -> None:
    target = tmp_path / "previous.png"
    target.write_bytes(b"previous payload")
    with pytest.raises(ValueError, match="cannot encode PNG within"):
        figures_script.write_png_limited(noisy_bgr, target, max_bytes=20)
    assert target.read_bytes() == b"previous payload"


def test_write_png_limited_validates_arguments(
    figures_script: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="BGR"):
        figures_script.write_png_limited(np.zeros((4, 4), np.uint8), tmp_path / "a.png")
    with pytest.raises(ValueError, match="non-empty"):
        figures_script.write_png_limited(
            np.zeros((0, 4, 3), np.uint8), tmp_path / "b.png"
        )
    with pytest.raises(ValueError, match="positive"):
        figures_script.write_png_limited(
            np.zeros((4, 4, 3), np.uint8), tmp_path / "c.png", max_bytes=0
        )
    with pytest.raises(TypeError, match="integer"):
        figures_script.write_png_limited(
            np.zeros((4, 4, 3), np.uint8), tmp_path / "d.png", max_bytes=1.5
        )
