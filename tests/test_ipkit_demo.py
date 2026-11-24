"""Kiểm thử script trình diễn ipkit (scripts/ipkit_demo.py)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PYTHON = REPO.parent / "ImageGuard Forensics" / ".venv" / "Scripts" / "python.exe"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)
SCRIPT = REPO / "scripts" / "ipkit_demo.py"

EXPECTED_PNGS = (
    "c2_montage.png",
    "c3_montage.png",
    "c4_montage.png",
    "c5_montage.png",
    "c6_montage.png",
    "c7_montage.png",
)


def _run_demo(
    out: Path, extra: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    cmd = [str(PYTHON), str(SCRIPT), "--out", str(out)]
    if extra:
        cmd.extend(extra)
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(REPO),
        timeout=30,
    )


def test_ipkit_demo_end_to_end_synthetic(tmp_path: Path) -> None:
    t0 = time.perf_counter()
    proc = _run_demo(tmp_path)
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert elapsed < 15.0, f"demo too slow: {elapsed:.2f}s"
    for name in EXPECTED_PNGS:
        p = tmp_path / name
        assert p.is_file(), f"missing {name}"
        assert p.stat().st_size > 100


def test_ipkit_demo_bad_image_returns_nonzero(tmp_path: Path) -> None:
    proc = _run_demo(tmp_path, ["--image", str(tmp_path / "no_such_file.png")])
    assert proc.returncode != 0


def _load_ipkit_demo_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ipkit_demo", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_output_rejects_escape(tmp_path: Path) -> None:
    mod = _load_ipkit_demo_module()
    out = mod.validate_out_dir(tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        mod.resolve_output_file(out, "../outside.png")


def test_synthetic_image_shape() -> None:
    mod = _load_ipkit_demo_module()
    img = mod.synthetic_demo_image(0)
    assert img.shape == (mod.DEMO_H, mod.DEMO_W, 3)


def test_ipkit_demo_default_out(monkeypatch, tmp_path: Path) -> None:
    mod = _load_ipkit_demo_module()
    monkeypatch.chdir(tmp_path)
    rc = mod.main([])
    assert rc == 0
    out_dir = tmp_path / "data" / "ipkit_demo"
    for name in EXPECTED_PNGS:
        p = out_dir / name
        assert p.is_file(), f"missing default output {name}"
        assert p.stat().st_size > 100


def test_demo_chapter_c2_color_geometry_rows(monkeypatch, tmp_path: Path) -> None:
    import cv2

    mod = _load_ipkit_demo_module()
    captured: list[str] = []
    real_row = mod.build_pair_row

    def spy_row(left, right, caption):
        captured.append(caption)
        return real_row(left, right, caption)

    monkeypatch.setattr(mod, "build_pair_row", spy_row)
    out_dir = mod.validate_out_dir(tmp_path)
    path = mod.demo_chapter_c2(out_dir, mod.synthetic_demo_image(mod.DEMO_SEED))

    # 8 hàng cũ + 2 YCbCr + 2 HSV + 3 hình học = 15 hàng.
    assert len(captured) == 15
    assert "C2: YCbCr (kenh Y)" in captured
    assert "C2: YCbCr (kenh Cb | kenh Cr)" in captured
    assert "C2: HSV (kenh H: 0-360 -> 0-255)" in captured
    assert "C2: HSV (kenh S | kenh V, x255)" in captured
    assert "C2: Dich phoi (+20,+10)" in captured
    assert "C2: Phong to 1.5x (cat giua)" in captured
    assert "C2: Xoay 17 do (bilinear)" in captured

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None
    # Mỗi hàng cao 22 (caption) + 128 (ảnh); bề rộng 2*160 + 4 (gap) = 324.
    assert img.shape[:2] == (15 * (22 + 128), 324)
