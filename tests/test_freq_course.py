"""Course test: FreqView blockiness — JPEG (Q=40) vs PNG on smooth synthetic image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.forensics.auxiliary.freq_view import FreqView


def _open_ok(path: Path, ts):
    out = safe_open(path, ts)
    assert not hasattr(out, "code"), getattr(out, "message", "")
    return out


def _build_smooth_gradient_noise_128() -> np.ndarray:
    """128×128 BGR — gentle gradient + Gaussian noise (no 8px-period structure)."""
    h, w = 128, 128
    rng = np.random.default_rng(23)
    # Horizontal ramp: smooth, no step edges aligned to block grid.
    ramp = np.linspace(48.0, 208.0, w, dtype=np.float64)
    ramp = np.broadcast_to(ramp, (h, w))
    noise = rng.normal(0.0, 4.0, size=(h, w))
    gray = np.clip(ramp + noise, 0.0, 255.0).astype(np.uint8)
    img = np.stack([gray, gray, gray], axis=-1)
    return img


def _write_png_and_jpeg_q40(tmp_path: Path, array_bgr: np.ndarray) -> tuple[Path, Path]:
    png_path = tmp_path / "course_smooth.png"
    jpg_path = tmp_path / "course_smooth_q40.jpg"
    ok = cv2.imwrite(str(png_path), array_bgr)
    assert ok
    cv2.imwrite(
        str(jpg_path),
        array_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 40],
    )
    return png_path, jpg_path


def test_freq_course_jpeg_blockiness_gt_png(tmp_path: Path) -> None:
    ts = load_thresholds()
    arr = _build_smooth_gradient_noise_128()
    png_path, jpg_path = _write_png_and_jpeg_q40(tmp_path, arr)

    fv = FreqView()
    pdoc = _open_ok(png_path, ts)
    jdoc = _open_ok(jpg_path, ts)

    rp = fv.analyze(pdoc, ts)
    rj = fv.analyze(jdoc, ts)

    assert rp.verdict != Verdict.SUSPICIOUS_INDICATION
    assert rj.verdict != Verdict.SUSPICIOUS_INDICATION

    pb = float(rp.evidence["blockiness"])
    jb = float(rj.evidence["blockiness"])
    assert jb > pb, (jb, pb)
