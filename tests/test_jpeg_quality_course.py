"""Course test: FreqView blockiness rises when JPEG quality falls (Q90 → Q40)."""

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
    """128×128 BGR — gentle gradient + Gaussian noise (seed 23)."""
    h = w = 128
    rng = np.random.default_rng(23)
    ramp = np.linspace(48.0, 208.0, w, dtype=np.float64)
    ramp = np.broadcast_to(ramp, (h, w))
    noise = rng.normal(0.0, 4.0, size=(h, w))
    gray = np.clip(ramp + noise, 0.0, 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def test_freq_course_jpeg_q40_blockiness_gt_q90(tmp_path: Path) -> None:
    ts = load_thresholds()
    arr = _build_smooth_gradient_noise_128()
    q90_path = tmp_path / "course_smooth_q90.jpg"
    q40_path = tmp_path / "course_smooth_q40.jpg"
    assert cv2.imwrite(str(q90_path), arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert cv2.imwrite(str(q40_path), arr, [cv2.IMWRITE_JPEG_QUALITY, 40])

    fv = FreqView()
    r90 = fv.analyze(_open_ok(q90_path, ts), ts)
    r40 = fv.analyze(_open_ok(q40_path, ts), ts)

    assert r90.verdict != Verdict.SUSPICIOUS_INDICATION
    assert r40.verdict != Verdict.SUSPICIOUS_INDICATION

    b90 = float(r90.evidence["blockiness"])
    b40 = float(r40.evidence["blockiness"])
    assert b40 > b90, (
        "JPEG Q40 blockiness must exceed Q90 on the same smooth seed-23 image: "
        f"q40={b40:.8f}, q90={b90:.8f}"
    )
