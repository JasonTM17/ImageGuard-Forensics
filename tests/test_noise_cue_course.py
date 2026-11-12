"""Noise cue outlier_frac course (deterministic seed 20261106).

Locked outcomes with the default ``ThresholdSet``:

- split image (left half strong Gaussian noise, right half the same noise after
  a heavy Gaussian blur): ``evidence["outlier_frac"] == 0.0`` (0/64 blocks);
- uniform image (strong Gaussian noise across the whole frame):
  ``evidence["outlier_frac"] == 0.0469`` (3/64 blocks);
- the naive expectation ``split > uniform`` FAILS on this seed and is locked as
  the inverted order ``split < uniform``. A symmetric 50/50 bimodal σ map makes
  MAD ≈ half the cluster gap, so no block reaches ``outlier_mad · MAD``; uniform
  noise has a tight σ distribution, so a few blocks exceed ``3 · MAD``. The
  fractions are locked; thresholds are not changed.

Không trùng ``tests/test_noise_course.py``: đó đo thứ tự residual std từng nửa
của ảnh hai nửa; đây so sánh tỉ lệ block MAD-outlier giữa ảnh hai nửa và ảnh
nhiễu đồng đều, và khoá hai tỉ lệ quan sát được.
"""

from __future__ import annotations

import cv2
import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.noise_map import NoiseMap

_SEED = 20261106
_IMG = 128
_SIGMA = 40.0
_BLUR_SIGMA = 4.0

# Locked observed outlier_frac on seed 20261106 with the default ThresholdSet.
_SPLIT_LOCKED_FRAC = 0.0
_UNIFORM_LOCKED_FRAC = 0.0469


def _doc_from(bgr: np.ndarray) -> ImageDocument:
    h, w = bgr.shape[:2]
    return ImageDocument(
        array=bgr, width=w, height=h, channels=3, image_format="png",
        sha256="n" * 64, file_bytes=0, file_name="noise_cue.png",
    )


def _split_noise_then_blur() -> np.ndarray:
    """Left half strong noise; right half the same noise after a heavy blur."""
    rng = np.random.default_rng(_SEED)
    h = w = _IMG
    half = w // 2
    noise = rng.normal(0.0, _SIGMA, (h, half)).astype(np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[:, :half] = noise[:, :, None]
    img[:, half:] = cv2.GaussianBlur(noise, (0, 0), _BLUR_SIGMA)[:, :, None]
    return np.clip(128.0 + img, 0, 255).astype(np.uint8)


def _uniform_noise() -> np.ndarray:
    """Strong noise across the whole frame (same seed, independent field)."""
    rng = np.random.default_rng(_SEED)
    noise = rng.normal(0.0, _SIGMA, (_IMG, _IMG)).astype(np.float32)
    img = np.repeat(128.0 + noise[:, :, None], 3, axis=2)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_noise_cue_rate_locked_course() -> None:
    ts = load_thresholds()
    split = NoiseMap().analyze(_doc_from(_split_noise_then_blur()), ts)
    uniform = NoiseMap().analyze(_doc_from(_uniform_noise()), ts)

    # Hard aux contract: NoiseMap never returns SUSPICIOUS_INDICATION.
    assert split.verdict != Verdict.SUSPICIOUS_INDICATION
    assert uniform.verdict != Verdict.SUSPICIOUS_INDICATION

    split_frac = float(split.evidence["outlier_frac"])
    uniform_frac = float(uniform.evidence["outlier_frac"])

    # Locked fractions — the naive split > uniform order fails on this seed.
    assert split_frac == _SPLIT_LOCKED_FRAC, (
        f"split outlier_frac changed: {split_frac} != {_SPLIT_LOCKED_FRAC}"
    )
    assert uniform_frac == _UNIFORM_LOCKED_FRAC, (
        f"uniform outlier_frac changed: {uniform_frac} != {_UNIFORM_LOCKED_FRAC}"
    )
    assert split_frac < uniform_frac, (
        f"locked order changed: split={split_frac}, uniform={uniform_frac}"
    )
