"""Course lock: NoiseLocal flags the smoother half of a noise field.

Seed 20261106 — left half strong Gaussian noise, right half the same noise
after a heavy blur. NoiseLocal must stay below SUSPICIOUS_INDICATION and put
strictly more mask mass on the smooth half than on the noisy half.
"""

from __future__ import annotations

import cv2
import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.noise_local import NoiseLocal

_SEED = 20261106
_IMG = 128
_SIGMA = 40.0
_BLUR_SIGMA = 4.0


def _doc_from(bgr: np.ndarray) -> ImageDocument:
    h, w = bgr.shape[:2]
    return ImageDocument(
        array=bgr,
        width=w,
        height=h,
        channels=3,
        image_format="png",
        sha256="l" * 64,
        file_bytes=0,
        file_name="noise_local_halves.png",
    )


def _split_noise_then_blur() -> np.ndarray:
    rng = np.random.default_rng(_SEED)
    h = w = _IMG
    half = w // 2
    noise = rng.normal(0.0, _SIGMA, (h, half)).astype(np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[:, :half] = noise[:, :, None]
    img[:, half:] = cv2.GaussianBlur(noise, (0, 0), _BLUR_SIGMA)[:, :, None]
    return np.clip(128.0 + img, 0, 255).astype(np.uint8)


def test_noise_local_mask_mass_on_smooth_half(ts) -> None:
    img = _split_noise_then_blur()
    result = NoiseLocal().analyze(_doc_from(img), ts)

    assert result.verdict != Verdict.SUSPICIOUS_INDICATION
    assert result.mask is not None

    half = _IMG // 2
    noisy_mass = int(np.count_nonzero(result.mask[:, :half]))
    smooth_mass = int(np.count_nonzero(result.mask[:, half:]))
    assert smooth_mass > noisy_mass, (
        "NoiseLocal must put more mask mass on the blurred half: "
        f"smooth={smooth_mass}, noisy={noisy_mass}"
    )
