"""Course lock: PSNR is undefined when the shapes differ."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_metrics import psnr


def test_psnr_shape_mismatch_is_none() -> None:
    assert psnr(np.zeros((4, 4), np.uint8), np.zeros((4, 5), np.uint8)) is None
