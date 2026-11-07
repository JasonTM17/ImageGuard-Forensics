"""Course lock: identical images have no PSNR and a one-level field is fixed."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_metrics import psnr


def test_psnr_identity_and_unit_delta_are_locked() -> None:
    zeros = np.zeros((8, 8), np.uint8)
    ones = np.full((8, 8), 1, np.uint8)
    assert psnr(zeros, zeros.copy()) is None
    assert psnr(zeros, ones) == 48.1308036086791
