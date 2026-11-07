"""Course lock: SSIM is undefined below the 11-pixel window."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_metrics import ssim


def test_ssim_below_window_is_none() -> None:
    field = np.full((8, 8), 10, np.uint8)
    assert ssim(field, field.copy()) is None
