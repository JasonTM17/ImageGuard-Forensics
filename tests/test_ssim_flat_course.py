"""Course lock: identical fields have SSIM 1 and a shape mismatch is undefined."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_metrics import ssim


def test_identical_field_ssim_is_one_and_mismatch_is_none() -> None:
    field = np.full((12, 12), 40, np.uint8)
    assert ssim(field, field.copy()) == 1.0
    assert ssim(field, field[:11]) is None
