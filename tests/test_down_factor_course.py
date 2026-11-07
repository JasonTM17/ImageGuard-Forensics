"""Course lock: a downsample factor that erases the image is rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import degrade


def test_downsample_factor_must_leave_a_pixel() -> None:
    image = np.zeros((4, 4), np.uint8)
    with pytest.raises(ValueError) as exc:
        degrade(image, "downsample", {"factor": 8}, None)
    assert str(exc.value) == "factor must leave a non-zero output dimension"
