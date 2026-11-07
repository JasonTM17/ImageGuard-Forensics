"""Course lock: a JPEG quality above 100 is rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import degrade


def test_jpeg_quality_above_100_is_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        degrade(np.zeros((4, 4), np.uint8), "jpeg", {"quality": 101}, None)
    assert str(exc.value) == "quality must be in [0, 100]"
