"""Course lock: an empty image is rejected before degradation."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import degrade


def test_empty_image_is_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        degrade(np.zeros((0, 0), np.uint8), "disk", {"radius": 0}, None)
    assert str(exc.value) == "image must be a non-empty 2D or 3D array"
