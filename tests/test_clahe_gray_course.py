"""Course lock: CLAHE rejects a single-channel image."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import restore


def test_gray_clahe_is_rejected() -> None:
    image = np.zeros((4, 4), np.uint8)
    params = {"clip_limit": 1.0, "tile_grid_size": (2, 2)}
    with pytest.raises(ValueError) as exc:
        restore(image, "clahe", params)
    assert str(exc.value) == "clahe expects a three-channel BGR image"
