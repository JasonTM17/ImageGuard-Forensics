"""Course lock: a one-value CLAHE tile size is rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import restore


def test_one_value_clahe_tile_is_rejected() -> None:
    image = np.zeros((8, 8, 3), np.uint8)
    params = {"clip_limit": 1.0, "tile_grid_size": (2,)}
    with pytest.raises(ValueError) as exc:
        restore(image, "clahe", params)
    assert str(exc.value) == "tile_grid_size must have two values"
