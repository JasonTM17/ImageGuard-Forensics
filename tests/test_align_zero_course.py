"""Course lock: a non-positive alignment height is rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_metrics import align_mask


def test_nonpositive_alignment_height_is_rejected() -> None:
    mask = np.zeros((2, 2), np.uint8)
    with pytest.raises(ValueError) as exc:
        align_mask(mask, (0, 4))
    assert str(exc.value) == "target height and width must be positive"
