"""Course lock: a negative Wiener noise ratio is rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import restore


def test_negative_wiener_nsr_is_rejected() -> None:
    image = np.zeros((4, 4), np.uint8)
    params = {"kernel_role": "oracle", "nsr": -0.1, "kernel": [[1.0]]}
    with pytest.raises(ValueError) as exc:
        restore(image, "wiener", params)
    assert str(exc.value) == "nsr must be at least 0.0"
