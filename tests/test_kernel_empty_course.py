"""Course lock: an empty Wiener kernel is rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import restore


def test_empty_wiener_kernel_is_rejected() -> None:
    image = np.zeros((4, 4), np.uint8)
    params = {"kernel_role": "oracle", "nsr": 0.01, "kernel": []}
    with pytest.raises(ValueError) as exc:
        restore(image, "wiener", params)
    assert str(exc.value) == "kernel must be a non-empty 2D array"
