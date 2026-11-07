"""Course lock: a zero-sum Wiener kernel is rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import restore


def test_zero_sum_wiener_kernel_is_rejected() -> None:
    image = np.zeros((4, 4), np.uint8)
    params = {"kernel_role": "oracle", "nsr": 0.01, "kernel": [[0.0]]}
    with pytest.raises(ValueError) as exc:
        restore(image, "wiener", params)
    assert str(exc.value) == "kernel must have a finite positive sum"
