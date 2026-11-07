"""Course lock: an unknown kernel role is rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import restore


def test_unknown_kernel_role_is_rejected() -> None:
    image = np.zeros((4, 4), np.uint8)
    params = {"kernel_role": "other", "nsr": 0.01, "kernel": [[1.0]]}
    with pytest.raises(ValueError) as exc:
        restore(image, "wiener", params)
    assert str(exc.value) == "kernel_role must be 'oracle', 'wrong', or 'estimated'"
